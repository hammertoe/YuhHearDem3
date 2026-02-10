"""Compare KG extraction between Gemini Flash and one or more Cerebras models.

This is an additive comparison harness. It does NOT modify or write KG rows
to Postgres; it only reads utterances from Postgres, calls the two LLMs,
and writes artifacts (jsonl + report) to an output directory.

Example:
  python scripts/compare_kg_models.py --youtube-video-id Syxyah7QIaM --max-windows 5
  python scripts/compare_kg_models.py --youtube-video-id Syxyah7QIaM --max-windows 5 \
    --cerebras-model gpt-oss-120b
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

from lib.db.postgres_client import PostgresClient
from lib.knowledge_graph.kg_extractor import KGExtractor
from lib.knowledge_graph.model_compare import (
    WindowRunMetrics,
    build_known_speakers_table,
    canonicalize_edges,
    canonicalize_nodes,
    compute_multi_model_report,
    edge_signature_loose,
)
from lib.knowledge_graph.oss_two_pass import (
    RefineMode,
    TwoPassMode,
    ValidationIssue,
    build_oss_additions_prompt,
    build_oss_draft_prompt,
    normalize_utterance_ids_in_data,
    normalize_evidence_in_data,
    merge_oss_additions,
    build_refine_prompt,
    should_run_second_pass,
    validate_kg_llm_data,
)
from lib.knowledge_graph.window_builder import (
    DEFAULT_STRIDE,
    DEFAULT_WINDOW_SIZE,
    ConceptWindow,
    WindowBuilder,
)


def _parse_json_loose(extractor: KGExtractor, text: str) -> dict[str, Any]:
    """Parse JSON response, tolerating non-JSON prefixes/suffixes.

    Cerebras models occasionally prepend short text even when instructed to return JSON.
    We keep production extractor unchanged and only make the comparison harness more robust.
    """
    try:
        return extractor._parse_json_response(text)
    except Exception:
        pass

    s = (text or "").strip()
    if not s:
        raise ValueError("Empty response")

    # Drop common markdown fences.
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()

    # Fast path: first '{' .. last '}'.
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = s[first : last + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Brace-matching to find a top-level JSON object.
    start = s.find("{")
    if start == -1:
        raise

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    return json.loads(candidate)

    raise ValueError("Could not locate valid JSON object in response")


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_CEREBRAS_MODEL = "gpt-oss-120b"
DEFAULT_CEREBRAS_MODELS = ["gpt-oss-120b"]

DEFAULT_OSS_TWO_PASS_MODE = TwoPassMode.ON_LOW_EDGES.value
DEFAULT_OSS_REFINE_MODE = RefineMode.AUDIT_REPAIR.value
DEFAULT_OSS_MIN_EDGES = 4
DEFAULT_OSS_MAX_ADDED_EDGES = 8

DEFAULT_OSS_REASONING_EFFORT = "high"
DEFAULT_OSS_REASONING_FORMAT = "hidden"  # keep JSON clean; use "parsed" for debugging

# When reasoning_effort is high, the model may generate many reasoning tokens.
# Use a higher completion limit to avoid truncated/incomplete JSON.
DEFAULT_OSS_MAX_COMPLETION_TOKENS = 16384


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_filename_component(value: str) -> str:
    cleaned = (value or "").strip()
    out: list[str] = []
    for ch in cleaned:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "model"


class GeminiProvider:
    def __init__(self, model: str, disable_thinking: bool = False):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.disable_thinking = disable_thinking

    def generate(self, prompt: str) -> tuple[float, str]:
        config_args: dict[str, Any] = {
            "temperature": 0.0,
            "response_mime_type": "application/json",
        }
        if self.disable_thinking:
            config_args["thinking_config"] = ThinkingConfig(
                thinking_budget=0, include_thoughts=False
            )

        start = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=GenerateContentConfig(**config_args),
        )
        elapsed = time.perf_counter() - start
        text = response.text
        if text is None:
            raise ValueError("Gemini returned empty response")
        return elapsed, text


class CerebrasProvider:
    def __init__(self, model: str):
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError(
                "CEREBRAS_API_KEY environment variable is not set (load it via --dotenv-path)"
            )
        try:
            from cerebras.cloud.sdk import Cerebras  # type: ignore[import-untyped]
        except Exception as e:
            raise RuntimeError(
                "Missing Cerebras SDK dependency. Install `cerebras-cloud-sdk` "
                "(or use the same venv as ../yuhgettintru)."
            ) from e

        self.client = Cerebras(api_key=api_key)
        self.model = model

    def generate_json(
        self,
        prompt: str,
        *,
        reasoning_effort: str | None = None,
        reasoning_format: str | None = None,
        max_completion_tokens: int = 8192,
    ) -> tuple[float, str, str | None]:
        system_prompt = (
            "You are extracting knowledge graph entities and relationships from parliamentary transcripts. "
            "Return JSON only. Do not include markdown."
        )

        def call(
            *,
            effort: str | None,
            fmt: str | None,
            use_response_format: bool,
        ) -> tuple[float, Any]:
            extra: dict[str, Any] = {}
            if effort is not None:
                extra["reasoning_effort"] = effort
            if fmt is not None:
                extra["reasoning_format"] = fmt

            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_completion_tokens": int(max_completion_tokens),
                **extra,
            }
            if use_response_format:
                kwargs["response_format"] = {"type": "json_object"}

            start = time.perf_counter()
            resp = self.client.chat.completions.create(**kwargs)
            return time.perf_counter() - start, resp

        # Some combinations of reasoning parameters and structured output can
        # intermittently yield empty content. We retry with a slightly reduced
        # reasoning effort, then (if needed) without response_format.
        attempts: list[tuple[str | None, bool]] = []
        if reasoning_effort in {"high", "medium", "low"}:
            attempts.append((reasoning_effort, True))
            if reasoning_effort == "high":
                attempts.append(("medium", True))
            if reasoning_effort in {"high", "medium"}:
                attempts.append(("low", True))
            attempts.append((reasoning_effort, False))
        else:
            attempts.append((reasoning_effort, True))

        last_err: str | None = None
        for effort, use_format in attempts:
            try:
                elapsed, response = call(
                    effort=effort,
                    fmt=reasoning_format,
                    use_response_format=use_format,
                )
                msg = response.choices[0].message
                text = msg.content
                reasoning = getattr(msg, "reasoning", None)
                if text:
                    return elapsed, text, reasoning

                finish = getattr(response.choices[0], "finish_reason", None)
                last_err = f"empty content (finish_reason={finish}, effort={effort}, response_format={use_format})"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"

        raise ValueError(f"Cerebras returned empty response: {last_err}")

    def generate(self, prompt: str) -> tuple[float, str]:
        elapsed, text, _reasoning = self.generate_json(prompt)
        return elapsed, text


def _safe_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=True) + "\n")


def _extract_window(
    *,
    extractor: KGExtractor,
    provider_name: str,
    provider_model: str,
    provider: Any,
    youtube_video_id: str,
    window: ConceptWindow,
    known_nodes_table: str,
    save_prompt: bool,
    save_raw: bool,
) -> WindowRunMetrics:
    prompt = extractor._build_prompt(window, known_nodes_table)

    window_speaker_ids = [sid for sid in window.speaker_ids if sid]

    elapsed_s = 0.0
    raw_text: str | None = None
    parse_success = False
    error: str | None = None
    nodes_new: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    try:
        elapsed_s, raw = provider.generate(prompt)
        raw_text = raw
        data = _parse_json_loose(extractor, raw)

        raw_nodes = data.get("nodes_new", [])
        if isinstance(raw_nodes, list):
            for n in raw_nodes:
                if isinstance(n, dict):
                    nodes_new.append(n)

        utterance_timestamps = {
            u.id: (u.timestamp_str, u.seconds_since_start) for u in window.utterances
        }
        parsed_edges = extractor._parse_edges_from_llm_data(
            data, utterance_timestamps, window
        )
        for e in parsed_edges:
            edges.append(asdict(e))

        parse_success = True

    except Exception as e:
        error = str(e)

    return WindowRunMetrics(
        provider=provider_name,
        model=provider_model,
        youtube_video_id=youtube_video_id,
        window_index=window.window_index,
        window_speaker_ids=window_speaker_ids,
        elapsed_s=elapsed_s,
        parse_success=parse_success,
        error=error,
        raw_response=raw_text if save_raw else None,
        prompt=prompt if save_prompt else None,
        nodes_new=nodes_new,
        edges=edges,
    )


def _extract_window_oss_two_pass(
    *,
    extractor: KGExtractor,
    provider: Any,
    model: str,
    youtube_video_id: str,
    window: ConceptWindow,
    known_nodes_table: str,
    save_prompt: bool,
    save_raw: bool,
    two_pass_mode: TwoPassMode,
    refine_mode: RefineMode,
    min_edges: int,
    max_added_edges: int,
    reasoning_effort: str,
    reasoning_format: str,
    max_completion_tokens: int,
) -> WindowRunMetrics:
    target_edges = len(window.utterances) + 2
    prompt1 = build_oss_draft_prompt(
        window_text=window.text,
        known_nodes_table=known_nodes_table,
        predicates=KGExtractor.PREDICATES,
        node_types=KGExtractor.NODE_TYPES,
        target_edges=target_edges,
    )
    window_speaker_ids = [sid for sid in window.speaker_ids if sid]

    elapsed1 = 0.0
    elapsed2: float | None = None
    raw1: str | None = None
    raw2: str | None = None
    prompt2: str | None = None
    trigger_reason: str | None = None

    pass1_parse_success = False
    pass1_edge_count: int | None = None
    pass1_violations: int | None = None
    issues: list[ValidationIssue] = []

    data_final: dict[str, Any] | None = None
    final_prompt = prompt1
    final_raw: str | None = None

    reasoning1: str | None = None
    reasoning2: str | None = None

    pass1_error: str | None = None
    pass2_error: str | None = None

    draft_json_for_pass2: str | None = None

    try:
        elapsed1, raw, reasoning1 = provider.generate_json(
            prompt1,
            reasoning_effort=reasoning_effort,
            reasoning_format=reasoning_format,
            max_completion_tokens=max_completion_tokens,
        )
        raw1 = raw
        final_raw = raw
        data1 = _parse_json_loose(extractor, raw)
        normalize_utterance_ids_in_data(data1, youtube_video_id=youtube_video_id)
        normalize_evidence_in_data(data1, window_text=window.text)
        draft_json_for_pass2 = json.dumps(data1, indent=2, ensure_ascii=True)
        pass1_parse_success = True

        validation = validate_kg_llm_data(
            data1,
            window_text=window.text,
            window_utterance_ids=set(window.utterance_ids),
            window_speaker_ids=window_speaker_ids,
            allowed_predicates=set(KGExtractor.PREDICATES),
            allowed_node_types=set(KGExtractor.NODE_TYPES),
        )
        pass1_edge_count = validation.edge_count
        pass1_violations = validation.violations_count
        issues = validation.issues

        should2, trigger_reason = should_run_second_pass(
            mode=two_pass_mode,
            pass1_parse_success=True,
            edge_count=validation.edge_count,
            violations_count=validation.violations_count,
            min_edges=min_edges,
        )
        if should2:
            # If pass 1 is already valid, focus pass 2 on recall (additions only).
            if validation.violations_count == 0:
                prompt2 = build_oss_additions_prompt(
                    window_text=window.text,
                    known_nodes_table=known_nodes_table,
                    predicates=KGExtractor.PREDICATES,
                    node_types=KGExtractor.NODE_TYPES,
                    draft_json=draft_json_for_pass2 or raw,
                    target_edges=target_edges,
                    max_added_edges=max_added_edges,
                )
                elapsed2, raw_fix, reasoning2 = provider.generate_json(
                    prompt2,
                    reasoning_effort=reasoning_effort,
                    reasoning_format=reasoning_format,
                    max_completion_tokens=max_completion_tokens,
                )
                raw2 = raw_fix
                additions = _parse_json_loose(extractor, raw_fix)
                normalize_utterance_ids_in_data(
                    additions, youtube_video_id=youtube_video_id
                )
                normalize_evidence_in_data(additions, window_text=window.text)
                data_final = merge_oss_additions(data1, additions)
                final_prompt = prompt2
                final_raw = raw_fix
            else:
                prompt2 = build_refine_prompt(
                    window_text=window.text,
                    known_nodes_table=known_nodes_table,
                    predicates=KGExtractor.PREDICATES,
                    node_types=KGExtractor.NODE_TYPES,
                    draft_json=draft_json_for_pass2 or raw,
                    issues=issues,
                    refine_mode=refine_mode,
                    max_added_edges=max_added_edges,
                )
                elapsed2, raw_fix, reasoning2 = provider.generate_json(
                    prompt2,
                    reasoning_effort=reasoning_effort,
                    reasoning_format=reasoning_format,
                    max_completion_tokens=max_completion_tokens,
                )
                raw2 = raw_fix
                data2 = _parse_json_loose(extractor, raw_fix)
                normalize_utterance_ids_in_data(
                    data2, youtube_video_id=youtube_video_id
                )
                normalize_evidence_in_data(data2, window_text=window.text)
                data_final = data2
                final_prompt = prompt2
                final_raw = raw_fix
        else:
            data_final = data1

    except Exception as e:
        # Pass 1 failed. Optionally try a repair pass that focuses on producing valid JSON.
        pass1_parse_success = False
        pass1_error = str(e)
        issues = [ValidationIssue(code="pass1_error", message=str(e))]
        should2, trigger_reason = should_run_second_pass(
            mode=two_pass_mode,
            pass1_parse_success=False,
            edge_count=0,
            violations_count=1,
            min_edges=min_edges,
        )
        if should2:
            try:
                prompt2 = build_refine_prompt(
                    window_text=window.text,
                    known_nodes_table=known_nodes_table,
                    predicates=KGExtractor.PREDICATES,
                    node_types=KGExtractor.NODE_TYPES,
                    draft_json=raw1 or "",
                    issues=issues,
                    refine_mode=refine_mode,
                    max_added_edges=max_added_edges,
                )
                elapsed2, raw_fix, reasoning2 = provider.generate_json(
                    prompt2,
                    reasoning_effort=reasoning_effort,
                    reasoning_format=reasoning_format,
                    max_completion_tokens=max_completion_tokens,
                )
                raw2 = raw_fix
                data2 = _parse_json_loose(extractor, raw_fix)
                normalize_utterance_ids_in_data(
                    data2, youtube_video_id=youtube_video_id
                )
                normalize_evidence_in_data(data2, window_text=window.text)
                data_final = data2
                final_prompt = prompt2
                final_raw = raw_fix
            except Exception as e2:
                pass2_error = str(e2)
                data_final = None

    elapsed_total = elapsed1 + (elapsed2 or 0.0)

    nodes_new: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    parse_success = False
    error: str | None = None

    if data_final is None:
        parse_success = False
        if pass1_error and pass2_error:
            error = f"both_passes_failed: pass1={pass1_error} pass2={pass2_error}"
        elif pass1_error:
            error = f"both_passes_failed: pass1={pass1_error}"
        else:
            error = "both_passes_failed" if trigger_reason else "pass1_failed"
    else:
        try:
            raw_nodes = data_final.get("nodes_new", [])
            if isinstance(raw_nodes, list):
                for n in raw_nodes:
                    if isinstance(n, dict):
                        nodes_new.append(n)

            utterance_timestamps = {
                u.id: (u.timestamp_str, u.seconds_since_start)
                for u in window.utterances
            }
            parsed_edges = extractor._parse_edges_from_llm_data(
                data_final, utterance_timestamps, window
            )
            for e in parsed_edges:
                edges.append(asdict(e))

            parse_success = True
        except Exception as e:
            parse_success = False
            error = str(e)

    return WindowRunMetrics(
        provider="cerebras",
        model=model,
        youtube_video_id=youtube_video_id,
        window_index=window.window_index,
        window_speaker_ids=window_speaker_ids,
        elapsed_s=elapsed_total,
        parse_success=parse_success,
        error=error,
        raw_response=final_raw if save_raw else None,
        prompt=final_prompt if save_prompt else None,
        nodes_new=nodes_new,
        edges=edges,
        pass1_elapsed_s=elapsed1,
        pass2_elapsed_s=elapsed2,
        pass2_trigger=trigger_reason,
        pass1_parse_success=pass1_parse_success,
        pass1_edge_count=pass1_edge_count,
        pass1_violations_count=pass1_violations,
        prompt_pass1=prompt1 if save_prompt else None,
        prompt_pass2=prompt2 if (save_prompt and prompt2) else None,
        raw_response_pass1=raw1 if save_raw else None,
        raw_response_pass2=raw2 if save_raw else None,
        reasoning_pass1=reasoning1
        if (save_raw and reasoning_format == "parsed")
        else None,
        reasoning_pass2=reasoning2
        if (save_raw and reasoning_format == "parsed")
        else None,
        pass1_error=pass1_error,
        pass2_error=pass2_error,
    )


def _collect_canonical_edges(runs: list[WindowRunMetrics]) -> list[dict[str, Any]]:
    all_edges: list[dict[str, Any]] = []
    for r in runs:
        if not r.parse_success:
            continue
        temp_to_canonical = canonicalize_nodes(r.nodes_new)
        canon = canonicalize_edges(
            r.edges,
            temp_to_canonical=temp_to_canonical,
            window_speaker_ids=r.window_speaker_ids,
        )
        all_edges.extend(canon)
    return all_edges


def _write_markdown_report(
    path: Path,
    *,
    youtube_video_id: str,
    model_keys: list[str],
    report: dict[str, Any],
    runs_by_key: dict[str, list[WindowRunMetrics]],
) -> None:
    edges_by_key: dict[str, list[dict[str, Any]]] = {
        key: _collect_canonical_edges(runs) for key, runs in runs_by_key.items()
    }
    by_sig: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    sigs: dict[str, set[tuple[str, str, str]]] = {}
    for key, edges in edges_by_key.items():
        d: dict[tuple[str, str, str], dict[str, Any]] = {}
        for e in edges:
            d.setdefault(edge_signature_loose(e), e)
        by_sig[key] = d
        sigs[key] = set(d.keys())

    def fmt_edge(e: dict[str, Any]) -> str:
        src = e.get("source_id", "")
        pred = e.get("predicate", "")
        tgt = e.get("target_id", "")
        secs = e.get("earliest_seconds")
        ev = " ".join(str(e.get("evidence", "")).split())
        if len(ev) > 160:
            ev = ev[:157] + "..."
        return f'- {src} --{pred}--> {tgt} (t={secs}) | evidence="{ev}"'

    failures_by_key = {
        key: [r for r in runs if not r.parse_success]
        for key, runs in runs_by_key.items()
    }

    lines: list[str] = []
    lines.append("# KG Model Comparison\n")
    lines.append(f"Video: `{youtube_video_id}`\n")
    lines.append("Models:\n")
    for key in model_keys:
        lines.append(f"- `{key}`")
    lines.append("")
    lines.append("## Metrics\n")
    lines.append("```json")
    lines.append(json.dumps(report, indent=2, ensure_ascii=True))
    lines.append("```\n")

    lines.append("## Per-Model Summary\n")
    lines.append(
        "| Model | Windows | Success | Failed | Nodes | Edges | p50(s) | p95(s) | Avg(s) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in model_keys:
        m = report.get("models", {}).get(key, {})
        lat = m.get("latency_s", {})
        lines.append(
            "| "
            + key
            + " | "
            + str(m.get("windows", 0))
            + " | "
            + str(m.get("success", 0))
            + " | "
            + str(m.get("failed", 0))
            + " | "
            + str(m.get("nodes_new", 0))
            + " | "
            + str(m.get("edges", 0))
            + " | "
            + f"{lat.get('p50', 0.0):.2f}"
            + " | "
            + f"{lat.get('p95', 0.0):.2f}"
            + " | "
            + f"{lat.get('avg', 0.0):.2f}"
            + " |"
        )
    lines.append("")

    lines.append("## Parse Failures\n")
    for key in model_keys:
        lines.append(f"- {key}_failed_windows={len(failures_by_key.get(key, []))}")
    lines.append("")

    for key in model_keys:
        failures = failures_by_key.get(key, [])
        if not failures:
            continue
        lines.append(f"### {key} failures (first 5)\n")
        for r in failures[:5]:
            lines.append(f"- window={r.window_index} error={r.error}")
        lines.append("")

    lines.append("## Edge Overlap (Loose)\n")
    overlap_loose = report.get("overlap", {}).get("edges_loose", {})
    if len(model_keys) >= 2:
        lines.append("| A | B | A edges | B edges | Intersection | Union | Jaccard |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for i, a in enumerate(model_keys):
            for b in model_keys[i + 1 :]:
                stats = (
                    overlap_loose.get(a, {}).get(b)
                    or overlap_loose.get(b, {}).get(a)
                    or {}
                )
                lines.append(
                    f"| {a} | {b} | {stats.get('a', 0)} | {stats.get('b', 0)} | {stats.get('intersection', 0)} | {stats.get('union', 0)} | {stats.get('jaccard', 0.0):.3f} |"
                )
        lines.append("")

    lines.append("## Pairwise Diffs (Samples)\n")
    for i, a in enumerate(model_keys):
        for b in model_keys[i + 1 :]:
            a_only = sorted(sigs[a] - sigs[b])
            b_only = sorted(sigs[b] - sigs[a])

            lines.append(f"### Only In {a} (vs {b})\n")
            if a_only:
                for sig in a_only[:20]:
                    lines.append(fmt_edge(by_sig[a][sig]))
            else:
                lines.append("- (none)")
            lines.append("")

            lines.append(f"### Only In {b} (vs {a})\n")
            if b_only:
                for sig in b_only[:20]:
                    lines.append(fmt_edge(by_sig[b][sig]))
            else:
                lines.append("- (none)")
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare KG extraction: gemini-2.5-flash vs Cerebras model(s)"
    )
    parser.add_argument("--youtube-video-id", required=True, help="YouTube video ID")
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Concept window size in utterances",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Stride in utterances between windows",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Limit number of windows (for quick tests)",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model to use",
    )
    parser.add_argument(
        "--cerebras-model",
        action="append",
        default=None,
        help=(
            "Cerebras model to include (repeatable). "
            "Default includes gpt-oss-120b and zai-glm-4.7"
        ),
    )
    parser.add_argument(
        "--dotenv-path",
        default="../yuhgettintru/.env",
        help="dotenv file to load (for CEREBRAS_API_KEY)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: artifacts/kg_model_compare/<timestamp>)",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Save prompts per window (can be large)",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw model responses per window (can be large)",
    )
    parser.add_argument(
        "--filter-short",
        action="store_true",
        help="Filter out ultra-short utterances when building windows",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable thinking on Gemini (default: enabled)",
    )
    parser.add_argument(
        "--oss-two-pass",
        choices=[m.value for m in TwoPassMode],
        default=DEFAULT_OSS_TWO_PASS_MODE,
        help="Two-pass refinement mode for cerebras/gpt-oss-120b",
    )
    parser.add_argument(
        "--oss-refine-mode",
        choices=[m.value for m in RefineMode],
        default=DEFAULT_OSS_REFINE_MODE,
        help="Second-pass strategy for cerebras/gpt-oss-120b",
    )
    parser.add_argument(
        "--oss-min-edges",
        type=int,
        default=DEFAULT_OSS_MIN_EDGES,
        help="Trigger threshold for on_low_edges mode",
    )
    parser.add_argument(
        "--oss-max-added-edges",
        type=int,
        default=DEFAULT_OSS_MAX_ADDED_EDGES,
        help="Max additional edges allowed in pass 2",
    )
    parser.add_argument(
        "--oss-reasoning-effort",
        choices=["low", "medium", "high"],
        default=DEFAULT_OSS_REASONING_EFFORT,
        help="Cerebras reasoning_effort for gpt-oss-120b",
    )
    parser.add_argument(
        "--oss-reasoning-format",
        choices=["hidden", "parsed"],
        default=DEFAULT_OSS_REASONING_FORMAT,
        help="Cerebras reasoning_format for gpt-oss-120b",
    )
    parser.add_argument(
        "--oss-max-completion-tokens",
        type=int,
        default=DEFAULT_OSS_MAX_COMPLETION_TOKENS,
        help="Max completion tokens for gpt-oss-120b calls (helps avoid truncated JSON)",
    )
    args = parser.parse_args()

    youtube_video_id = args.youtube_video_id

    dotenv_path = Path(args.dotenv_path)
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(
            "artifacts"  # local artifacts only
        )
        / "kg_model_compare"
        / _utc_slug()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("KG Model Comparison")
    print(f"Video: {youtube_video_id}")
    print(
        f"Gemini: {args.gemini_model} (thinking={'disabled' if args.disable_thinking else 'enabled'})"
    )
    cerebras_models = args.cerebras_model or DEFAULT_CEREBRAS_MODELS
    print("Cerebras models: " + ", ".join(cerebras_models))
    print(
        "OSS two-pass: "
        + f"mode={args.oss_two_pass} refine_mode={args.oss_refine_mode} "
        + f"min_edges={args.oss_min_edges} max_added_edges={args.oss_max_added_edges} "
        + f"reasoning_effort={args.oss_reasoning_effort} reasoning_format={args.oss_reasoning_format} "
        + f"max_completion_tokens={args.oss_max_completion_tokens}"
    )
    print(f"Artifacts: {out_dir}")
    print("=" * 60)

    # Use KGExtractor helpers without initializing API clients.
    extractor = KGExtractor.__new__(KGExtractor)

    gemini = GeminiProvider(args.gemini_model, disable_thinking=args.disable_thinking)
    cerebras_providers = {m: CerebrasProvider(m) for m in cerebras_models}

    oss_two_pass_mode = TwoPassMode(args.oss_two_pass)
    oss_refine_mode = RefineMode(args.oss_refine_mode)

    model_keys: list[str] = []
    runs_by_key: dict[str, list[WindowRunMetrics]] = {}
    jsonl_by_key: dict[str, Path] = {}

    gemini_key = f"gemini/{args.gemini_model}"
    model_keys.append(gemini_key)
    runs_by_key[gemini_key] = []
    jsonl_by_key[gemini_key] = out_dir / "gemini.jsonl"

    for m in cerebras_models:
        c_key = f"cerebras/{m}"
        model_keys.append(c_key)
        runs_by_key[c_key] = []
        jsonl_by_key[c_key] = out_dir / f"cerebras_{_safe_filename_component(m)}.jsonl"

    with PostgresClient() as pg:
        window_builder = WindowBuilder(pg, embedding_client=None)
        windows = window_builder.build_all_windows(
            youtube_video_id,
            window_size=args.window_size,
            stride=args.stride,
            filter_short=args.filter_short,
        )
        concept_windows = [w for w in windows if isinstance(w, ConceptWindow)]

        if args.max_windows is not None:
            concept_windows = concept_windows[: args.max_windows]

        print(f"Windows: {len(concept_windows)}")

        for i, window in enumerate(concept_windows, 1):
            speaker_ids = [sid for sid in window.speaker_ids if sid]
            known_nodes_table = build_known_speakers_table(pg, speaker_ids)

            print(f"[{i}/{len(concept_windows)}] window={window.window_index} ...")

            g_run = _extract_window(
                extractor=extractor,
                provider_name="gemini",
                provider_model=args.gemini_model,
                provider=gemini,
                youtube_video_id=youtube_video_id,
                window=window,
                known_nodes_table=known_nodes_table,
                save_prompt=args.save_prompts,
                save_raw=args.save_raw,
            )
            runs_by_key[gemini_key].append(g_run)
            _append_jsonl(jsonl_by_key[gemini_key], asdict(g_run))

            g_status = "OK" if g_run.parse_success else "FAIL"
            print(
                f"  {gemini_key}={g_status} ({g_run.elapsed_s:.2f}s) nodes={len(g_run.nodes_new)} edges={len(g_run.edges)}"
            )

            for model, provider in cerebras_providers.items():
                c_key = f"cerebras/{model}"
                if model == "gpt-oss-120b" and oss_two_pass_mode != TwoPassMode.NONE:
                    c_run = _extract_window_oss_two_pass(
                        extractor=extractor,
                        provider=provider,
                        model=model,
                        youtube_video_id=youtube_video_id,
                        window=window,
                        known_nodes_table=known_nodes_table,
                        save_prompt=args.save_prompts,
                        save_raw=args.save_raw,
                        two_pass_mode=oss_two_pass_mode,
                        refine_mode=oss_refine_mode,
                        min_edges=args.oss_min_edges,
                        max_added_edges=args.oss_max_added_edges,
                        reasoning_effort=args.oss_reasoning_effort,
                        reasoning_format=args.oss_reasoning_format,
                        max_completion_tokens=args.oss_max_completion_tokens,
                    )
                else:
                    c_run = _extract_window(
                        extractor=extractor,
                        provider_name="cerebras",
                        provider_model=model,
                        provider=provider,
                        youtube_video_id=youtube_video_id,
                        window=window,
                        known_nodes_table=known_nodes_table,
                        save_prompt=args.save_prompts,
                        save_raw=args.save_raw,
                    )
                runs_by_key[c_key].append(c_run)
                _append_jsonl(jsonl_by_key[c_key], asdict(c_run))

                c_status = "OK" if c_run.parse_success else "FAIL"
                if c_run.model == "gpt-oss-120b" and c_run.pass2_trigger:
                    p1 = c_run.pass1_elapsed_s or 0.0
                    p2 = c_run.pass2_elapsed_s or 0.0
                    e1 = c_run.pass1_edge_count
                    e2 = len(c_run.edges)
                    v1 = c_run.pass1_violations_count
                    print(
                        f"  {c_key}={c_status} total={c_run.elapsed_s:.2f}s (p1={p1:.2f}s p2={p2:.2f}s trigger={c_run.pass2_trigger}) nodes={len(c_run.nodes_new)} edges={e2} (pass1={e1} delta={None if e1 is None else (e2 - e1)}) viol_pass1={v1}"
                    )
                else:
                    print(
                        f"  {c_key}={c_status} ({c_run.elapsed_s:.2f}s) nodes={len(c_run.nodes_new)} edges={len(c_run.edges)}"
                    )

    report = compute_multi_model_report(runs_by_key)

    # Add oss two-pass summary if present.
    oss_key = "cerebras/gpt-oss-120b"
    oss_runs = runs_by_key.get(oss_key, [])
    oss_with_pass2 = [r for r in oss_runs if r.pass2_trigger]
    if oss_runs:
        pass2_times = [
            r.pass2_elapsed_s for r in oss_with_pass2 if r.pass2_elapsed_s is not None
        ]
        deltas = []
        for r in oss_with_pass2:
            if r.pass1_edge_count is None:
                continue
            deltas.append(len(r.edges) - r.pass1_edge_count)

        report["oss_two_pass"] = {
            "model_key": oss_key,
            "mode": args.oss_two_pass,
            "refine_mode": args.oss_refine_mode,
            "min_edges": args.oss_min_edges,
            "max_added_edges": args.oss_max_added_edges,
            "windows": len(oss_runs),
            "pass2_triggered": len(oss_with_pass2),
            "pass2_trigger_rate": (len(oss_with_pass2) / len(oss_runs))
            if oss_runs
            else 0.0,
            "avg_pass2_s": (sum(pass2_times) / len(pass2_times))
            if pass2_times
            else 0.0,
            "avg_edge_delta": (sum(deltas) / len(deltas)) if deltas else 0.0,
            "edge_delta": {
                "min": min(deltas) if deltas else 0,
                "max": max(deltas) if deltas else 0,
            },
        }
    _safe_write_json(out_dir / "report.json", report)

    meta = {
        "youtube_video_id": youtube_video_id,
        "window_size": args.window_size,
        "stride": args.stride,
        "max_windows": args.max_windows,
        "gemini_model": args.gemini_model,
        "gemini_disable_thinking": args.disable_thinking,
        "cerebras_models": cerebras_models,
        "oss_two_pass": {
            "enabled_for_model": "gpt-oss-120b",
            "mode": args.oss_two_pass,
            "refine_mode": args.oss_refine_mode,
            "min_edges": args.oss_min_edges,
            "max_added_edges": args.oss_max_added_edges,
            "reasoning_effort": args.oss_reasoning_effort,
            "reasoning_format": args.oss_reasoning_format,
            "max_completion_tokens": args.oss_max_completion_tokens,
        },
        "dotenv_path": str(args.dotenv_path),
        "generated_at_utc": _utc_slug(),
    }
    _safe_write_json(out_dir / "meta.json", meta)

    _write_markdown_report(
        out_dir / "report.md",
        youtube_video_id=youtube_video_id,
        model_keys=model_keys,
        report=report,
        runs_by_key=runs_by_key,
    )

    print("=" * 60)
    print("Done")
    for key in model_keys:
        print(f"- {jsonl_by_key[key]}")
    print(f"- {out_dir / 'report.json'}")
    print(f"- {out_dir / 'report.md'}")
    print(f"- {out_dir / 'meta.json'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
