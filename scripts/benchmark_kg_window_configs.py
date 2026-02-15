"""Benchmark KG extraction with different window configurations.

This script compares different window_size/stride configurations to evaluate
token efficiency and extraction quality.

Example:
  python scripts/benchmark_kg_window_configs.py --youtube-video-id Syxyah7QIaM --max-windows 12
  python scripts/benchmark_kg_window_configs.py --youtube-video-id Syxyah7QIaM --configs 10-6 20-12 30-18 --max-windows 12
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from dotenv import load_dotenv

from cerebras.cloud.sdk import Cerebras  # type: ignore[import-untyped]
from cerebras.cloud.sdk.types.chat.chat_completion import (
    ChatCompletionResponseUsage,
)

from lib.db.postgres_client import PostgresClient
from lib.knowledge_graph.kg_extractor import KGExtractor
from lib.knowledge_graph.model_compare import build_known_speakers_table
from lib.knowledge_graph.oss_two_pass import (
    RefineMode,
    TwoPassMode,
    build_oss_additions_prompt,
    build_oss_draft_prompt,
    build_refine_prompt,
    merge_oss_additions,
    normalize_evidence_in_data,
    normalize_utterance_ids_in_data,
    validate_kg_llm_data,
)
from lib.knowledge_graph.window_benchmark import (
    BenchmarkConfig,
    WindowBenchmarkMetrics,
    aggregate_metrics,
    compute_comparison_report,
)
from lib.knowledge_graph.window_builder import ConceptWindow, WindowBuilder


DEFAULT_MODEL = "gpt-oss-120b"
DEFAULT_CONFIGS = [
    (10, 6),  # baseline
    (20, 12),  # 2x window
    (30, 18),  # 3x window
]

DEFAULT_OSS_REASONING_EFFORT = "medium"
DEFAULT_OSS_REASONING_FORMAT = "hidden"
DEFAULT_OSS_MAX_COMPLETION_TOKENS = 16384
DEFAULT_OSS_TWO_PASS = TwoPassMode.ALWAYS
DEFAULT_OSS_REFINE_MODE = RefineMode.AUDIT_REPAIR
DEFAULT_OSS_MIN_EDGES = 4
DEFAULT_OSS_MAX_ADDED_EDGES = 8

PREDICATES = [
    "AMENDS",
    "GOVERNS",
    "MODERNIZES",
    "AIMS_TO_REDUCE",
    "REQUIRES_APPROVAL",
    "IMPLEMENTED_BY",
    "RESPONSIBLE_FOR",
    "ASSOCIATED_WITH",
    "CAUSES",
    "ADDRESSES",
    "PROPOSES",
    "RESPONDS_TO",
    "AGREES_WITH",
    "DISAGREES_WITH",
    "QUESTIONS",
]

NODE_TYPES = [
    "foaf:Person",
    "schema:Legislation",
    "schema:Organization",
    "schema:Place",
    "skos:Concept",
]


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


def _parse_json_loose(extractor: KGExtractor, text: str) -> dict[str, Any]:
    """Parse JSON response, tolerating non-JSON prefixes/suffixes."""
    try:
        return extractor._parse_json_response(text)
    except Exception:
        pass

    s = (text or "").strip()
    if not s:
        raise ValueError("Empty response")

    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()

    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = s[first : last + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

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


class CerebrasProvider:
    """Simplified Cerebras provider with token tracking."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_OSS_REASONING_EFFORT,
        reasoning_format: str = DEFAULT_OSS_REASONING_FORMAT,
        max_completion_tokens: int = DEFAULT_OSS_MAX_COMPLETION_TOKENS,
    ):
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY environment variable is not set")
        self.client = Cerebras(api_key=api_key)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.reasoning_format = reasoning_format
        self.max_completion_tokens = max_completion_tokens

    def generate_json(
        self,
        prompt: str,
    ) -> tuple[float, str, str | None, ChatCompletionResponseUsage]:
        """Generate JSON with token usage tracking.

        Returns:
            (elapsed_s, content, reasoning, usage)
        """
        system_prompt = (
            "You are extracting knowledge graph entities and relationships from parliamentary transcripts. "
            "Return JSON only. Do not include markdown."
        )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_completion_tokens": self.max_completion_tokens,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_format": self.reasoning_format,
            "response_format": {"type": "json_object"},
        }

        start = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - start

        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning", None)

        usage: ChatCompletionResponseUsage = resp.usage  # type: ignore[assignment]

        return elapsed, content, reasoning, usage


def extract_window_oss_two_pass(
    *,
    extractor: KGExtractor,
    provider: CerebrasProvider,
    youtube_video_id: str,
    window: ConceptWindow,
    known_nodes_table: str,
    save_prompt: bool = False,
    save_raw: bool = False,
) -> WindowBenchmarkMetrics:
    """Extract from a single window with token tracking."""
    target_edges = len(window.utterances) + 2
    prompt1 = build_oss_draft_prompt(
        window_text=window.text,
        known_nodes_table=known_nodes_table,
        predicates=PREDICATES,
        node_types=NODE_TYPES,
        target_edges=target_edges,
    )

    window_speaker_ids = [sid for sid in window.speaker_ids if sid]

    pass1_elapsed = 0.0
    pass1_usage: ChatCompletionResponseUsage | None = None

    issues: list[Any] = []

    data_final: dict[str, Any] | None = None
    pass2_elapsed: float | None = None
    pass2_usage: ChatCompletionResponseUsage | None = None

    pass1_error: str | None = None

    try:
        pass1_elapsed, raw, _reasoning1, pass1_usage = provider.generate_json(prompt1)
        data1 = _parse_json_loose(extractor, raw)
        normalize_utterance_ids_in_data(data1, youtube_video_id=youtube_video_id)
        normalize_evidence_in_data(data1, window_text=window.text)

        validation = validate_kg_llm_data(
            data1,
            window_text=window.text,
            window_utterance_ids=set(window.utterance_ids),
            window_speaker_ids=window_speaker_ids,
            allowed_predicates=set(PREDICATES),
            allowed_node_types=set(NODE_TYPES),
        )
        issues = validation.issues

        should2 = True  # Always run pass 2 in benchmark mode

        if should2:
            if validation.violations_count == 0:
                prompt2 = build_oss_additions_prompt(
                    window_text=window.text,
                    known_nodes_table=known_nodes_table,
                    predicates=PREDICATES,
                    node_types=NODE_TYPES,
                    draft_json=json.dumps(data1, indent=2),
                    target_edges=target_edges,
                    max_added_edges=DEFAULT_OSS_MAX_ADDED_EDGES,
                )
                pass2_elapsed, raw_fix, _reasoning2, pass2_usage = provider.generate_json(prompt2)
                additions = _parse_json_loose(extractor, raw_fix)
                normalize_utterance_ids_in_data(additions, youtube_video_id=youtube_video_id)
                normalize_evidence_in_data(additions, window_text=window.text)
                data_final = merge_oss_additions(data1, additions)
            else:
                prompt2 = build_refine_prompt(
                    window_text=window.text,
                    known_nodes_table=known_nodes_table,
                    predicates=PREDICATES,
                    node_types=NODE_TYPES,
                    draft_json=json.dumps(data1, indent=2),
                    issues=issues,
                    refine_mode=DEFAULT_OSS_REFINE_MODE,
                    max_added_edges=DEFAULT_OSS_MAX_ADDED_EDGES,
                )
                pass2_elapsed, raw_fix, _reasoning2, pass2_usage = provider.generate_json(prompt2)
                data2 = _parse_json_loose(extractor, raw_fix)
                normalize_utterance_ids_in_data(data2, youtube_video_id=youtube_video_id)
                normalize_evidence_in_data(data2, window_text=window.text)
                data_final = data2
        else:
            data_final = data1

    except Exception as e:
        pass1_error = str(e)

    elapsed_total = pass1_elapsed + (pass2_elapsed or 0.0)

    # Aggregate token usage
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    if pass1_usage:
        prompt_tokens += pass1_usage.prompt_tokens or 0
        completion_tokens += pass1_usage.completion_tokens or 0

    if pass2_usage:
        prompt_tokens += pass2_usage.prompt_tokens or 0
        completion_tokens += pass2_usage.completion_tokens or 0
        total_tokens += pass2_usage.total_tokens or 0
    elif pass1_usage:
        total_tokens = pass1_usage.total_tokens or 0

    nodes_count = 0
    edges_count = 0
    parse_success = False
    error: str | None = None

    if data_final is None:
        parse_success = False
        error = pass1_error or "extraction failed"
    else:
        try:
            raw_nodes = data_final.get("nodes_new", [])
            if isinstance(raw_nodes, list):
                nodes_count = len(raw_nodes)

            utterance_timestamps = {
                u.id: (u.timestamp_str, u.seconds_since_start) for u in window.utterances
            }
            parsed_edges = extractor._parse_edges_from_llm_data(
                data_final, utterance_timestamps, window
            )
            edges_count = len(parsed_edges)
            parse_success = True
        except Exception as e:
            parse_success = False
            error = str(e)

    return WindowBenchmarkMetrics(
        window_index=window.window_index,
        window_size=window.window_size,
        stride=window.stride,
        elapsed_s=elapsed_total,
        parse_success=parse_success,
        nodes_count=nodes_count,
        edges_count=edges_count,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens or (prompt_tokens + completion_tokens),
        pass2_triggered=pass2_elapsed is not None and pass2_elapsed > 0,
        pass2_elapsed_s=pass2_elapsed,
        error=error,
    )


def _write_markdown_report(
    path: Path,
    *,
    youtube_video_id: str,
    report: dict[str, Any],
) -> None:
    """Write markdown comparison report."""
    lines: list[str] = []
    lines.append("# Window Config Benchmark Results\n")
    lines.append(f"Video: `{youtube_video_id}`\n")
    lines.append(f"Model: `{DEFAULT_MODEL}`\n")
    lines.append("")

    lines.append("## Configuration Comparison\n")
    lines.append(
        "| Config | Windows | Success | Tokens | Tokens/Window | Edges | Edges/1K Tokens | Avg(s) | p50(s) | p95(s) |"
    )
    lines.append(
        "|:-------|--------:|--------:|-------:|-------------:|------:|---------------:|-------:|-------:|-------:|"
    )

    for key, data in report.get("comparison", {}).items():
        lines.append(
            f"| w{data['window_size']}_s{data['stride']} "
            f"| {data['total_windows']} "
            f"| {data['success_rate']:.0%} "
            f"| {data['total_tokens']:,} "
            f"| {data['tokens_per_window']:.0f} "
            f"| {data['total_edges']} "
            f"| {data['edges_per_1k_tokens']:.2f} "
            f"| {data['avg_elapsed_s']:.2f} "
            f"| {data['p50_elapsed_s']:.2f} "
            f"| {data['p95_elapsed_s']:.2f} |"
        )

    lines.append("")
    lines.append("## Token Efficiency Analysis\n")

    comparison = report.get("comparison", {})
    configs = list(comparison.keys())

    if len(configs) >= 2:
        baseline = comparison[configs[0]]
        lines.append("### Relative to Baseline\n")
        lines.append("| Config | Token Δ | Edge Δ | Edges/1K Tokens Δ | Latency Δ |")
        lines.append("|:-------|--------:|-------:|-----------------:|----------:|")

        for key in configs[1:]:
            data = comparison[key]
            token_ratio = data["total_tokens"] / baseline["total_tokens"]
            edge_ratio = data["total_edges"] / baseline["total_edges"]
            edges_per_1k_ratio = data["edges_per_1k_tokens"] / baseline["edges_per_1k_tokens"]
            latency_ratio = data["avg_elapsed_s"] / baseline["avg_elapsed_s"]

            lines.append(
                f"| w{data['window_size']}_s{data['stride']} "
                f"| {token_ratio:.1f}x "
                f"| {edge_ratio:.1f}x "
                f"| {edges_per_1k_ratio:.1f}x "
                f"| {latency_ratio:.1f}x |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark KG extraction with different window configurations"
    )
    parser.add_argument("--youtube-video-id", required=True, help="YouTube video ID")
    parser.add_argument(
        "--configs",
        default="10-6,20-12,30-18",
        help="Comma-separated window configs (format: size-stride, e.g., 10-6,20-12)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=12,
        help="Max windows per config (default: 12)",
    )
    parser.add_argument(
        "--dotenv-path",
        default="../yuhgettintru/.env",
        help="dotenv file to load (for CEREBRAS_API_KEY)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: artifacts/kg_window_benchmark/<timestamp>)",
    )
    parser.add_argument(
        "--oss-reasoning-effort",
        choices=["low", "medium", "high"],
        default=DEFAULT_OSS_REASONING_EFFORT,
        help="Cerebras reasoning_effort",
    )
    parser.add_argument(
        "--oss-max-completion-tokens",
        type=int,
        default=DEFAULT_OSS_MAX_COMPLETION_TOKENS,
        help="Max completion tokens",
    )
    args = parser.parse_args()

    youtube_video_id = args.youtube_video_id

    dotenv_path = Path(args.dotenv_path)
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("artifacts") / "kg_window_benchmark" / _utc_slug()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse configs
    config_pairs: list[tuple[int, int]] = []
    for part in args.configs.split(","):
        parts = part.strip().split("-")
        if len(parts) == 2:
            try:
                size = int(parts[0])
                stride = int(parts[1])
                config_pairs.append((size, stride))
            except ValueError:
                pass

    if not config_pairs:
        config_pairs = DEFAULT_CONFIGS

    print("=" * 60)
    print("Window Config Benchmark")
    print(f"Video: {youtube_video_id}")
    print(f"Configs: {config_pairs}")
    print(f"Max windows: {args.max_windows}")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Reasoning: {args.oss_reasoning_effort}")
    print(f"Artifacts: {out_dir}")
    print("=" * 60)

    extractor = KGExtractor.__new__(KGExtractor)

    provider = CerebrasProvider(
        reasoning_effort=args.oss_reasoning_effort,
        max_completion_tokens=args.oss_max_completion_tokens,
    )

    all_results: list[WindowBenchmarkMetrics] = []
    all_aggregated: list[Any] = []

    with PostgresClient() as pg:
        window_builder = WindowBuilder(pg, embedding_client=None)

        for window_size, stride in config_pairs:
            config_key = f"w{window_size}_s{stride}"
            print(f"\n{'=' * 60}")
            print(f"Config: {config_key} (size={window_size}, stride={stride})")
            print(f"{'=' * 60}")

            windows = window_builder.build_all_windows(
                youtube_video_id,
                window_size=window_size,
                stride=stride,
                filter_short=True,
            )
            concept_windows = [w for w in windows if isinstance(w, ConceptWindow)]

            if args.max_windows:
                concept_windows = concept_windows[: args.max_windows]

            print(f"Windows: {len(concept_windows)}")

            config_metrics: list[WindowBenchmarkMetrics] = []

            for i, window in enumerate(concept_windows, 1):
                speaker_ids = [sid for sid in window.speaker_ids if sid]
                known_nodes_table = build_known_speakers_table(pg, speaker_ids)

                print(f"[{i}/{len(concept_windows)}] window={window.window_index} ...", end=" ")

                metrics = extract_window_oss_two_pass(
                    extractor=extractor,
                    provider=provider,
                    youtube_video_id=youtube_video_id,
                    window=window,
                    known_nodes_table=known_nodes_table,
                )

                config_metrics.append(metrics)
                all_results.append(metrics)

                status = "OK" if metrics.parse_success else "FAIL"
                print(
                    f"{status} ({metrics.elapsed_s:.2f}s) tokens={metrics.total_tokens} edges={metrics.edges_count}"
                )

            config = BenchmarkConfig(window_size=window_size, stride=stride)
            aggregated = aggregate_metrics(config_metrics, config)
            all_aggregated.append(aggregated)

            print(f"\n{config_key} Summary:")
            print(f"  Windows: {aggregated.total_windows}")
            print(f"  Success: {aggregated.success_rate:.0%}")
            print(f"  Tokens: {aggregated.total_tokens:,} ({aggregated.tokens_per_window:.0f}/win)")
            print(
                f"  Edges: {aggregated.total_edges} ({aggregated.edges_per_1k_tokens:.2f}/1K tokens)"
            )
            print(
                f"  Latency: {aggregated.avg_elapsed_s:.2f}s avg, {aggregated.p50_elapsed_s:.2f}s p50"
            )

    report = compute_comparison_report(all_aggregated)

    meta = {
        "youtube_video_id": youtube_video_id,
        "configs": [
            {"window_size": c.window_size, "stride": c.stride}
            for c in [a.config for a in all_aggregated]
        ],
        "max_windows": args.max_windows,
        "model": DEFAULT_MODEL,
        "reasoning_effort": args.oss_reasoning_effort,
        "max_completion_tokens": args.oss_max_completion_tokens,
        "dotenv_path": str(args.dotenv_path),
        "generated_at_utc": _utc_slug(),
    }

    report_path = out_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    _write_markdown_report(out_dir / "report.md", youtube_video_id=youtube_video_id, report=report)

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print(f"Report: {report_path}")
    print(f"Markdown: {out_dir / 'report.md'}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
