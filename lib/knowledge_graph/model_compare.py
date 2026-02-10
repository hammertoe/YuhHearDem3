"""Helpers for comparing KG extraction across LLM providers.

This module is intentionally pure/side-effect-light so we can unit test
canonicalization and comparison logic without calling external APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lib.id_generators import generate_kg_node_id


@dataclass(frozen=True)
class WindowRunMetrics:
    provider: str
    model: str
    youtube_video_id: str
    window_index: int
    window_speaker_ids: list[str]
    elapsed_s: float
    parse_success: bool
    error: str | None
    raw_response: str | None
    prompt: str | None
    nodes_new: list[dict[str, Any]]
    edges: list[dict[str, Any]]

    # Optional: two-pass metrics (used for cerebras/gpt-oss-120b thinking emulation)
    pass1_elapsed_s: float | None = None
    pass2_elapsed_s: float | None = None
    pass2_trigger: str | None = None
    pass1_parse_success: bool | None = None
    pass1_edge_count: int | None = None
    pass1_violations_count: int | None = None

    prompt_pass1: str | None = None
    prompt_pass2: str | None = None
    raw_response_pass1: str | None = None
    raw_response_pass2: str | None = None

    reasoning_pass1: str | None = None
    reasoning_pass2: str | None = None

    pass1_error: str | None = None
    pass2_error: str | None = None


def build_known_speakers_table(
    postgres: Any,
    speaker_ids: list[str],
) -> str:
    """Build a markdown table for known speaker nodes.

    We avoid embeddings/vector context for the comparison harness. This table
    gives the LLM stable IDs it can cite.
    """
    if not speaker_ids:
        return "| ID | Type | Label | Aliases |\n|---|---|---|---|"

    rows = postgres.execute_query(
        """
        SELECT id, normalized_name, full_name, title
        FROM speakers
        WHERE id = ANY(%s)
        """,
        (speaker_ids,),
    )
    meta_by_id: dict[str, dict[str, str]] = {
        row[0]: {
            "normalized_name": row[1] or "",
            "full_name": row[2] or "",
            "title": row[3] or "",
        }
        for row in rows
    }

    lines = ["| ID | Type | Label | Aliases |", "|---|---|---|---|"]
    for sid in speaker_ids:
        meta = meta_by_id.get(sid, {})
        label = meta.get("full_name") or meta.get("normalized_name") or sid
        aliases = [
            x
            for x in [
                meta.get("full_name"),
                meta.get("normalized_name"),
                meta.get("title"),
            ]
            if x
        ]
        aliases_str = ", ".join(aliases[:3])
        lines.append(f"| speaker_{sid} | foaf:Person | {label} | {aliases_str} |")

    return "\n".join(lines)


def normalize_speaker_ref(ref: str, window_speaker_ids: list[str]) -> str | None:
    """Normalize speaker references to canonical `speaker_{speaker_id}` format."""
    ref = (ref or "").strip()
    if not ref:
        return None

    if ref.startswith("speaker_"):
        sid = ref.removeprefix("speaker_")
        return ref if sid in window_speaker_ids else None

    if ref.startswith("s_"):
        return f"speaker_{ref}" if ref in window_speaker_ids else None

    return ref


def canonicalize_nodes(nodes_new: list[dict[str, Any]]) -> dict[str, str]:
    """Return temp_id -> canonical kg_node_id mapping for nodes_new."""
    temp_to_canonical: dict[str, str] = {}
    for node in nodes_new:
        temp_id = str(node.get("temp_id", "")).strip()
        node_type = str(node.get("type", "")).strip()
        label = str(node.get("label", "")).strip()
        if not temp_id or not node_type or not label:
            continue
        temp_to_canonical[temp_id] = generate_kg_node_id(node_type, label)
    return temp_to_canonical


def canonicalize_edges(
    edges: list[dict[str, Any]],
    *,
    temp_to_canonical: dict[str, str],
    window_speaker_ids: list[str],
) -> list[dict[str, Any]]:
    """Canonicalize edge endpoints and filter invalid speaker refs."""
    out: list[dict[str, Any]] = []

    for e in edges:
        source_ref = normalize_speaker_ref(
            str(e.get("source_ref", "")), window_speaker_ids
        )
        target_ref = normalize_speaker_ref(
            str(e.get("target_ref", "")), window_speaker_ids
        )
        if source_ref is None or target_ref is None:
            continue

        source_id = temp_to_canonical.get(source_ref, source_ref)
        target_id = temp_to_canonical.get(target_ref, target_ref)

        out.append(
            {
                "source_id": source_id,
                "predicate": str(e.get("predicate", "")).strip(),
                "target_id": target_id,
                "earliest_seconds": e.get("earliest_seconds"),
                "utterance_ids": e.get("utterance_ids") or [],
                "evidence": str(e.get("evidence", "")),
            }
        )

    return out


def edge_signature_loose(edge: dict[str, Any]) -> tuple[str, str, str]:
    """Loose signature for overlap comparisons (ignore evidence/timestamp)."""
    return (
        str(edge.get("source_id", "")),
        str(edge.get("predicate", "")),
        str(edge.get("target_id", "")),
    )


def edge_signature_strict(edge: dict[str, Any]) -> tuple[str, str, str, int, str]:
    """Stricter signature (includes earliest_seconds and evidence)."""
    secs = edge.get("earliest_seconds")
    secs_int = int(secs) if isinstance(secs, int) else -1
    evidence = " ".join(str(edge.get("evidence", "")).split())
    return (
        str(edge.get("source_id", "")),
        str(edge.get("predicate", "")),
        str(edge.get("target_id", "")),
        secs_int,
        evidence,
    )


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def compute_comparison_report(
    *,
    gemini_runs: list[WindowRunMetrics],
    cerebras_runs: list[WindowRunMetrics],
) -> dict[str, Any]:
    """Compute a comparison report across two providers."""
    runs_by_model_key = {
        "gemini": gemini_runs,
        "cerebras": cerebras_runs,
    }
    return compute_multi_model_report(runs_by_model_key)


def compute_multi_model_report(
    runs_by_model_key: dict[str, list[WindowRunMetrics]],
) -> dict[str, Any]:
    """Compute metrics and overlap for an arbitrary set of model runs.

    Args:
        runs_by_model_key: Mapping like {"gemini/gemini-2.5-flash": [...], "cerebras/gpt-oss-120b": [...]}
    """
    report: dict[str, Any] = {
        "models": {},
        "overlap": {
            "edges_loose": {},
            "edges_strict": {},
        },
    }

    def provider_stats(runs: list[WindowRunMetrics]) -> dict[str, Any]:
        elapsed = [r.elapsed_s for r in runs if r.elapsed_s is not None]
        successes = [r for r in runs if r.parse_success]
        failures = [r for r in runs if not r.parse_success]
        nodes_new = sum(len(r.nodes_new or []) for r in successes)
        edges = sum(len(r.edges or []) for r in successes)
        return {
            "windows": len(runs),
            "success": len(successes),
            "failed": len(failures),
            "nodes_new": nodes_new,
            "edges": edges,
            "latency_s": {
                "avg": sum(elapsed) / len(elapsed) if elapsed else 0.0,
                "p50": _percentile(elapsed, 50),
                "p95": _percentile(elapsed, 95),
                "max": max(elapsed) if elapsed else 0.0,
            },
        }

    for key, runs in runs_by_model_key.items():
        report["models"][key] = provider_stats(runs)

    def collect_signatures(
        runs: list[WindowRunMetrics],
        *,
        strict: bool,
    ) -> set[tuple]:
        sigs: set[tuple] = set()
        for r in runs:
            if not r.parse_success:
                continue
            temp_to_canonical = canonicalize_nodes(r.nodes_new)
            canon_edges = canonicalize_edges(
                r.edges,
                temp_to_canonical=temp_to_canonical,
                window_speaker_ids=r.window_speaker_ids,
            )
            for e in canon_edges:
                sigs.add(
                    edge_signature_strict(e) if strict else edge_signature_loose(e)
                )
        return sigs

    sigs_loose: dict[str, set[tuple]] = {}
    sigs_strict: dict[str, set[tuple]] = {}
    for key, runs in runs_by_model_key.items():
        sigs_loose[key] = collect_signatures(runs, strict=False)
        sigs_strict[key] = collect_signatures(runs, strict=True)

    def overlap(a: set[tuple], b: set[tuple]) -> dict[str, Any]:
        inter = a & b
        union = a | b
        return {
            "a": len(a),
            "b": len(b),
            "intersection": len(inter),
            "union": len(union),
            "jaccard": (len(inter) / len(union)) if union else 0.0,
        }

    keys = sorted(runs_by_model_key.keys())
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            report["overlap"]["edges_loose"].setdefault(a, {})[b] = overlap(
                sigs_loose[a], sigs_loose[b]
            )
            report["overlap"]["edges_strict"].setdefault(a, {})[b] = overlap(
                sigs_strict[a], sigs_strict[b]
            )

    return report
