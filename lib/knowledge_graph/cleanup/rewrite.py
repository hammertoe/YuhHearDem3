"""KG cleanup edge rewriting and deduplication."""

from typing import Any

from lib.knowledge_graph.cleanup.contracts import is_discourse_predicate


def rewrite_edge_endpoints(
    edges: list[dict[str, Any]], merge_map: dict[str, str]
) -> list[dict[str, Any]]:
    """Rewrite edge endpoints via merge map."""
    rewritten = []
    for edge in edges:
        new_edge = edge.copy()
        source_id = edge.get("source_id")
        target_id = edge.get("target_id")

        if source_id in merge_map:
            new_edge["source_id"] = merge_map[source_id]
        if target_id in merge_map:
            new_edge["target_id"] = merge_map[target_id]

        rewritten.append(new_edge)
    return rewritten


def drop_invalid_edges(edges: list[dict[str, Any]], valid_nodes: set[str]) -> list[dict[str, Any]]:
    """Drop edges where endpoints are missing."""
    valid = []
    for edge in edges:
        source_id = edge.get("source_id")
        target_id = edge.get("target_id")
        if source_id in valid_nodes and target_id in valid_nodes:
            valid.append(edge)
    return valid


def filter_discourse_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter discourse edges to Person->Person only."""
    filtered = []
    for edge in edges:
        predicate = edge.get("predicate")
        source_type = edge.get("source_type")
        target_type = edge.get("target_type")

        if not predicate or not is_discourse_predicate(predicate):
            filtered.append(edge)
            continue

        if source_type == "foaf:Person" and target_type == "foaf:Person":
            filtered.append(edge)

    return filtered


def aggregate_provenance(edges: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate provenance from multiple edges.

    Aggregates:
    - earliest_timestamp_str: min
    - earliest_seconds: min
    - utterance_ids: stable union
    - evidence: longest anchored evidence
    - confidence: max + support-aware blend
    - support_count: number of merged rows
    """
    if not edges:
        return {}

    earliest_seconds = min(
        e.get("earliest_seconds", float("inf"))
        for e in edges
        if e.get("earliest_seconds") is not None
    )
    earliest_edge = min(edges, key=lambda e: e.get("earliest_seconds", float("inf")))

    all_utterance_ids: set[str] = set()
    for edge in edges:
        utterance_ids = edge.get("utterance_ids", [])
        all_utterance_ids.update(utterance_ids)

    best_evidence = max(edges, key=lambda e: len(e.get("evidence", "")))
    evidence = best_evidence.get("evidence", "")

    max_confidence = max((e.get("confidence", 0.0) for e in edges), default=0.0)
    max_edge_weight = max((e.get("edge_weight", 0.0) for e in edges), default=0.0)
    max_edge_rank_score = max((e.get("edge_rank_score", 0.0) for e in edges), default=0.0)

    support_count = len(edges)
    blended_confidence = max_confidence

    return {
        "earliest_timestamp_str": earliest_edge.get("earliest_timestamp_str"),
        "earliest_seconds": earliest_seconds,
        "utterance_ids": sorted(list(all_utterance_ids)),
        "evidence": evidence,
        "confidence": blended_confidence,
        "support_count": str(support_count),
        "edge_weight": str(max_edge_weight),
        "edge_rank_score": str(max_edge_rank_score),
    }


def collapse_duplicate_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse duplicate edges by (source_id, predicate, target_id, youtube_video_id)."""
    groups: dict[tuple, list[dict[str, Any]]] = {}

    for edge in edges:
        key = (
            edge.get("source_id"),
            edge.get("predicate"),
            edge.get("target_id"),
            edge.get("youtube_video_id"),
        )
        if key not in groups:
            groups[key] = []
        groups[key].append(edge)

    collapsed = []
    for key, edge_group in groups.items():
        if len(edge_group) == 1:
            edge = edge_group[0].copy()
            edge.setdefault("support_count", "1")
            edge.setdefault("edge_weight", "0.0")
            edge.setdefault("edge_rank_score", "0.0")
            collapsed.append(edge)
        else:
            aggregated = aggregate_provenance(edge_group)
            collapsed.append(
                {
                    "id": f"collapsed_{len(collapsed)}",
                    "source_id": edge_group[0].get("source_id"),
                    "predicate": edge_group[0].get("predicate"),
                    "target_id": edge_group[0].get("target_id"),
                    "youtube_video_id": edge_group[0].get("youtube_video_id"),
                    **aggregated,
                }
            )

    return collapsed


def rewrite_and_clean_edges(
    edges: list[dict[str, Any]],
    merge_map: dict[str, str],
    valid_nodes: set[str],
) -> dict[str, Any]:
    """Full pipeline: rewrite endpoints, drop invalid, filter discourse, collapse duplicates.

    Returns dict with:
    - dropped_edges: list of dropped edge IDs with reasons
    - clean_edges: list of cleaned edges
    """
    dropped_edges: list[tuple[str, str]] = []

    rewritten = rewrite_edge_endpoints(edges, merge_map)

    filtered = []
    for edge in rewritten:
        source_id = edge.get("source_id")
        target_id = edge.get("target_id")
        if source_id in valid_nodes and target_id in valid_nodes:
            filtered.append(edge)
        else:
            dropped_edges.append((str(edge.get("id", "")), "invalid_endpoint"))

    filtered_discourse = []
    for edge in filtered:
        predicate = edge.get("predicate")
        source_type = edge.get("source_type")
        target_type = edge.get("target_type")

        if not predicate or not is_discourse_predicate(predicate):
            filtered_discourse.append(edge)
            continue

        if source_type == "foaf:Person" and target_type == "foaf:Person":
            filtered_discourse.append(edge)
        else:
            dropped_edges.append((str(edge.get("id", "")), "discourse_constraint"))

    collapsed = collapse_duplicate_edges(filtered_discourse)

    return {
        "dropped_edges": dropped_edges,
        "clean_edges": collapsed,
    }
