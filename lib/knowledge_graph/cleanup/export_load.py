"""KG cleanup export/load functions."""

import csv
import json
from pathlib import Path
from typing import Any

from lib.knowledge_graph.cleanup.contracts import (
    ALLOWED_NODE_TYPES,
    ALLOWED_PREDICATES,
    DISCOURSE_PREDICATES,
)


def export_to_csv(data: list[dict[str, Any]], output_path: Path, fieldnames: list[str]) -> None:
    """Export data to CSV file."""
    if not data:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def export_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    """Export metrics to JSON file."""
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def export_all_artifacts(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    aliases: list[dict[str, Any]],
    merge_map: dict[str, str],
    edge_drops: list[tuple[str, str]],
    metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    """Export all cleanup artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    export_to_csv(
        nodes,
        output_dir / "kg_nodes_clean.csv",
        [
            "id",
            "label",
            "type",
            "aliases",
            "pagerank_score",
            "merge_cluster_id",
            "merged_from_count",
        ],
    )

    export_to_csv(
        edges,
        output_dir / "kg_edges_clean.csv",
        [
            "id",
            "source_id",
            "predicate",
            "target_id",
            "youtube_video_id",
            "earliest_timestamp_str",
            "earliest_seconds",
            "utterance_ids",
            "evidence",
            "confidence",
            "support_count",
            "edge_weight",
            "edge_rank_score",
        ],
    )

    export_to_csv(aliases, output_dir / "kg_aliases_clean.csv", ["alias_norm", "node_id"])

    merge_map_rows = [{"old_node_id": old, "new_node_id": new} for old, new in merge_map.items()]
    export_to_csv(merge_map_rows, output_dir / "node_merge_map.csv", ["old_node_id", "new_node_id"])

    edge_drop_rows = [{"edge_id": edge_id, "reason": reason} for edge_id, reason in edge_drops]
    export_to_csv(edge_drop_rows, output_dir / "edge_drop_log.csv", ["edge_id", "reason"])

    export_metrics(metrics, output_dir / "metrics_after.json")


def load_nodes_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Load nodes from CSV file."""
    nodes = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            nodes.append(dict(row))
    return nodes


def load_edges_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Load edges from CSV file."""
    edges = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append(dict(row))
    return edges


def load_aliases_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Load aliases from CSV file."""
    aliases = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            aliases.append(dict(row))
    return aliases


def load_merge_map_from_csv(csv_path: Path) -> dict[str, str]:
    """Load merge map from CSV file."""
    merge_map = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_id = row["old_node_id"]
            new_id = row["new_node_id"]
            merge_map[old_id] = new_id
    return merge_map


def verify_loaded_data(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Verify loaded data consistency."""
    errors: list[str] = []
    node_ids = {n["id"] for n in nodes}
    node_type_by_id = {n["id"]: n.get("type") for n in nodes}

    expected_nodes = metrics.get("nodes_after")
    expected_edges = metrics.get("edges_after")
    if expected_nodes is not None and int(expected_nodes) != len(nodes):
        errors.append(f"Node count mismatch: expected {expected_nodes}, got {len(nodes)}")
    if expected_edges is not None and int(expected_edges) != len(edges):
        errors.append(f"Edge count mismatch: expected {expected_edges}, got {len(edges)}")

    non_allowed_node_types = sorted(
        {
            node_type
            for node_type in node_type_by_id.values()
            if node_type and node_type not in ALLOWED_NODE_TYPES
        }
    )
    if non_allowed_node_types:
        errors.append(f"Found non-allowed node types: {', '.join(non_allowed_node_types)}")

    non_allowed_predicates = sorted(
        {
            str(edge.get("predicate"))
            for edge in edges
            if edge.get("predicate") not in ALLOWED_PREDICATES
        }
    )
    if non_allowed_predicates:
        errors.append(f"Found non-allowed predicates: {', '.join(non_allowed_predicates)}")

    discourse_violations = 0

    for edge in edges:
        source_id = edge.get("source_id")
        target_id = edge.get("target_id")
        predicate = edge.get("predicate")

        if source_id not in node_ids:
            errors.append(f"Edge {edge.get('id')} has invalid source_id: {source_id}")
        if target_id not in node_ids:
            errors.append(f"Edge {edge.get('id')} has invalid target_id: {target_id}")

        if predicate in DISCOURSE_PREDICATES:
            source_type = node_type_by_id.get(str(source_id))
            target_type = node_type_by_id.get(str(target_id))
            if source_type != "foaf:Person" or target_type != "foaf:Person":
                discourse_violations += 1

    if discourse_violations:
        errors.append(
            f"Discourse edge type violations: {discourse_violations} edges are not Person->Person"
        )

    observed = {
        "nodes_count": len(nodes),
        "edges_count": len(edges),
        "non_allowed_type_count": len(non_allowed_node_types),
        "non_allowed_predicate_count": len(non_allowed_predicates),
        "discourse_violation_count": discourse_violations,
    }

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "metrics": observed,
    }
