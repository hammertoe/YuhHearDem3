"""KG cleanup pass - aggressive same-type deduplication + graph ranking."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.db.postgres_client import PostgresClient
from lib.knowledge_graph.cleanup.contracts import (
    ALLOWED_NODE_TYPES,
    ALLOWED_PREDICATES,
    PREDICATE_PRIOR_WEIGHTS,
    get_remapped_node_type,
    get_remapped_predicate,
    is_node_type_allowed,
    is_predicate_allowed,
)
from lib.knowledge_graph.cleanup.candidates import (
    build_type_blocks,
    generate_candidate_pairs,
)
from lib.knowledge_graph.cleanup.cluster import (
    cluster_from_candidate_pairs,
)
from lib.knowledge_graph.cleanup.normalize import strip_honorifics
from lib.knowledge_graph.cleanup.rewrite import (
    rewrite_and_clean_edges,
)
from lib.knowledge_graph.cleanup.rank import (
    compute_all_ranking_scores,
    compute_edge_weight,
    normalize_scores,
)
from lib.knowledge_graph.cleanup.export_load import (
    export_all_artifacts,
    export_metrics,
    load_edges_from_csv,
    load_merge_map_from_csv,
    load_nodes_from_csv,
    verify_loaded_data,
)


def generate_run_id() -> str:
    """Generate unique run ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def profile_mode(postgres: PostgresClient, output_dir: Path) -> None:
    """Profile current KG state and save metrics."""
    print("📊 Profiling current KG state...")

    nodes_query = """
    SELECT id, type, label, aliases
    FROM kg_nodes
    """
    nodes = postgres.execute_query(nodes_query)

    edges_query = """
    SELECT id, source_id, predicate, target_id, confidence, support_count
    FROM kg_edges
    """
    edges = postgres.execute_query(edges_query)

    type_counts: dict[str, int] = {}
    predicate_counts: dict[str, int] = {}

    for node in nodes:
        node_type = node[1]
        type_counts[node_type] = type_counts.get(node_type, 0) + 1

    for edge in edges:
        predicate = edge[2]
        predicate_counts[predicate] = predicate_counts.get(predicate, 0) + 1

    non_allowed_types = {t for t in type_counts if t not in ALLOWED_NODE_TYPES}
    non_allowed_predicates = {p for p in predicate_counts if p not in ALLOWED_PREDICATES}

    metrics = {
        "profile_timestamp": datetime.now().isoformat(),
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "type_counts": type_counts,
        "predicate_counts": predicate_counts,
        "non_allowed_types": list(non_allowed_types),
        "non_allowed_predicates": list(non_allowed_predicates),
        "non_allowed_type_count": len(non_allowed_types),
        "non_allowed_predicate_count": len(non_allowed_predicates),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    export_metrics(metrics, output_dir / "profile_before.json")

    print(f"✅ Profile saved to {output_dir / 'profile_before.json'}")
    print(f"   Total nodes: {len(nodes)}")
    print(f"   Total edges: {len(edges)}")
    print(f"   Non-allowed types: {len(non_allowed_types)}")
    print(f"   Non-allowed predicates: {len(non_allowed_predicates)}")


def dry_run_mode(
    postgres: PostgresClient,
    output_dir: Path,
    aggressive: bool,
    same_type_only: bool,
) -> None:
    """Run cleanup pipeline without DB mutation."""
    print("🔄 Running cleanup in dry-run mode...")

    nodes_query = """
    SELECT id, type, label, aliases, embedding
    FROM kg_nodes
    """
    nodes_rows = postgres.execute_query(nodes_query)
    nodes = {
        row[0]: {
            "id": row[0],
            "type": row[1],
            "label": row[2],
            "aliases": row[3] if row[3] else [],
            "embedding": row[4],
        }
        for row in nodes_rows
    }

    edges_query = """
    SELECT
        e.id, e.source_id, e.target_id, e.predicate,
        e.youtube_video_id, e.earliest_timestamp_str, e.earliest_seconds,
        e.utterance_ids, e.evidence, e.confidence,
        n1.type as source_type, n2.type as target_type
    FROM kg_edges e
    JOIN kg_nodes n1 ON e.source_id = n1.id
    JOIN kg_nodes n2 ON e.target_id = n2.id
    """
    edges_rows = postgres.execute_query(edges_query)
    edges = [
        {
            "id": row[0],
            "source_id": row[1],
            "target_id": row[2],
            "predicate": row[3],
            "youtube_video_id": row[4],
            "earliest_timestamp_str": row[5],
            "earliest_seconds": row[6],
            "utterance_ids": row[7] if row[7] else [],
            "evidence": row[8],
            "confidence": float(row[9]) if row[9] else 0.5,
            "source_type": row[10],
            "target_type": row[11],
        }
        for row in edges_rows
    ]

    print(f"   Loaded {len(nodes)} nodes and {len(edges)} edges")

    if aggressive:
        print("   Applying type/predicate remapping...")
        for node_id, node in nodes.items():
            if not is_node_type_allowed(node["type"]):
                old_type = node["type"]
                new_type = get_remapped_node_type(node["type"])
                if new_type:
                    node["type"] = new_type
                    print(f"     Remapped {node_id}: {old_type} -> {new_type}")

        for edge in edges:
            if not is_predicate_allowed(edge["predicate"]):
                old_predicate = edge["predicate"]
                new_predicate = get_remapped_predicate(edge["predicate"])
                if new_predicate:
                    edge["predicate"] = new_predicate
                    print(f"     Remapped predicate: {old_predicate} -> {new_predicate}")

    degree_by_node: dict[str, int] = {node_id: 0 for node_id in nodes}
    for edge in edges:
        source_id = edge.get("source_id")
        target_id = edge.get("target_id")
        if source_id in degree_by_node:
            degree_by_node[source_id] += 1
        if target_id in degree_by_node:
            degree_by_node[target_id] += 1

    global_merge_map: dict[str, str] = {node_id: node_id for node_id in nodes}

    if same_type_only:
        print("   Running same-type aggressive deduplication...")

        for node_type in ALLOWED_NODE_TYPES:
            blocks = build_type_blocks(nodes, node_type)
            candidate_pairs = generate_candidate_pairs(nodes, blocks, node_type)

            node_data = {
                nid: {
                    **node,
                    "degree": degree_by_node.get(nid, 0),
                    "alias_count": len(node.get("aliases", [])),
                    "is_speaker": node["id"].startswith("speaker_"),
                }
                for nid, node in nodes.items()
                if node["type"] == node_type
            }

            merge_map = cluster_from_candidate_pairs(candidate_pairs, node_data, threshold=0.5)
            merge_count = sum(
                1 for node_id, canonical_id in merge_map.items() if node_id != canonical_id
            )
            print(f"   Generated {merge_count} merges for {node_type}")
            global_merge_map.update(merge_map)

    edges = rewrite_edge_endpoints_via_map(edges, global_merge_map)
    canonical_nodes = build_canonical_nodes(nodes, global_merge_map)

    for edge in edges:
        edge["source_type"] = canonical_nodes.get(edge["source_id"], {}).get("type")
        edge["target_type"] = canonical_nodes.get(edge["target_id"], {}).get("type")

    print("   Computing ranking scores...")
    node_dict = {n["id"]: n for n in canonical_nodes.values()}
    edge_list = list(edges)
    pagerank_scores, edge_rank_scores = compute_all_ranking_scores(edge_list, node_dict)

    print("   Normalizing scores to [0, 1] range...")
    pagerank_scores = normalize_scores(pagerank_scores)
    edge_rank_scores = normalize_scores(edge_rank_scores)

    for node_id, pr_score in pagerank_scores.items():
        if node_id in node_dict:
            node_dict[node_id]["pagerank_score"] = pr_score

    for edge in edge_list:
        edge_id = edge["id"]
        support_count_str = edge.get("support_count")
        if support_count_str:
            support_count = int(support_count_str) if support_count_str.isdigit() else 1
        else:
            support_count = 1
        confidence_val = edge.get("confidence")
        if confidence_val is not None:
            if isinstance(confidence_val, float):
                confidence = confidence_val
            elif isinstance(confidence_val, str):
                confidence = (
                    float(confidence_val)
                    if confidence_val.replace(".", "", "").isdigit()
                    or confidence_val.replace(".", "", "").lstrip("-").isdigit()
                    else 0.5
                )
            else:
                confidence = 0.5
        else:
            confidence = 0.5
        predicate_weight = PREDICATE_PRIOR_WEIGHTS.get(str(edge.get("predicate")), 0.1)
        edge["edge_weight"] = compute_edge_weight(support_count, confidence, predicate_weight)
        if edge_id in edge_rank_scores:
            edge["edge_rank_score"] = edge_rank_scores[edge_id]

    print("   Cleaning and collapsing edges...")
    valid_nodes = set(canonical_nodes.keys())
    result = rewrite_and_clean_edges(edges, {}, valid_nodes)

    print(f"   Dropped {len(result['dropped_edges'])} edges")
    print(f"   Kept {len(result['clean_edges'])} edges")

    print("   Filtering to allowed types and predicates...")
    canonical_nodes_allowed = {
        node_id: node
        for node_id, node in canonical_nodes.items()
        if node["type"] in ALLOWED_NODE_TYPES
    }
    allowed_node_ids = set(canonical_nodes_allowed.keys())
    clean_edges_allowed = [
        edge
        for edge in result["clean_edges"]
        if (
            edge.get("predicate") in ALLOWED_PREDICATES
            and edge.get("source_id") in allowed_node_ids
            and edge.get("target_id") in allowed_node_ids
        )
    ]
    filtered_merge_map = {
        old_id: new_id
        for old_id, new_id in global_merge_map.items()
        if new_id in canonical_nodes_allowed
    }

    nodes_before = len(canonical_nodes)
    edges_before = len(result["clean_edges"])
    nodes_after = len(canonical_nodes_allowed)
    edges_after = len(clean_edges_allowed)
    print(
        f"   Filtered nodes: {nodes_before} -> {nodes_after} (removed {nodes_before - nodes_after})"
    )
    print(
        f"   Filtered edges: {edges_before} -> {edges_after} (removed {edges_before - edges_after})"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    export_all_artifacts(
        list(canonical_nodes_allowed.values()),
        clean_edges_allowed,
        [],
        filtered_merge_map,
        result["dropped_edges"],
        {},
        output_dir,
    )

    merges_total = sum(
        1 for node_id, canonical_id in global_merge_map.items() if node_id != canonical_id
    )
    duplicate_reduction_pct = (merges_total / len(nodes) * 100.0) if nodes else 0.0
    non_allowed_node_types_after = sorted(
        {
            node["type"]
            for node in canonical_nodes_allowed.values()
            if node["type"] not in ALLOWED_NODE_TYPES
        }
    )
    non_allowed_predicates_after = sorted(
        {
            edge.get("predicate")
            for edge in clean_edges_allowed
            if edge.get("predicate") not in ALLOWED_PREDICATES
        }
    )
    discourse_violation_count = sum(
        1
        for edge in clean_edges_allowed
        if edge.get("predicate") in {"RESPONDS_TO", "AGREES_WITH", "DISAGREES_WITH", "QUESTIONS"}
        and (edge.get("source_type") != "foaf:Person" or edge.get("target_type") != "foaf:Person")
    )

    metrics_after = {
        "dry_run_timestamp": datetime.now().isoformat(),
        "nodes_before": len(nodes),
        "edges_before": len(edges_rows),
        "nodes_after": nodes_after,
        "edges_after": edges_after,
        "edges_dropped": len(result["dropped_edges"]) + (edges_before - edges_after),
        "merge_count": merges_total,
        "duplicate_reduction_pct": duplicate_reduction_pct,
        "non_allowed_types_after": non_allowed_node_types_after,
        "non_allowed_predicates_after": non_allowed_predicates_after,
        "discourse_violation_count": discourse_violation_count,
    }
    export_metrics(metrics_after, output_dir / "metrics_after.json")

    print(f"✅ Dry-run artifacts saved to {output_dir}")


def rewrite_edge_endpoints_via_map(edges: list[dict], merge_map: dict[str, str]) -> list[dict]:
    """Rewrite edge endpoints using merge map."""
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


def build_canonical_nodes(nodes: dict[str, dict], merge_map: dict[str, str]) -> dict[str, dict]:
    """Collapse nodes into canonical IDs and merge aliases/metadata."""
    canonical_nodes: dict[str, dict] = {}
    merged_sources: dict[str, set[str]] = {}

    for node_id, node in nodes.items():
        canonical_id = merge_map.get(node_id, node_id)

        if canonical_id not in canonical_nodes:
            canonical_node = node.copy()
            canonical_node["id"] = canonical_id
            canonical_node["aliases"] = list(node.get("aliases", []))

            if canonical_node.get("type") == "foaf:Person":
                canonical_node["label"] = strip_honorifics(canonical_node["label"])

            canonical_nodes[canonical_id] = canonical_node
            merged_sources[canonical_id] = set()

        merged_sources[canonical_id].add(node_id)
        alias_set = set(canonical_nodes[canonical_id].get("aliases", []))
        alias_set.update(node.get("aliases", []))
        canonical_nodes[canonical_id]["aliases"] = sorted(alias_set)

    for canonical_id, source_ids in merged_sources.items():
        canonical_nodes[canonical_id]["merge_cluster_id"] = canonical_id
        canonical_nodes[canonical_id]["merged_from_count"] = len(source_ids)

    return canonical_nodes


def export_mode(
    postgres: PostgresClient,
    output_dir: Path,
    aggressive: bool,
    same_type_only: bool,
) -> None:
    """Export cleaned KG artifacts."""
    print("📦 Running cleanup and exporting artifacts...")

    dry_run_mode(postgres, output_dir, aggressive, same_type_only)

    print(f"✅ Export complete. Artifacts in {output_dir}")


def apply_mode(postgres: PostgresClient, output_dir: Path) -> None:
    """Apply cleaned artifacts to database with transaction safety."""
    print("🔧 Applying cleanup changes to database...")

    nodes_path = output_dir / "kg_nodes_clean.csv"
    edges_path = output_dir / "kg_edges_clean.csv"
    merge_map_path = output_dir / "node_merge_map.csv"
    edge_drops_path = output_dir / "edge_drop_log.csv"
    metrics_path = output_dir / "metrics_after.json"

    for path in [nodes_path, edges_path, merge_map_path, edge_drops_path, metrics_path]:
        if not path.exists():
            print(f"❌ Missing artifact: {path}")
            return

    print("   Loading artifacts...")
    nodes = load_nodes_from_csv(nodes_path)
    edges = load_edges_from_csv(edges_path)
    merge_map = load_merge_map_from_csv(merge_map_path)

    print("   Filtering to allowed types and predicates...")
    nodes_allowed = [node for node in nodes if node.get("type") in ALLOWED_NODE_TYPES]
    allowed_node_ids = {n["id"] for n in nodes_allowed}
    edges_allowed = [
        edge
        for edge in edges
        if (
            edge.get("predicate") in ALLOWED_PREDICATES
            and edge.get("source_id") in allowed_node_ids
            and edge.get("target_id") in allowed_node_ids
        )
    ]
    merge_map_filtered = {
        old_id: new_id
        for old_id, new_id in merge_map.items()
        if any(n["id"] == new_id for n in nodes_allowed)
    }
    print(f"   Filtered nodes: {len(nodes)} -> {len(nodes_allowed)}")
    print(f"   Filtered edges: {len(edges)} -> {len(edges_allowed)}")

    nodes = nodes_allowed
    edges = edges_allowed
    merge_map = merge_map_filtered

    with open(edge_drops_path) as f:
        import csv

        edge_drops = [(row["edge_id"], row["reason"]) for row in csv.DictReader(f)]

    with open(metrics_path) as f:
        import json

        metrics = json.load(f)

    print(f"   Loaded {len(nodes)} nodes, {len(edges)} edges")
    print(f"   Merge map: {len(merge_map)} entries")
    print(f"   Edge drops: {len(edge_drops)}")

    print("   Creating backup tables...")
    postgres.execute_update("DROP TABLE IF EXISTS kg_nodes_backup CASCADE")
    postgres.execute_update("CREATE TABLE kg_nodes_backup AS SELECT * FROM kg_nodes")
    postgres.execute_update("DROP TABLE IF EXISTS kg_edges_backup CASCADE")
    postgres.execute_update("CREATE TABLE kg_edges_backup AS SELECT * FROM kg_edges")
    print("   ✅ Backup tables created")

    print("   Applying node merge map to edges...")
    merge_map_list = list(merge_map.items())
    if merge_map_list:
        for old_id, new_id in merge_map_list:
            if old_id != new_id:
                postgres.execute_update(
                    "UPDATE kg_edges SET source_id = %s WHERE source_id = %s",
                    (new_id, old_id),
                )
                postgres.execute_update(
                    "UPDATE kg_edges SET target_id = %s WHERE target_id = %s",
                    (new_id, old_id),
                )
        print("   ✅ Applied merge map to edges")

    print("   Deleting dropped edges...")
    dropped_edge_ids = [edge_id for edge_id, _ in edge_drops if edge_id]
    if dropped_edge_ids:
        for edge_id in dropped_edge_ids:
            postgres.execute_update("DELETE FROM kg_edges WHERE id = %s", (edge_id,))
        print(f"   ✅ Deleted {len(dropped_edge_ids)} dropped edges")

    print("   Updating node ranking metadata...")
    for node in nodes:
        pagerank_score = node.get("pagerank_score")
        merge_cluster_id = node.get("merge_cluster_id")
        merged_from_count = node.get("merged_from_count")
        postgres.execute_update(
            """
            UPDATE kg_nodes
            SET pagerank_score = %s,
                merge_cluster_id = %s,
                merged_from_count = %s
            WHERE id = %s
            """,
            (pagerank_score, merge_cluster_id, merged_from_count, node["id"]),
        )
    print(f"   ✅ Updated {len(nodes)} nodes with ranking data")

    print("   Updating edge ranking metadata...")
    for edge in edges:
        edge_weight = float(edge.get("edge_weight") or 0.0)
        edge_rank_score = float(edge.get("edge_rank_score") or 0.0)
        support_count_str = edge.get("support_count") or "1"
        support_count = int(support_count_str) if support_count_str else 1
        postgres.execute_update(
            """
            UPDATE kg_edges
            SET edge_weight = %s,
                edge_rank_score = %s,
                support_count = %s
            WHERE id = %s
            """,
            (edge_weight, edge_rank_score, support_count, edge["id"]),
        )
    print(f"   ✅ Updated {len(edges)} edges with ranking data")

    print("   Deleting merged non-canonical nodes...")
    canonical_ids = {node["id"] for node in nodes}
    for old_id, new_id in merge_map_list:
        if old_id != new_id and old_id not in canonical_ids:
            postgres.execute_update("DELETE FROM kg_nodes WHERE id = %s", (old_id,))
    print("   ✅ Deleted merged nodes")

    print("✅ Cleanup applied successfully")
    print(f"   Nodes before: {metrics.get('nodes_before')}")
    print(f"   Nodes after: {metrics.get('nodes_after')}")
    print(f"   Edges before: {metrics.get('edges_before')}")
    print(f"   Edges after: {metrics.get('edges_after')}")
    print(f"   Merged: {metrics.get('merge_count')} nodes")


def verify_mode(output_dir: Path, min_duplicate_reduction_pct: float = 3.0) -> None:
    """Verify exported data consistency."""
    print("✅ Verifying exported data...")

    nodes_path = output_dir / "kg_nodes_clean.csv"
    edges_path = output_dir / "kg_edges_clean.csv"
    metrics_path = output_dir / "metrics_after.json"

    if not nodes_path.exists() or not edges_path.exists() or not metrics_path.exists():
        print("❌ Verification failed: Missing CSV files")
        return

    import json

    with open(metrics_path) as f:
        metrics = json.load(f)

    nodes = load_nodes_from_csv(nodes_path)
    edges = load_edges_from_csv(edges_path)

    result = verify_loaded_data(nodes, edges, metrics)

    if result["valid"]:
        duplicate_reduction_pct = float(metrics.get("duplicate_reduction_pct", 0.0))
        if duplicate_reduction_pct < min_duplicate_reduction_pct:
            print(
                "❌ Verification failed: duplicate reduction below target "
                f"({duplicate_reduction_pct:.2f}% < {min_duplicate_reduction_pct:.2f}%)"
            )
            return
        print(
            "✅ Verification passed: "
            f"{result['metrics']['nodes_count']} nodes, {result['metrics']['edges_count']} edges"
        )
    else:
        print(f"❌ Verification failed: {len(result['errors'])} errors")
        for error in result["errors"]:
            print(f"   {error}")


def main() -> None:
    """Main entry point for KG cleanup pass."""
    parser = argparse.ArgumentParser(
        description="KG cleanup pass: aggressive same-type deduplication + graph ranking"
    )
    parser.add_argument(
        "--mode",
        choices=["profile", "dry-run", "export", "verify", "apply"],
        required=True,
        help="Execution mode",
    )
    parser.add_argument("--out-dir", type=str, help="Output directory for artifacts")
    parser.add_argument("--aggressive", action="store_true", help="Enable aggressive cleanup mode")
    parser.add_argument(
        "--same-type-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only merge same-type nodes (default)",
    )
    parser.add_argument(
        "--min-duplicate-reduction-pct",
        type=float,
        default=3.0,
        help="Minimum duplicate reduction percentage required for verification (default: 3.0)",
    )

    args = parser.parse_args()

    output_dir = (
        Path(args.out_dir) if args.out_dir else Path("artifacts/kg_cleanup") / generate_run_id()
    )

    postgres = PostgresClient()

    match args.mode:
        case "profile":
            profile_mode(postgres, output_dir)
        case "dry-run":
            dry_run_mode(postgres, output_dir, args.aggressive, args.same_type_only)
        case "export":
            export_mode(postgres, output_dir, args.aggressive, args.same_type_only)
        case "verify":
            verify_mode(output_dir, args.min_duplicate_reduction_pct)
        case "apply":
            apply_mode(postgres, output_dir)

    print("✅ Done")


if __name__ == "__main__":
    main()
