"""Script to sync Postgres KG to Memgraph."""

from __future__ import annotations

import argparse
import os
import re
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.db.memgraph_client import MemgraphClient
from lib.db.postgres_client import PostgresClient


def sanitize_relationship_type(predicate: str) -> str:
    """Sanitize predicate for Memgraph relationship type."""
    predicate = predicate.upper()
    predicate = re.sub(r"[^A-Z0-9_]", "_", predicate)
    return predicate


def sync_nodes(postgres_client: PostgresClient, memgraph_client: MemgraphClient) -> int:
    """Sync KG nodes from Postgres to Memgraph."""
    print("Syncing nodes...")

    query = """
        SELECT id, label, type, aliases
        FROM kg_nodes
    """
    rows = postgres_client.execute_query(query)

    count = 0
    for row in rows:
        node_id, label, node_type, aliases = row

        memgraph_client.merge_entity(
            "KGNode",
            "id",
            {
                "id": node_id,
                "label": label,
                "type": node_type,
                "aliases": aliases or [],
            },
        )
        count += 1

        if count % 100 == 0:
            print(f"  Synced {count} nodes...")

    print(f"✅ Synced {count} nodes")
    return count


def sync_edges(
    postgres_client: PostgresClient,
    memgraph_client: MemgraphClient,
    youtube_video_id: str | None = None,
    kg_run_id: str | None = None,
) -> int:
    """Sync KG edges from Postgres to Memgraph."""
    print("Syncing edges...")

    query = """
        SELECT
            id,
            source_id,
            predicate,
            predicate_raw,
            target_id,
            youtube_video_id,
            earliest_timestamp_str,
            earliest_seconds,
            utterance_ids,
            evidence,
            speaker_ids,
            confidence,
            extractor_model,
            kg_run_id
        FROM kg_edges
    """

    params: tuple = ()

    if youtube_video_id:
        query += " WHERE youtube_video_id = %s"
        params = (youtube_video_id,)
    elif kg_run_id:
        query += " WHERE kg_run_id = %s"
        params = (kg_run_id,)

    rows = postgres_client.execute_query(query, params)

    count = 0
    for row in rows:
        (
            edge_id,
            source_id,
            predicate,
            predicate_raw,
            target_id,
            video_id,
            earliest_timestamp_str,
            earliest_seconds,
            utterance_ids,
            evidence,
            speaker_ids,
            confidence,
            extractor_model,
            run_id,
        ) = row

        rel_type = sanitize_relationship_type(predicate)

        properties = {
            "edge_id": edge_id,
            "youtube_video_id": video_id,
            "earliest_timestamp_str": earliest_timestamp_str,
            "earliest_seconds": earliest_seconds,
            "utterance_ids": utterance_ids or [],
            "evidence": evidence,
            "speaker_ids": speaker_ids or [],
            "confidence": confidence,
            "extractor_model": extractor_model,
            "kg_run_id": run_id,
            "predicate_raw": predicate_raw,
        }

        memgraph_client.merge_relationship(
            source_id,
            target_id,
            rel_type,
            match_properties={"edge_id": edge_id},
            set_properties=properties,
        )
        count += 1

        if count % 100 == 0:
            print(f"  Synced {count} edges...")

    print(f"✅ Synced {count} edges")
    return count


def main():
    parser = argparse.ArgumentParser(description="Sync Postgres KG to Memgraph")
    parser.add_argument("--youtube-video-id", help="Sync only edges for this video")
    parser.add_argument("--run-id", help="Sync only edges for this KG run")
    parser.add_argument("--skip-nodes", action="store_true", help="Skip syncing nodes")
    parser.add_argument("--skip-edges", action="store_true", help="Skip syncing edges")
    args = parser.parse_args()

    if args.youtube_video_id and args.run_id:
        print("❌ Error: Cannot specify both --youtube-video-id and --run-id")
        return

    print("=" * 60)
    print("KG Sync: Postgres -> Memgraph")
    print("=" * 60)

    if args.youtube_video_id:
        print(f"Filtering by video ID: {args.youtube_video_id}")
    elif args.run_id:
        print(f"Filtering by run ID: {args.run_id}")

    try:
        with PostgresClient() as pg_client, MemgraphClient() as mg_client:
            stats = {
                "nodes": 0,
                "edges": 0,
            }

            if not args.skip_nodes:
                stats["nodes"] = sync_nodes(pg_client, mg_client)

            if not args.skip_edges:
                stats["edges"] = sync_edges(
                    pg_client,
                    mg_client,
                    youtube_video_id=args.youtube_video_id,
                    kg_run_id=args.run_id,
                )

            print("\n" + "=" * 60)
            print("Summary")
            print("=" * 60)
            print(f"Nodes synced: {stats['nodes']}")
            print(f"Edges synced: {stats['edges']}")
            print("=" * 60)

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
