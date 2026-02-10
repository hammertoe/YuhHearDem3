from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone


_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from lib.db.memgraph_client import MemgraphClient
from lib.knowledge_graph.ingestor import KnowledgeGraphMemgraphIngestor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest knowledge_graph.json into Memgraph"
    )
    parser.add_argument(
        "--kg-file",
        default="knowledge_graph.json",
        help="Path to knowledge_graph.json",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Tag this ingestion run (default: timestamp-based)",
    )
    parser.add_argument(
        "--youtube-video-id",
        default=None,
        help="Optional youtube_video_id to attach to nodes/edges",
    )
    parser.add_argument(
        "--delete-run-id",
        default=None,
        help="If set, delete any nodes/edges with this kg_run_id before ingest",
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("kg_%Y%m%dT%H%M%SZ")

    kg_path = args.kg_file
    if not os.path.isabs(kg_path) and not os.path.exists(kg_path):
        candidate = os.path.join(_REPO_ROOT, kg_path)
        if os.path.exists(candidate):
            kg_path = candidate

    if not os.path.exists(kg_path):
        print(f"❌ Knowledge graph file not found: {args.kg_file}")
        print("   Create it first with:")
        print(
            "   python build_knowledge_graph.py --input-file transcription_output.json --output-json knowledge_graph.json"
        )
        return 1

    with open(kg_path, "r", encoding="utf-8") as f:
        kg = json.load(f)

    with MemgraphClient() as memgraph:
        ingestor = KnowledgeGraphMemgraphIngestor(memgraph)

        if args.delete_run_id:
            deleted = ingestor.delete_run(args.delete_run_id)
            print(f"🧹 Deleted nodes for run_id={args.delete_run_id}: {deleted}")

        stats = ingestor.ingest(
            kg,
            run_id=run_id,
            youtube_video_id=args.youtube_video_id,
        )

    print(f"✅ Ingested knowledge graph into Memgraph (run_id={run_id})")
    print(f"   nodes={stats.nodes} edges={stats.edges}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
