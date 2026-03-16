"""Extract KG nodes/edges from bill excerpts using sliding windows."""

from __future__ import annotations

import argparse
import os
import uuid

from lib.db.postgres_client import PostgresClient
from lib.embeddings.google_client import GoogleEmbeddingClient
from lib.knowledge_graph.bill_window_builder import BillWindowBuilder
from lib.knowledge_graph.kg_extractor import DEFAULT_GEMINI_MODEL, KGExtractor
from lib.knowledge_graph.kg_store import canonicalize_and_store
from lib.knowledge_graph.oss_kg_extractor import DEFAULT_MODEL, OssKGExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract KG from bill excerpts with overlapping windows"
    )
    parser.add_argument("--bill-id", default=None, help="Single bill ID to process")
    parser.add_argument("--max-bills", type=int, default=None, help="Limit bills to process")
    parser.add_argument(
        "--max-windows-per-bill",
        type=int,
        default=None,
        help="Limit windows processed per bill",
    )
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument(
        "--provider",
        choices=["auto", "oss", "gemini"],
        default="auto",
        help="LLM provider for extraction (default: auto)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("Bill KG Extraction")
    print("=" * 72)

    with PostgresClient() as pg_client:
        embedding_client = GoogleEmbeddingClient()
        provider = args.provider
        if provider == "auto":
            provider = "oss" if os.getenv("CEREBRAS_API_KEY") else "gemini"

        if provider == "oss":
            extractor = OssKGExtractor(
                postgres_client=pg_client,
                embedding_client=embedding_client,
                model=DEFAULT_MODEL,
            )
            extractor_model = DEFAULT_MODEL
        else:
            extractor = KGExtractor(
                postgres_client=pg_client,
                embedding_client=embedding_client,
                model=DEFAULT_GEMINI_MODEL,
            )
            extractor_model = DEFAULT_GEMINI_MODEL

        print(f"Provider: {provider}")
        bill_builder = BillWindowBuilder(postgres_client=pg_client)

        if args.bill_id:
            bill_ids = [args.bill_id]
        else:
            bill_ids = bill_builder.fetch_bill_ids()
            if args.max_bills:
                bill_ids = bill_ids[: args.max_bills]

        print(f"Bills to process: {len(bill_ids)}")

        total_windows = 0
        total_edges = 0
        total_nodes = 0

        for idx, bill_id in enumerate(bill_ids, 1):
            print(f"\n[{idx}/{len(bill_ids)}] bill_id={bill_id}")
            windows = bill_builder.build_bill_windows(
                bill_id=bill_id,
                window_size=args.window_size,
                stride=args.stride,
            )
            if args.max_windows_per_bill:
                windows = windows[: args.max_windows_per_bill]

            print(f"  windows: {len(windows)}")
            if not windows:
                continue

            run_id = f"kg_bill_{bill_id}_{uuid.uuid4().hex[:8]}"
            results = []
            for w_idx, window in enumerate(windows, 1):
                print(f"    extracting window {w_idx}/{len(windows)}")
                result = extractor.extract_from_concept_window(
                    window=window,
                    youtube_video_id=bill_id,
                    top_k=args.top_k,
                )

                nodes_new = [
                    n if isinstance(n, dict) else dict(n.__dict__)
                    for n in list(result.nodes_new or [])
                ]
                edges = [
                    e if isinstance(e, dict) else dict(e.__dict__) for e in list(result.edges or [])
                ]

                results.append(
                    (
                        result.window,
                        nodes_new,
                        edges,
                        result.raw_response,
                        result.parse_success,
                        result.error,
                    )
                )

            stats = canonicalize_and_store(
                postgres=pg_client,
                embedding=embedding_client,
                results=results,
                youtube_video_id=None,
                kg_run_id=run_id,
                extractor_model=extractor_model,
                source_kind="bill",
                source_ref_id=bill_id,
            )

            total_windows += stats["windows_processed"]
            total_edges += stats["edges"]
            total_nodes += stats["new_nodes"]

            print(
                f"  stored windows={stats['windows_processed']} nodes={stats['new_nodes']} edges={stats['edges']}"
            )

        print("\n" + "=" * 72)
        print("Bill KG Extraction Complete")
        print(f"windows={total_windows} nodes={total_nodes} edges={total_edges}")
        print("=" * 72)


if __name__ == "__main__":
    main()
