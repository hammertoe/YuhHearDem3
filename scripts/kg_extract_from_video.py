"""Script to extract knowledge graph from a single video."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.db.postgres_client import PostgresClient
from lib.embeddings.google_client import GoogleEmbeddingClient
from lib.knowledge_graph.oss_kg_extractor import OssKGExtractor, DEFAULT_MODEL
from lib.knowledge_graph.kg_store import canonicalize_and_store
from lib.knowledge_graph.base_kg_seeder import BaseKGSeeder
from lib.knowledge_graph.window_builder import (
    DEFAULT_STRIDE,
    DEFAULT_WINDOW_SIZE,
    WindowBuilder,
    ConceptWindow,
)


def main():
    print("=" * 60)
    print("CEREBRAS KG EXTRACTION CONFIGURATION")
    print("=" * 60)
    print(f"Model: {DEFAULT_MODEL}")
    print("Two-pass: always")
    print("Reasoning effort: medium")
    print("Reasoning format: hidden")
    print("Max completion tokens: 16384")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Extract knowledge graph from a video")
    parser.add_argument("--youtube-video-id", required=True, help="YouTube video ID")
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Window size",
    )
    parser.add_argument(
        "--stride", type=int, default=DEFAULT_STRIDE, help="Window stride"
    )
    parser.add_argument(
        "--max-windows", type=int, help="Maximum windows to process (for testing)"
    )
    parser.add_argument("--run-id", help="KG run ID (auto-generated if not provided)")
    parser.add_argument(
        "--top-k", type=int, default=25, help="Top K candidate nodes to retrieve"
    )
    parser.add_argument(
        "--no-filter-short",
        action="store_true",
        help="Do not filter short utterances in concept windows",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode: save failed responses to file"
    )
    args = parser.parse_args()

    run_id = args.run_id or str(uuid.uuid4())
    youtube_video_id = args.youtube_video_id

    print()
    print(f"KG Extraction: {youtube_video_id}")
    print(f"Run ID: {run_id}")
    print("=" * 60)

    try:
        with PostgresClient() as pg_client:
            embedding_client = GoogleEmbeddingClient()

            print("\nSeeding base knowledge graph...")
            seeder = BaseKGSeeder(pg_client, embedding_client)
            seed_counts = seeder.seed_all()
            print(
                "✅ Seeded base KG: "
                f"speakers={seed_counts.get('speakers', 0)}, "
                f"order_paper_items={seed_counts.get('order_paper_items', 0)}, "
                f"bills={seed_counts.get('bills', 0)}"
            )

            extractor = OssKGExtractor(
                pg_client,
                embedding_client,
            )
            window_builder = WindowBuilder(pg_client, embedding_client)

            print(f"\nBuilding windows for video {youtube_video_id}...")
            windows = window_builder.build_all_windows(
                youtube_video_id,
                window_size=args.window_size,
                stride=args.stride,
                filter_short=not args.no_filter_short,
            )

            concept_windows = [w for w in windows if isinstance(w, ConceptWindow)]

            print(f"✅ Found {len(concept_windows)} windows")

            if args.max_windows:
                concept_windows = concept_windows[: args.max_windows]
                print(f"🔧 Limited to {len(concept_windows)} windows")

            all_results = []

            print(f"\n{'=' * 60}")
            print("Processing windows...")
            print(f"{'=' * 60}")

            for i, window in enumerate(concept_windows, 1):
                print(f"\n[{i}/{len(concept_windows)}] Window {window.window_index}")

                result = extractor.extract_from_concept_window(
                    window, youtube_video_id, top_k=args.top_k
                )
                all_results.append(result)

                if result.parse_success:
                    print(
                        f"  ✅ {len(result.nodes_new)} new nodes, {len(result.edges)} edges"
                    )
                    if result.pass1_elapsed_s is not None:
                        print(
                            f"     Pass 1: {result.pass1_elapsed_s:.2f}s ({result.pass1_edge_count} edges, {result.pass1_violations_count} violations)"
                        )
                    if result.pass2_elapsed_s is not None:
                        print(
                            f"     Pass 2: {result.pass2_elapsed_s:.2f}s ({result.pass2_trigger})"
                        )
                else:
                    print(f"  ❌ Failed: {result.error}")

                    if args.debug:
                        debug_file = f"debug_window_{i}.txt"
                        with open(debug_file, "w") as f:
                            f.write(f"Window text:\n{window.text}\n\n")
                            if result.prompt_pass1:
                                f.write(f"Pass 1 prompt:\n{result.prompt_pass1}\n\n")
                            f.write(
                                f"Raw response pass 1:\n{result.raw_response_pass1}\n\n"
                            )
                            if result.reasoning_pass1:
                                f.write(
                                    f"Reasoning pass 1:\n{result.reasoning_pass1}\n\n"
                                )
                            if result.prompt_pass2:
                                f.write(f"Pass 2 prompt:\n{result.prompt_pass2}\n\n")
                            if result.raw_response_pass2:
                                f.write(
                                    f"Raw response pass 2:\n{result.raw_response_pass2}\n\n"
                                )
                            if result.reasoning_pass2:
                                f.write(
                                    f"Reasoning pass 2:\n{result.reasoning_pass2}\n\n"
                                )
                            f.write(f"Error:\n{result.error}\n")
                        print(f"  🔧 Debug info saved to {debug_file}")

            print(f"\n{'=' * 60}")
            print("Canonicalizing and storing...")
            print(f"{'=' * 60}")

            results_for_store = [
                (
                    r.window,
                    r.nodes_new,
                    r.edges,
                    r.raw_response,
                    r.parse_success,
                    r.error,
                )
                for r in all_results
            ]

            stats = canonicalize_and_store(
                postgres=pg_client,
                embedding=embedding_client,
                results=results_for_store,
                youtube_video_id=youtube_video_id,
                kg_run_id=run_id,
                extractor_model=DEFAULT_MODEL,
            )

            print(f"\n{'=' * 60}")
            print("Summary")
            print(f"{'=' * 60}")
            print(f"Windows processed: {stats['windows_processed']}")
            print(f"  Successful: {stats['windows_successful']}")
            print(f"  Failed: {stats['windows_failed']}")
            print(f"New nodes: {stats['new_nodes']}")
            print(f"Edges: {stats['edges']}")
            print(f"Links to known nodes: {stats['links_to_known']}")
            if stats["edges_skipped_invalid_speaker_ref"] > 0:
                print(
                    f"Edges skipped (invalid speaker ref): {stats['edges_skipped_invalid_speaker_ref']}"
                )
            if stats["edges_skipped_missing_nodes"] > 0:
                print(
                    f"Edges skipped (missing nodes): {stats['edges_skipped_missing_nodes']}"
                )
            print(f"Run ID: {run_id}")
            print("=" * 60)

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
