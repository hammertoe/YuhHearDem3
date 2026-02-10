"""Script to extract knowledge graph from a single video."""

from __future__ import annotations

import argparse
import os
import sys
import uuid


_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.db.postgres_client import PostgresClient
from lib.embeddings.google_client import GoogleEmbeddingClient
from lib.knowledge_graph.kg_extractor import DEFAULT_GEMINI_MODEL, KGExtractor
from lib.knowledge_graph.window_builder import (
    DEFAULT_STRIDE,
    DEFAULT_WINDOW_SIZE,
    WindowBuilder,
    ConceptWindow,
    DiscourseWindow,
)


def main():
    parser = argparse.ArgumentParser(description="Extract knowledge graph from a video")
    parser.add_argument("--youtube-video-id", required=True, help="YouTube video ID")
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Concept window size",
    )
    parser.add_argument(
        "--stride", type=int, default=DEFAULT_STRIDE, help="Concept window stride"
    )
    parser.add_argument(
        "--context-size", type=int, default=3, help="Discourse window context size"
    )
    parser.add_argument(
        "--max-windows", type=int, help="Maximum windows to process (for testing)"
    )
    parser.add_argument("--run-id", help="KG run ID (auto-generated if not provided)")
    parser.add_argument(
        "--model", default=DEFAULT_GEMINI_MODEL, help="Gemini model to use"
    )
    parser.add_argument(
        "--predicates-file",
        default=None,
        help="Path to config JSON with concept_predicates/discourse_predicates",
    )
    parser.add_argument(
        "--top-k", type=int, default=25, help="Top K candidate nodes to retrieve"
    )
    parser.add_argument(
        "--skip-concept", action="store_true", help="Skip concept extraction"
    )
    parser.add_argument(
        "--skip-discourse", action="store_true", help="Skip discourse extraction"
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

    print("=" * 60)
    print(f"KG Extraction: {youtube_video_id}")
    print(f"Run ID: {run_id}")
    print("=" * 60)

    try:
        with PostgresClient() as pg_client:
            embedding_client = GoogleEmbeddingClient()
            extractor = KGExtractor(
                pg_client,
                embedding_client,
                model=args.model,
                predicates_file=args.predicates_file,
            )
            window_builder = WindowBuilder(pg_client, embedding_client)

            print(f"\nBuilding windows for video {youtube_video_id}...")
            all_windows = window_builder.build_all_windows(
                youtube_video_id,
                window_size=args.window_size,
                stride=args.stride,
                context_size=args.context_size,
                filter_short=not args.no_filter_short,
            )

            concept_windows = [
                w for w in all_windows["concept"] if isinstance(w, ConceptWindow)
            ]
            discourse_windows = [
                w for w in all_windows["discourse"] if isinstance(w, DiscourseWindow)
            ]

            print(f"✅ Found {len(concept_windows)} concept windows")
            print(f"✅ Found {len(discourse_windows)} discourse windows")

            if args.max_windows:
                concept_windows = concept_windows[: args.max_windows]
                discourse_windows = discourse_windows[: args.max_windows]
                print(
                    f"🔧 Limited to {len(concept_windows)} concept windows and {len(discourse_windows)} discourse windows"
                )

            all_results = []

            if not args.skip_concept:
                print(f"\n{'=' * 60}")
                print("Processing concept windows...")
                print(f"{'=' * 60}")

                for i, window in enumerate(concept_windows, 1):
                    print(
                        f"\n[{i}/{len(concept_windows)}] Concept window {window.window_index}"
                    )

                    result = extractor.extract_from_concept_window(
                        window, youtube_video_id, top_k=args.top_k
                    )
                    all_results.append(result)

                    if result.parse_success:
                        print(
                            f"  ✅ {len(result.nodes_new)} new nodes, {len(result.edges)} edges"
                        )
                    else:
                        print(f"  ❌ Failed: {result.error}")

                        if args.debug:
                            debug_file = f"debug_concept_{i}.txt"
                            with open(debug_file, "w") as f:
                                f.write(f"Window text:\n{window.text}\n\n")
                                f.write(f"Raw response:\n{result.raw_response}\n\n")
                                f.write(f"Error:\n{result.error}\n")
                            print(f"  🔧 Debug info saved to {debug_file}")

            if not args.skip_discourse:
                print(f"\n{'=' * 60}")
                print("Processing discourse windows...")
                print(f"{'=' * 60}")

                for i, window in enumerate(discourse_windows, 1):
                    print(
                        f"\n[{i}/{len(discourse_windows)}] Discourse window {window.window_index}"
                    )

                    result = extractor.extract_from_discourse_window(
                        window, youtube_video_id, top_k=args.top_k
                    )
                    all_results.append(result)

                    if result.parse_success:
                        print(
                            f"  ✅ {len(result.nodes_new)} new nodes, {len(result.edges)} edges"
                        )
                    else:
                        print(f"  ❌ Failed: {result.error}")

                        if args.debug:
                            debug_file = f"debug_discourse_{i}.txt"
                            with open(debug_file, "w") as f:
                                f.write(f"Window text:\n{window.text}\n\n")
                                f.write(f"Raw response:\n{result.raw_response}\n\n")
                                f.write(f"Error:\n{result.error}\n")
                            print(f"  🔧 Debug info saved to {debug_file}")

            print(f"\n{'=' * 60}")
            print("Canonicalizing and storing...")
            print(f"{'=' * 60}")

            stats = extractor.canonicalize_and_store(
                all_results, youtube_video_id, run_id, args.model
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
            print(f"Run ID: {run_id}")
            print("=" * 60)

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
