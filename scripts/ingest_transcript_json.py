from __future__ import annotations

import argparse
import json
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from lib.db.postgres_client import PostgresClient
from lib.embeddings.google_client import GoogleEmbeddingClient
from lib.transcripts.ingestor import TranscriptIngestor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest a transcription_output.json into three-tier Postgres tables"
    )
    parser.add_argument(
        "--transcript-file",
        default="transcription_output.json",
        help="Path to transcript JSON file",
    )
    parser.add_argument(
        "--youtube-video-id",
        required=True,
        help="YouTube video ID for this transcript",
    )
    parser.add_argument(
        "--skip-entity-embeddings",
        action="store_true",
        help="Skip generating/storing entity embeddings (paragraph embeddings still generated)",
    )
    parser.add_argument(
        "--skip-paragraph-embeddings",
        action="store_true",
        help="Skip generating/storing paragraph embeddings",
    )
    args = parser.parse_args()

    with open(args.transcript_file, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    embeddings = GoogleEmbeddingClient()
    with PostgresClient() as postgres:
        ingestor = TranscriptIngestor(
            postgres=postgres,
            embedding_client=embeddings,
        )
        stats = ingestor.ingest_transcript_json(
            transcript_data,
            youtube_video_id=args.youtube_video_id,
            embed_paragraphs=not args.skip_paragraph_embeddings,
            embed_entities=not args.skip_entity_embeddings,
        )

    print("✅ Transcript ingested")
    print(
        f"   paragraphs={stats.paragraphs} sentences={stats.sentences} speakers={stats.speakers} "
        f"entities={stats.entities} sentence_entities={stats.sentence_entities} "
        f"paragraph_entities={stats.paragraph_entities}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
