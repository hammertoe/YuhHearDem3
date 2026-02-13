"""Migrate existing transcription data to three-tier storage."""

import json
import argparse
from datetime import datetime

from lib.db.postgres_client import PostgresClient
from lib.embeddings.google_client import GoogleEmbeddingClient
from lib.processors.paragraph_splitter import (
    group_transcripts_into_paragraphs,
    split_paragraph_into_sentences,
)
from lib.id_generators import (
    generate_speaker_id,
    generate_bill_id,
)


def load_transcript_data(filepath: str) -> dict[str, object]:
    """Load transcript JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def load_knowledge_graph_data(filepath: str) -> dict[str, object]:
    """Load knowledge graph JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def migrate_speakers(
    postgres_client: PostgresClient, speakers_data: list[dict[str, object]]
) -> dict[str, str]:
    """Migrate speakers to PostgreSQL."""
    speaker_id_map = {}

    for speaker in speakers_data:
        speaker_id = speaker.get("speaker_id", "")
        if not speaker_id or speaker_id.startswith("speaker_"):
            speaker_id = generate_speaker_id(
                speaker.get("name", "Unknown"), set(speaker_id_map.values())
            )

        postgres_client.execute_update(
            """
            INSERT INTO speakers (
                id, normalized_name, full_name, title, position,
                party, first_appearance_date, last_appearance_date
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """,
            (
                speaker_id,
                speaker.get("name", "").lower().replace(" ", "_"),
                speaker.get("name", ""),
                "",
                speaker.get("position", ""),
                "",
                datetime.now().date(),
                datetime.now().date(),
            ),
        )

        speaker_id_map[speaker.get("speaker_id", "")] = speaker_id

    print(f"✅ Migrated {len(speakers_data)} speakers")
    return speaker_id_map


def migrate_bills(
    postgres_client: PostgresClient, legislation_data: list[dict[str, object]]
) -> dict[str, str]:
    """Migrate bills to PostgreSQL."""
    bill_id_map = {}

    for bill in legislation_data:
        bill_id = bill.get("id", generate_bill_id(bill.get("name", ""), set(bill_id_map.values())))

        postgres_client.execute_update(
            """
            INSERT INTO bills (
                id, bill_number, title, description, status, source_text
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                description = EXCLUDED.description,
                status = EXCLUDED.status
        """,
            (
                bill_id,
                bill.get("name", ""),
                bill.get("name", ""),
                bill.get("description", ""),
                "",
                "",
            ),
        )

        bill_id_map[bill.get("id", "")] = bill_id

    print(f"✅ Migrated {len(legislation_data)} bills")
    return bill_id_map


def migrate_paragraphs_and_sentences(
    postgres_client: PostgresClient,
    embedding_client: GoogleEmbeddingClient,
    transcripts: list[dict[str, object]],
    video_id: str,
    video_title: str,
    video_date: str,
) -> None:
    """Migrate transcripts to paragraphs and sentences with embeddings."""
    paragraphs = group_transcripts_into_paragraphs(video_id, transcripts)

    print(f"Grouped {len(transcripts)} sentences into {len(paragraphs)} paragraphs")

    paragraph_texts = [p.get_text() for p in paragraphs]
    paragraph_embeddings = embedding_client.generate_embeddings_batch(paragraph_texts)

    for i, paragraph in enumerate(paragraphs):
        embedding = paragraph_embeddings[i]

        postgres_client.execute_update(
            """
            INSERT INTO paragraphs (
                id, youtube_video_id, start_seconds, end_seconds,
                text, speaker_id, voice_id, start_timestamp,
                end_timestamp, embedding, video_date, video_title,
                sentence_count
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """,
            (
                paragraph.id,
                paragraph.youtube_video_id,
                paragraph.start_seconds,
                paragraph.end_seconds,
                paragraph.get_text(),
                paragraph.speaker_id,
                paragraph.voice_id,
                paragraph.start_timestamp,
                paragraph.end_timestamp,
                embedding,
                video_date,
                video_title,
                len(paragraph.sentences),
            ),
        )

    for paragraph in paragraphs:
        sentences = split_paragraph_into_sentences(paragraph, video_id, video_date, video_title)

        for sentence in sentences:
            postgres_client.execute_update(
                """
                INSERT INTO sentences (
                    id, youtube_video_id, seconds_since_start, timestamp_str,
                    text, speaker_id, voice_id, paragraph_id,
                    sentence_order, video_date, video_title
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """,
                (
                    sentence["id"],
                    sentence["youtube_video_id"],
                    sentence["seconds_since_start"],
                    sentence["timestamp_str"],
                    sentence["text"],
                    sentence["speaker_id"],
                    sentence["voice_id"],
                    sentence["paragraph_id"],
                    sentence["sentence_order"],
                    sentence["video_date"],
                    sentence["video_title"],
                ),
            )

    print(f"✅ Migrated {len(paragraphs)} paragraphs with embeddings")
    print(f"✅ Migrated {len(transcripts)} sentences (no embeddings)")


def migrate_video_metadata(
    postgres_client: PostgresClient, transcript_data: dict[str, object]
) -> str:
    """Migrate video metadata."""
    video_metadata = transcript_data.get("video_metadata", {})
    video_id = "Syxyah7QIaM"

    postgres_client.execute_update(
        """
        INSERT INTO videos (
            youtube_id, title, description, upload_date, duration_seconds,
            processed, num_sentences, num_speakers
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (youtube_id) DO UPDATE SET
            processed = EXCLUDED.processed,
            processed_at = NOW()
    """,
        (
            video_id,
            video_metadata.get("title", ""),
            "",
            video_metadata.get("upload_date", ""),
            0,
            True,
            len(transcript_data.get("transcripts", [])),
            len(transcript_data.get("speakers", [])),
        ),
    )

    return video_id


def main():
    parser = argparse.ArgumentParser(description="Migrate existing data to three-tier storage")
    parser.add_argument(
        "--transcript-file",
        default="transcription_output.json",
        help="Transcript JSON file",
    )
    parser.add_argument(
        "--kg-file", default="knowledge_graph.json", help="Knowledge graph JSON file"
    )
    parser.add_argument("--video-id", default="Syxyah7QIaM", help="YouTube video ID")

    args = parser.parse_args()

    print("=" * 80)
    print("Starting Migration to Three-Tier Storage")
    print("=" * 80)

    try:
        transcript_data = load_transcript_data(args.transcript_file)

        with (
            PostgresClient() as postgres,
            GoogleEmbeddingClient() as embeddings,
        ):
            video_id = migrate_video_metadata(postgres, transcript_data)

            migrate_speakers(postgres, transcript_data.get("speakers", []))

            migrate_bills(postgres, transcript_data.get("legislation", []))

            migrate_paragraphs_and_sentences(
                postgres,
                embeddings,
                transcript_data.get("transcripts", []),
                video_id,
                transcript_data["video_metadata"]["title"],
                transcript_data["video_metadata"]["upload_date"],
            )

        print("\n" + "=" * 80)
        print("✅ Migration Complete!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        raise


if __name__ == "__main__":
    main()
