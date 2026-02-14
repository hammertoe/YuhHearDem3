from __future__ import annotations

from unittest.mock import Mock

from lib.transcripts.ingestor import TranscriptIngestor


def test_transcript_ingestor_writes_core_tables() -> None:
    postgres = Mock()
    embeddings = Mock()
    embeddings.generate_embeddings_batch.return_value = [[0.0] * 768]
    embeddings.generate_embedding.return_value = [0.0] * 768

    ingestor = TranscriptIngestor(
        postgres=postgres,
        embedding_client=embeddings,
    )

    transcript_data = {
        "video_metadata": {
            "title": "Test Video",
            "upload_date": "20260106",
            "duration": "0:01:00",
        },
        "speakers": [
            {
                "speaker_id": "s_speaker_1",
                "voice_id": 1,
                "name": "Speaker One",
                "position": "Unknown",
                "role_in_video": "member",
            }
        ],
        "transcripts": [
            {
                "start": "00:00:10",
                "text": "The Road Traffic Act is important.",
                "voice_id": 1,
                "speaker_id": "s_speaker_1",
            }
        ],
        "legislation": [],
    }

    ingestor.ingest_transcript_json(transcript_data, youtube_video_id="test_video")

    # At minimum we insert: videos, speakers, paragraphs, sentences.
    assert postgres.execute_update.call_count > 0


def test_transcript_ingestor_upserts_speaker_video_roles() -> None:
    postgres = Mock()
    embeddings = Mock()
    embeddings.generate_embeddings_batch.return_value = [[0.0] * 768]

    ingestor = TranscriptIngestor(
        postgres=postgres,
        embedding_client=embeddings,
    )

    transcript_data = {
        "video_metadata": {
            "title": "Test Video",
            "upload_date": "20260106",
            "duration": "0:01:00",
        },
        "speakers": [
            {
                "speaker_id": "s_speaker_1",
                "voice_id": 1,
                "name": "Speaker One",
                "position": "Minister for Education",
                "role_in_video": "member",
            }
        ],
        "transcripts": [
            {
                "start": "00:00:10",
                "text": "Hello.",
                "voice_id": 1,
                "speaker_id": "s_speaker_1",
            }
        ],
        "legislation": [],
    }

    ingestor.ingest_transcript_json(transcript_data, youtube_video_id="test_video")

    role_queries = [
        call.args[0]
        for call in postgres.execute_update.call_args_list
        if "INSERT INTO speaker_video_roles" in call.args[0]
    ]
    assert len(role_queries) >= 1


def test_transcript_ingestor_should_upsert_legislation_into_bills() -> None:
    postgres = Mock()
    embeddings = Mock()
    embeddings.generate_embeddings_batch.return_value = [[0.0] * 768]

    ingestor = TranscriptIngestor(
        postgres=postgres,
        embedding_client=embeddings,
    )

    transcript_data = {
        "video_metadata": {
            "title": "Test Video",
            "upload_date": "20260106",
            "duration": "0:01:00",
        },
        "speakers": [],
        "transcripts": [
            {
                "start": "00:00:10",
                "text": "The Road Traffic Bill is important.",
                "voice_id": 1,
                "speaker_id": "s_speaker_1",
            }
        ],
        "legislation": [
            {
                "id": "L_ROAD_TRAFFIC_BILL_1",
                "name": "Road Traffic Bill",
                "description": "Modernizes road traffic penalties",
                "source": "audio",
            }
        ],
    }

    ingestor.ingest_transcript_json(transcript_data, youtube_video_id="test_video")

    bill_queries = [
        call.args[0]
        for call in postgres.execute_update.call_args_list
        if "INSERT INTO bills" in call.args[0]
    ]
    assert len(bill_queries) == 1
