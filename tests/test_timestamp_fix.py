"""Timestamp fixing tests."""

from lib.id_generators import format_seconds_to_timestamp
from lib.transcripts.timestamp_fix import (
    build_paragraph_id_mapping,
    build_sentence_id_mapping,
    build_sentence_id_mapping_from_existing_ids,
    extract_video_id_from_filename,
    normalize_timestamp_str,
    parse_duration_to_seconds,
)


def test_parse_duration_to_seconds():
    """Parses video duration strings to seconds."""
    assert parse_duration_to_seconds("2:36:56") == 9416
    assert parse_duration_to_seconds("36:12") == 2172


def test_normalize_timestamp_str_prefers_hms_when_plausible():
    """Uses HH:MM:SS when values are valid and within range."""
    seconds, reason = normalize_timestamp_str("0:3:19", None, 9416)
    assert format_seconds_to_timestamp(seconds) == "00:03:19"
    assert reason == "hms"


def test_normalize_timestamp_str_treats_millis_as_mmss():
    """Uses MM:SS when millis-like third part is present."""
    seconds, reason = normalize_timestamp_str("46:18:000", 2700, 9416)
    assert format_seconds_to_timestamp(seconds) == "00:46:18"
    assert reason == "mmss_millis"


def test_normalize_timestamp_str_treats_large_hours_as_minutes():
    """Treats large leading values as minutes for three-part strings."""
    seconds, reason = normalize_timestamp_str("49:08:00", 2800, 9416)
    assert format_seconds_to_timestamp(seconds) == "00:49:08"
    assert reason == "mmss_hours"


def test_normalize_timestamp_str_treats_zero_hours_large_minutes_as_mmss():
    """Treats 0:MM:SS with large minutes as MM:SS."""
    seconds, reason = normalize_timestamp_str("0:87:47", 5200, 9416)
    assert format_seconds_to_timestamp(seconds) == "01:27:47"
    assert reason == "mmss_zero_hours"


def test_build_sentence_id_mapping_preserves_suffixes():
    """Keeps duplicate suffixes stable between old and new ids."""
    transcripts = [
        {"start": "0:0:10", "text": "A", "speaker_id": "s_1", "voice_id": 1},
        {"start": "0:0:10", "text": "B", "speaker_id": "s_1", "voice_id": 1},
    ]
    mapping = build_sentence_id_mapping(
        transcripts=transcripts,
        youtube_video_id="abc123",
        duration_seconds=3600,
    )

    assert mapping.old_to_new["abc123:10"] == "abc123:10"
    assert mapping.old_to_new["abc123:10_2"] == "abc123:10_2"


def test_build_sentence_id_mapping_from_existing_ids_uses_order():
    """Uses existing id order when old ids don't match legacy parsing."""
    transcripts = [
        {"start": "0:0:10", "text": "A", "speaker_id": "s_1", "voice_id": 1},
        {"start": "0:0:20", "text": "B", "speaker_id": "s_1", "voice_id": 1},
    ]
    existing_ids = ["abc123:999", "abc123:1000"]

    mapping = build_sentence_id_mapping_from_existing_ids(
        transcripts=transcripts,
        youtube_video_id="abc123",
        duration_seconds=3600,
        existing_ids=existing_ids,
    )

    assert mapping.old_to_new == {
        "abc123:999": "abc123:10",
        "abc123:1000": "abc123:20",
    }


def test_extract_video_id_from_filename():
    """Extracts video ids from transcript filenames."""
    assert extract_video_id_from_filename("transcription_output_hIi-c4Q2lC4.json") == "hIi-c4Q2lC4"


def test_build_paragraph_id_mapping_dedupes_old_ids():
    """Dedupes paragraph updates when legacy ids collide."""
    transcripts = [
        {"start": "0:0:10", "text": "A", "speaker_id": "s_1", "voice_id": 1},
        {"start": "0:0:10", "text": "B", "speaker_id": "s_2", "voice_id": 2},
        {"start": "0:0:11", "text": "C", "speaker_id": "s_2", "voice_id": 2},
    ]
    updates = build_paragraph_id_mapping(
        transcripts=transcripts,
        youtube_video_id="abc123",
        duration_seconds=3600,
    )

    old_ids = [u["old_id"] for u in updates]
    assert len(old_ids) == len(set(old_ids))


def test_build_paragraph_id_mapping_dedupes_new_ids():
    """Dedupes paragraph updates when new ids would collide."""
    transcripts = [
        {"start": "0:0:10", "text": "A", "speaker_id": "s_1", "voice_id": 1},
        {"start": "0:0:10", "text": "B", "speaker_id": "s_2", "voice_id": 2},
    ]
    updates = build_paragraph_id_mapping(
        transcripts=transcripts,
        youtube_video_id="abc123",
        duration_seconds=3600,
    )

    new_ids = [u["new_id"] for u in updates]
    assert len(new_ids) == len(set(new_ids))
