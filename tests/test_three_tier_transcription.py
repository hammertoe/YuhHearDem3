# Tests for three-tier transcription processing.
import pytest
from unittest.mock import patch

from lib.processors.three_tier_transcription import ThreeTierTranscriptionProcessor


@pytest.fixture
def sample_transcripts():
    """Sample transcript data for testing."""
    return [
        {
            "start": "00:36:00",
            "text": "Let us pray.",
            "voice_id": 2,
            "speaker_id": "s_reverend_1",
        },
        {
            "start": "00:36:10",
            "text": "Almighty God, our help in ages past.",
            "voice_id": 2,
            "speaker_id": "s_reverend_1",
        },
        {
            "start": "00:36:20",
            "text": "This House is in session.",
            "voice_id": 3,
            "speaker_id": "s_speaker_2",
        },
        {
            "start": "00:36:30",
            "text": "The Road Traffic Bill is now before the House.",
            "voice_id": 3,
            "speaker_id": "s_speaker_2",
        },
        {
            "start": "00:37:00",
            "text": "This bill addresses several key issues.",
            "voice_id": 3,
            "speaker_id": "s_speaker_2",
        },
    ]


def test_processor_init():
    """Test processor initialization."""
    processor = ThreeTierTranscriptionProcessor()
    print("✅ Processor initialization works")


def test_group_into_paragraphs(sample_transcripts):
    """Test transcript grouping into paragraphs."""
    processor = ThreeTierTranscriptionProcessor()

    result = processor.process_transcript_to_three_tier(
        "test_video", "Test Video", "2026-01-06", sample_transcripts
    )

    assert "video_metadata" in result
    assert "paragraphs" in result
    assert "sentences" in result
    assert "speakers" in result
    assert "legislation" in result

    assert len(result["paragraphs"]) == 2

    first_para = result["paragraphs"][0]
    assert first_para["speaker_id"] == "s_reverend_1"
    assert len(first_para["sentences"]) == 2
    assert first_para["sentence_count"] == 2

    print("✅ Transcript grouping works")


def test_paragraph_structure(sample_transcripts):
    """Test paragraph data structure."""
    processor = ThreeTierTranscriptionProcessor()

    result = processor.process_transcript_to_three_tier(
        "test_video", "Test Video", "2026-01-06", sample_transcripts
    )

    para = result["paragraphs"][0]

    assert "id" in para
    assert para["id"].startswith("test_video:")
    assert "youtube_video_id" in para
    assert "video_title" in para
    assert "video_date" in para
    assert "text" in para
    assert "speaker_id" in para
    assert "voice_id" in para
    assert "start_seconds" in para
    assert "end_seconds" in para
    assert "start_timestamp" in para
    assert "end_timestamp" in para
    assert "sentence_count" in para

    print("✅ Paragraph structure is correct")


def test_sentence_structure(sample_transcripts):
    """Test sentence data structure."""
    processor = ThreeTierTranscriptionProcessor()

    result = processor.process_transcript_to_three_tier(
        "test_video", "Test Video", "2026-01-06", sample_transcripts
    )

    sentences = result["sentences"]
    first_sentence = sentences[0]

    assert "id" in first_sentence
    assert first_sentence["id"].startswith("test_video:")
    assert "youtube_video_id" in first_sentence
    assert "video_title" in first_sentence
    assert "video_date" in first_sentence
    assert "text" in first_sentence
    assert "speaker_id" in first_sentence
    assert "voice_id" in first_sentence
    assert "paragraph_id" in first_sentence
    assert "sentence_order" in first_sentence
    assert "timestamp_str" in first_sentence
    assert "seconds_since_start" in first_sentence

    print("✅ Sentence structure is correct")


def test_speaker_extraction(sample_transcripts):
    """Test speaker extraction from transcripts."""
    processor = ThreeTierTranscriptionProcessor()

    result = processor.process_transcript_to_three_tier(
        "test_video", "Test Video", "2026-01-06", sample_transcripts
    )

    speakers = result["speakers"]

    assert len(speakers) == 2

    speaker_1 = speakers[0]
    assert speaker_1["speaker_id"] == "s_reverend_1"
    assert speaker_1["normalized_name"] == "reverend"
    assert speaker_1["title"] == ""
    assert speaker_1["position"] == "Unknown"
    assert speaker_1["role_in_video"] == "member"

    speaker_2 = speakers[1]
    assert speaker_2["speaker_id"] == "s_speaker_2"
    assert speaker_2["normalized_name"] == "speaker"
    assert speaker_2["title"] == ""
    assert speaker_2["position"] == "Unknown"

    assert all("first_appearance" in s for s in speakers)
    assert all("total_appearances" in s for s in speakers)
    reverend_speaker = next(s for s in speakers if s["speaker_id"] == "s_reverend_1")
    assert reverend_speaker["total_appearances"] == 2
    speaker_2_obj = next(s for s in speakers if s["speaker_id"] == "s_speaker_2")
    assert speaker_2_obj["total_appearances"] == 3

    print("✅ Speaker extraction works")


def test_legislation_extraction():
    """Test legislation extraction from transcripts."""
    processor = ThreeTierTranscriptionProcessor()

    result = processor.process_transcript_to_three_tier(
        "test_video",
        "Test Video",
        "2026-01-06",
        [
            {
                "start": "00:36:30",
                "text": "The Road Traffic Bill is now before the House.",
                "voice_id": 3,
                "speaker_id": "s_speaker_2",
            },
            {
                "start": "00:37:00",
                "text": "This bill addresses several key issues.",
                "voice_id": 3,
                "speaker_id": "s_speaker_2",
            },
        ],
    )

    legislation = result["legislation"]

    assert len(legislation) >= 1

    leg_1 = legislation[0]
    assert "Road Traffic Bill" in leg_1["name"]
    assert leg_1["source"] == "audio"

    print("✅ Legislation extraction works")


def test_timestamp_conversion():
    """Test timestamp parsing and conversion."""
    from lib.id_generators import (
        parse_timestamp_to_seconds,
        format_seconds_to_timestamp,
    )

    assert parse_timestamp_to_seconds("00:36:34") == 2194
    assert format_seconds_to_timestamp(2194) == "00:36:34"

    assert parse_timestamp_to_seconds("01:30:15") == 5415
    assert format_seconds_to_timestamp(5415) == "01:30:15"

    print("✅ Timestamp conversion works")


def test_save_output(sample_transcripts):
    """Test saving three-tier output."""
    processor = ThreeTierTranscriptionProcessor()

    data = processor.process_transcript_to_three_tier(
        "test_video", "Test Video", "2026-01-06", sample_transcripts[:3]
    )

    with patch("builtins.open", create=True) as mock_open:
        processor.save_three_tier_output(data, "test_output.json")
        mock_open.assert_called_once_with("test_output.json", "w")
        handle = mock_open.return_value.__enter__.return_value
        assert handle.write.call_count > 0

    print("✅ Output saving works")


def test_video_metadata():
    """Test video metadata structure."""
    processor = ThreeTierTranscriptionProcessor()

    result = processor.process_transcript_to_three_tier(
        "test_video", "Test Video", "2026-01-06", []
    )

    vm = result["video_metadata"]

    assert vm["youtube_id"] == "test_video"
    assert vm["title"] == "Test Video"
    assert vm["upload_date"] == "2026-01-06"
    assert "processed_at" in vm
    assert isinstance(vm["processed_at"], str)

    print("✅ Video metadata structure is correct")


def test_empty_transcripts():
    """Test handling of empty transcript list."""
    processor = ThreeTierTranscriptionProcessor()

    result = processor.process_transcript_to_three_tier(
        "test_video", "Test Video", "2026-01-06", []
    )

    assert len(result["paragraphs"]) == 0
    assert len(result["sentences"]) == 0
    assert len(result["speakers"]) == 0
    assert len(result["legislation"]) == 0

    assert "video_metadata" in result
    assert result["video_metadata"]["youtube_id"] == "test_video"

    print("✅ Empty transcript handling works")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
