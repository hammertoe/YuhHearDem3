"""Paragraph splitter tests."""

import pytest
from lib.processors.paragraph_splitter import (
    group_transcripts_into_paragraphs,
    split_paragraph_into_sentences,
)


@pytest.fixture
def sample_transcripts():
    """Sample transcript data."""
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
    ]


def test_group_transcripts_by_speaker(sample_transcripts):
    """Test grouping transcripts into paragraphs."""
    paragraphs = group_transcripts_into_paragraphs("Syxyah7QIaM", sample_transcripts)

    assert len(paragraphs) == 2

    first_para = paragraphs[0]
    assert first_para.speaker_id == "s_reverend_1"
    assert len(first_para.sentences) == 2
    assert "Let us pray." in first_para.get_text()
    assert "Almighty God" in first_para.get_text()

    second_para = paragraphs[1]
    assert second_para.speaker_id == "s_speaker_2"
    assert len(second_para.sentences) == 2
    assert "This House is in session." in second_para.get_text()

    print("✅ Transcript grouping by speaker works")


def test_paragraph_to_dict(sample_transcripts):
    """Test paragraph conversion to dictionary."""
    paragraphs = group_transcripts_into_paragraphs("Syxyah7QIaM", sample_transcripts)

    first_para = paragraphs[0]
    para_dict = first_para.to_dict()

    assert "id" in para_dict
    assert "youtube_video_id" in para_dict
    assert "start_seconds" in para_dict
    assert "end_seconds" in para_dict
    assert "text" in para_dict
    assert "speaker_id" in para_dict
    assert "sentence_count" in para_dict

    assert para_dict["speaker_id"] == "s_reverend_1"
    assert para_dict["sentence_count"] == 2

    print("✅ Paragraph to dict conversion works")


def test_paragraph_timestamps(sample_transcripts):
    """Test paragraph timestamp calculation."""
    paragraphs = group_transcripts_into_paragraphs("Syxyah7QIaM", sample_transcripts)

    first_para = paragraphs[0]
    assert first_para.start_seconds == 2160
    assert first_para.end_seconds == 2170
    assert first_para.start_timestamp == "00:36:00"
    assert first_para.end_timestamp == "00:36:10"

    print("✅ Paragraph timestamp calculation works")


def test_split_paragraph_to_sentences(sample_transcripts):
    """Test splitting paragraph into sentences."""
    paragraphs = group_transcripts_into_paragraphs("Syxyah7QIaM", sample_transcripts)

    first_para = paragraphs[0]
    sentences = split_paragraph_into_sentences(first_para, "Syxyah7QIaM")

    assert len(sentences) == 2

    first_sent = sentences[0]
    assert first_sent["id"] == "Syxyah7QIaM:2160"
    assert first_sent["text"] == "Let us pray."
    assert first_sent["speaker_id"] == "s_reverend_1"
    assert first_sent["sentence_order"] == 1
    assert first_sent["paragraph_id"] == first_para.id

    second_sent = sentences[1]
    assert second_sent["id"] == "Syxyah7QIaM:2170"
    assert second_sent["sentence_order"] == 2

    print("✅ Paragraph to sentence splitting works")


def test_empty_transcripts():
    """Test handling of empty transcript list."""
    paragraphs = group_transcripts_into_paragraphs("Syxyah7QIaM", [])
    assert len(paragraphs) == 0
    print("✅ Empty transcript handling works")


def test_single_speaker_multiple_paragraphs():
    """Test multiple paragraphs from same speaker (interleaved)."""
    transcripts = [
        {
            "start": "00:36:00",
            "text": "First point.",
            "voice_id": 1,
            "speaker_id": "s_speaker_1",
        },
        {
            "start": "00:36:10",
            "text": "Second point.",
            "voice_id": 1,
            "speaker_id": "s_speaker_1",
        },
        {
            "start": "00:36:20",
            "text": "Other speaker.",
            "voice_id": 2,
            "speaker_id": "s_speaker_2",
        },
        {
            "start": "00:36:30",
            "text": "Third point.",
            "voice_id": 1,
            "speaker_id": "s_speaker_1",
        },
        {
            "start": "00:36:40",
            "text": "Fourth point.",
            "voice_id": 1,
            "speaker_id": "s_speaker_1",
        },
    ]

    paragraphs = group_transcripts_into_paragraphs("test", transcripts)
    assert len(paragraphs) == 3

    assert paragraphs[0].speaker_id == "s_speaker_1"
    assert len(paragraphs[0].sentences) == 2

    assert paragraphs[1].speaker_id == "s_speaker_2"
    assert len(paragraphs[1].sentences) == 1

    assert paragraphs[2].speaker_id == "s_speaker_1"
    assert len(paragraphs[2].sentences) == 2

    print("✅ Multiple paragraphs from same speaker work")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
