"""Paragraph splitter to group sentences by speaker."""

from __future__ import annotations

from typing import Any

from lib.id_generators import (
    parse_timestamp_to_seconds,
    generate_paragraph_id,
)


class Paragraph:
    """Represents a paragraph (speaker's coherent thought)."""

    def __init__(self, youtube_video_id: str, sentences: list[dict[str, Any]]):
        self.youtube_video_id = youtube_video_id
        self.sentences = sentences
        self.speaker_id = sentences[0]["speaker_id"]
        # Some inputs may omit voice_id; treat as unknown.
        self.voice_id = sentences[0].get("voice_id", 0)
        self.start_seconds = parse_timestamp_to_seconds(sentences[0]["start"])
        self.end_seconds = parse_timestamp_to_seconds(sentences[-1]["start"])
        self.id = generate_paragraph_id(self.youtube_video_id, self.start_seconds)
        self.start_timestamp = sentences[0]["start"]
        self.end_timestamp = sentences[-1]["start"]

    def get_text(self) -> str:
        """Combine all sentences into paragraph text."""
        return " ".join(s["text"] for s in self.sentences)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "youtube_video_id": self.youtube_video_id,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "text": self.get_text(),
            "speaker_id": self.speaker_id,
            "voice_id": self.voice_id,
            "sentence_count": len(self.sentences),
            # Keep original sentence dicts for JSON exports/debugging.
            "sentences": self.sentences,
        }


def group_transcripts_into_paragraphs(
    youtube_video_id: str, transcripts: list[dict[str, Any]]
) -> list[Paragraph]:
    """Group transcript entries into paragraphs by speaker.

    Groups consecutive sentences by the same speaker to create
    coherent semantic units (paragraphs).
    """
    if not transcripts:
        return []

    paragraphs = []
    current_speaker = None
    current_sentences = []

    for entry in transcripts:
        if entry["speaker_id"] != current_speaker:
            if current_speaker is not None:
                paragraphs.append(Paragraph(youtube_video_id, current_sentences))

            current_speaker = entry["speaker_id"]
            current_sentences = []

        current_sentences.append(entry)

    if current_sentences:
        paragraphs.append(Paragraph(youtube_video_id, current_sentences))

    return paragraphs


def split_paragraph_into_sentences(
    paragraph: Paragraph,
    youtube_video_id: str,
    video_date: str | None = None,
    video_title: str | None = None,
    existing_sentence_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Split a paragraph into individual sentences with IDs."""
    sentences = []
    seen_ids = existing_sentence_ids if existing_sentence_ids is not None else set()

    for i, entry in enumerate(paragraph.sentences):
        start_seconds = parse_timestamp_to_seconds(entry["start"])
        base_sentence_id = f"{youtube_video_id}:{start_seconds}"
        sentence_id = base_sentence_id
        suffix = 2
        while sentence_id in seen_ids:
            sentence_id = f"{base_sentence_id}_{suffix}"
            suffix += 1
        seen_ids.add(sentence_id)

        sentences.append(
            {
                "id": sentence_id,
                "youtube_video_id": youtube_video_id,
                "seconds_since_start": start_seconds,
                "timestamp_str": entry["start"],
                "text": entry["text"],
                "speaker_id": paragraph.speaker_id,
                "voice_id": paragraph.voice_id,
                "paragraph_id": paragraph.id,
                "sentence_order": i + 1,
                "video_date": video_date,
                "video_title": video_title,
            }
        )

    return sentences


def combine_sentences_to_text(sentences: list[dict[str, Any]]) -> str:
    """Combine a list of sentences into paragraph text."""
    return " ".join(s["text"] for s in sentences)
