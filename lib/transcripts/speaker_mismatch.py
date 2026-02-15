"""Speaker mismatch heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpeakerBlock:
    speaker_id: str
    voice_id: int | None
    start_seconds: int
    end_seconds: int
    sentence_count: int


@dataclass(frozen=True)
class SpeakerSummary:
    speaker_id: str
    voice_id: int | None
    sentence_count: int
    total_seconds: int
    longest_block_seconds: int
    sample_timestamps: list[int]


def build_speaker_blocks(sentences: list[dict[str, Any]]) -> list[SpeakerBlock]:
    """Group contiguous sentences by speaker."""
    if not sentences:
        return []

    ordered = sorted(
        sentences,
        key=lambda s: (int(s.get("seconds_since_start", 0)), str(s.get("speaker_id", ""))),
    )

    blocks: list[SpeakerBlock] = []
    current_id = ordered[0].get("speaker_id")
    current_voice = ordered[0].get("voice_id")
    start_seconds = int(ordered[0].get("seconds_since_start", 0))
    end_seconds = start_seconds
    count = 1

    for entry in ordered[1:]:
        speaker_id = entry.get("speaker_id")
        voice_id = entry.get("voice_id")
        seconds = int(entry.get("seconds_since_start", 0))
        if speaker_id != current_id:
            blocks.append(
                SpeakerBlock(
                    speaker_id=str(current_id),
                    voice_id=int(current_voice) if current_voice is not None else None,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    sentence_count=count,
                )
            )
            current_id = speaker_id
            current_voice = voice_id
            start_seconds = seconds
            end_seconds = seconds
            count = 1
        else:
            end_seconds = max(end_seconds, seconds)
            count += 1

    blocks.append(
        SpeakerBlock(
            speaker_id=str(current_id),
            voice_id=int(current_voice) if current_voice is not None else None,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            sentence_count=count,
        )
    )
    return blocks


def pick_sample_timestamps(start_seconds: int, end_seconds: int) -> list[int]:
    """Pick start/mid/end samples for a block."""
    mid = start_seconds + ((end_seconds - start_seconds) // 2)
    return [start_seconds, mid, end_seconds]


def summarize_speaker_blocks(sentences: list[dict[str, Any]]) -> dict[str, SpeakerSummary]:
    """Summarize speaker blocks and select longest block samples."""
    blocks = build_speaker_blocks(sentences)
    summary: dict[str, SpeakerSummary] = {}

    for block in blocks:
        duration = max(0, block.end_seconds - block.start_seconds)
        existing = summary.get(block.speaker_id)
        total_seconds = duration + (existing.total_seconds if existing else 0)
        sentence_count = block.sentence_count + (existing.sentence_count if existing else 0)
        longest_block_seconds = max(duration, existing.longest_block_seconds if existing else 0)

        if not existing or duration >= existing.longest_block_seconds:
            sample_timestamps = pick_sample_timestamps(block.start_seconds, block.end_seconds)
        else:
            sample_timestamps = existing.sample_timestamps

        summary[block.speaker_id] = SpeakerSummary(
            speaker_id=block.speaker_id,
            voice_id=block.voice_id,
            sentence_count=sentence_count,
            total_seconds=total_seconds,
            longest_block_seconds=longest_block_seconds,
            sample_timestamps=sample_timestamps,
        )

    return summary
