"""Speaker mismatch heuristics tests."""

from lib.id_generators import format_seconds_to_timestamp
from lib.transcripts.speaker_mismatch import (
    build_speaker_blocks,
    pick_sample_timestamps,
    summarize_speaker_blocks,
)


def test_build_speaker_blocks_groups_by_speaker() -> None:
    """Builds contiguous blocks per speaker in order."""
    sentences = [
        {"speaker_id": "s_1", "seconds_since_start": 10, "voice_id": 1},
        {"speaker_id": "s_1", "seconds_since_start": 20, "voice_id": 1},
        {"speaker_id": "s_2", "seconds_since_start": 30, "voice_id": 2},
        {"speaker_id": "s_2", "seconds_since_start": 40, "voice_id": 2},
        {"speaker_id": "s_1", "seconds_since_start": 50, "voice_id": 1},
    ]

    blocks = build_speaker_blocks(sentences)

    assert [b.speaker_id for b in blocks] == ["s_1", "s_2", "s_1"]
    assert blocks[0].start_seconds == 10
    assert blocks[0].end_seconds == 20
    assert blocks[1].start_seconds == 30
    assert blocks[1].end_seconds == 40


def test_pick_sample_timestamps_returns_start_mid_end() -> None:
    """Pick samples from a block."""
    seconds = pick_sample_timestamps(100, 160)
    assert seconds == [100, 130, 160]


def test_summarize_speaker_blocks_returns_longest_block() -> None:
    """Summarizes blocks and chooses longest block for samples."""
    sentences = [
        {"speaker_id": "s_1", "seconds_since_start": 10, "voice_id": 1},
        {"speaker_id": "s_1", "seconds_since_start": 20, "voice_id": 1},
        {"speaker_id": "s_2", "seconds_since_start": 30, "voice_id": 2},
        {"speaker_id": "s_2", "seconds_since_start": 90, "voice_id": 2},
        {"speaker_id": "s_1", "seconds_since_start": 110, "voice_id": 1},
    ]

    summary = summarize_speaker_blocks(sentences)

    assert summary["s_2"].longest_block_seconds == 60
    assert summary["s_2"].sample_timestamps == [30, 60, 90]
    assert format_seconds_to_timestamp(summary["s_2"].sample_timestamps[0]) == "00:00:30"
