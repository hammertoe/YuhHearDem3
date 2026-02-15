"""Speaker verification tool tests."""

import csv
from pathlib import Path

from lib.transcripts.speaker_verification import (
    SpeakerVerificationResult,
    build_verification_prompt,
    is_actionable_result,
    parse_flags_csv,
)


def test_parse_flags_csv_parses_sample_timestamps(tmp_path: Path) -> None:
    """Parses sample timestamps into seconds."""
    csv_path = tmp_path / "flags.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "speaker_id",
                "speaker_name",
                "voice_id",
                "sentence_count",
                "total_seconds",
                "has_role",
                "role_label",
                "voice_id_conflict",
                "sample_timestamps",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "video_id": "CtEXU7pbfO0",
                "speaker_id": "s_santia_bradshaw_1",
                "speaker_name": "santia bradshaw",
                "voice_id": "3",
                "sentence_count": "10",
                "total_seconds": "120",
                "has_role": "false",
                "role_label": "",
                "voice_id_conflict": "true",
                "sample_timestamps": "00:45:08,00:45:23,00:45:38",
            }
        )

    flags = parse_flags_csv(csv_path)

    assert len(flags) == 1
    assert flags[0].sample_seconds == [2708, 2723, 2738]


def test_build_verification_prompt_mentions_on_screen_text() -> None:
    """Prompt should instruct model to read on-screen text."""
    prompt = build_verification_prompt("Test Video", "00:45:08", "Hon. Test")

    assert "on-screen" in prompt.lower()
    assert "json" in prompt.lower()


def test_is_actionable_result_requires_confident_mismatch() -> None:
    """Actionable only when mismatch is confident."""
    result = SpeakerVerificationResult(
        name="Hon. Someone",
        on_screen_text="Hon. Someone",
        match_requested=False,
        confidence=0.9,
    )

    assert is_actionable_result(result, 0.85) is True
    assert is_actionable_result(result, 0.95) is False
