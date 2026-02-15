"""Speaker verification helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pydantic

from lib.id_generators import parse_timestamp_to_seconds


@dataclass(frozen=True)
class SpeakerMismatchFlag:
    video_id: str
    speaker_id: str
    speaker_name: str
    voice_id: int | None
    sample_seconds: list[int]


class SpeakerVerificationResult(pydantic.BaseModel):
    name: str | None
    on_screen_text: str | None = None
    match_requested: bool | None = None
    confidence: float | None = None


def parse_flags_csv(path: Path) -> list[SpeakerMismatchFlag]:
    """Parse CSV output from flag_speaker_mismatches script."""
    import csv

    flags: list[SpeakerMismatchFlag] = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = str(row.get("video_id") or "").strip()
            speaker_id = str(row.get("speaker_id") or "").strip()
            speaker_name = str(row.get("speaker_name") or "").strip()
            voice_id_raw = str(row.get("voice_id") or "").strip()
            voice_id = int(voice_id_raw) if voice_id_raw else None

            sample_raw = str(row.get("sample_timestamps") or "").strip()
            sample_seconds = []
            if sample_raw:
                for token in sample_raw.split(","):
                    token = token.strip()
                    if token:
                        sample_seconds.append(parse_timestamp_to_seconds(token))

            if not video_id or not speaker_id or not sample_seconds:
                continue

            flags.append(
                SpeakerMismatchFlag(
                    video_id=video_id,
                    speaker_id=speaker_id,
                    speaker_name=speaker_name,
                    voice_id=voice_id,
                    sample_seconds=sample_seconds,
                )
            )
    return flags


def build_verification_prompt(video_title: str, timestamp_str: str, requested_name: str) -> str:
    """Build prompt for on-screen speaker verification."""
    return (
        "You are given a 1-second video clip from a parliamentary session.\n"
        f"Video title: {video_title}\n"
        f"Timestamp: {timestamp_str}\n"
        f"Expected speaker (from transcript): {requested_name}\n\n"
        "Task: Read the on-screen lower-third text and identify the speaker name.\n"
        "Use only on-screen text; do not infer from audio or context.\n"
        "If the text is unreadable, return name=null and confidence=0.\n"
        "Return JSON with fields: name, on_screen_text, match_requested, confidence.\n"
    )


def is_actionable_result(result: SpeakerVerificationResult | None, min_confidence: float) -> bool:
    """Decide whether a result is actionable for overrides."""
    if result is None:
        return False
    if result.match_requested is None:
        return False
    if result.match_requested:
        return False
    if result.name is None:
        return False
    if result.confidence is None:
        return False
    return result.confidence >= min_confidence
