"""Speaker override parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lib.id_generators import parse_timestamp_to_seconds


@dataclass(frozen=True)
class SpeakerOverride:
    video_id: str
    start_seconds: int
    end_seconds: int
    new_speaker_id: str
    old_speaker_id: str | None = None
    voice_id: int | None = None
    note: str | None = None


def parse_override_entry(entry: dict[str, Any]) -> SpeakerOverride:
    """Parse a single override entry to normalized seconds."""
    video_id = str(entry.get("video_id") or "").strip()
    new_speaker_id = str(entry.get("new_speaker_id") or "").strip()
    start = str(entry.get("start") or "").strip()
    end = str(entry.get("end") or "").strip()

    if not video_id or not new_speaker_id or not start or not end:
        raise ValueError("Override entry missing required fields")

    start_seconds = parse_timestamp_to_seconds(start)
    end_seconds = parse_timestamp_to_seconds(end)
    if start_seconds > end_seconds:
        raise ValueError("Override start must be <= end")

    old_speaker_id = entry.get("old_speaker_id")
    if old_speaker_id is not None:
        old_speaker_id = str(old_speaker_id).strip() or None

    voice_id = entry.get("voice_id")
    if voice_id is not None:
        voice_id = int(voice_id)

    note = entry.get("note")
    if note is not None:
        note = str(note).strip() or None

    return SpeakerOverride(
        video_id=video_id,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        new_speaker_id=new_speaker_id,
        old_speaker_id=old_speaker_id,
        voice_id=voice_id,
        note=note,
    )


def load_overrides(path: Path) -> list[SpeakerOverride]:
    """Load overrides from a JSON file."""
    data = Path(path).read_text(encoding="utf-8")
    payload = __import__("json").loads(data)
    entries = payload.get("overrides", []) if isinstance(payload, dict) else []
    return [parse_override_entry(entry) for entry in entries]
