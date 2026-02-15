"""Utilities for correcting malformed transcript timestamps."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from lib.id_generators import format_seconds_to_timestamp


@dataclass
class SentenceIdMapping:
    """Mapping between old and new sentence IDs plus cleaned transcripts."""

    old_to_new: dict[str, str]
    new_seconds_by_old_id: dict[str, int]
    new_timestamp_by_old_id: dict[str, str]
    cleaned_transcripts: list[dict[str, Any]]
    reasons: list[str]


def parse_duration_to_seconds(duration: str) -> int | None:
    """Parse video duration string (HH:MM:SS or MM:SS) into seconds."""
    raw = str(duration or "").strip()
    if not raw:
        return None
    parts = raw.split(":")
    if len(parts) == 3:
        hours, mins, secs = map(int, parts)
        return hours * 3600 + mins * 60 + secs
    if len(parts) == 2:
        mins, secs = map(int, parts)
        return mins * 60 + secs
    return None


def extract_video_id_from_filename(filename: str) -> str | None:
    """Extract YouTube id from transcription_output_<id>.json filenames."""
    base = os.path.basename(str(filename))
    match = re.match(r"^transcription_output_(.+)\.json$", base)
    if not match:
        return None
    return match.group(1)


def _candidate_seconds_from_parts(
    hours_raw: int, mins_raw: int, secs_raw: int, secs_token: str
) -> tuple[int | None, int | None, str | None]:
    hms = None
    if mins_raw <= 59 and secs_raw <= 59:
        hms = hours_raw * 3600 + mins_raw * 60 + secs_raw

    mmss = None
    mmss_reason = None
    has_millis = len(secs_token) > 2 or secs_raw > 59
    if mins_raw <= 59:
        mmss = hours_raw * 60 + mins_raw
        if has_millis:
            mmss_reason = "mmss_millis"
        elif hours_raw >= 24:
            mmss_reason = "mmss_hours"
        else:
            mmss_reason = "mmss_alt"
    elif hours_raw == 0 and secs_raw <= 59:
        mmss = mins_raw * 60 + secs_raw
        mmss_reason = "mmss_zero_hours"

    return hms, mmss, mmss_reason


def normalize_timestamp_str(
    raw: str,
    prev_seconds: int | None,
    duration_seconds: int | None,
) -> tuple[int, str]:
    """Normalize a timestamp string to seconds and report reason."""
    token = str(raw or "").strip()
    parts = token.split(":")
    if len(parts) == 2:
        mins, secs = map(int, parts)
        return mins * 60 + secs, "mmss"
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp format: {raw}")

    hours_raw, mins_raw, secs_raw = parts
    hours = int(hours_raw)
    mins = int(mins_raw)
    secs = int(secs_raw)

    hms, mmss, mmss_reason = _candidate_seconds_from_parts(hours, mins, secs, secs_raw)
    candidates: list[tuple[int, str]] = []
    if hms is not None:
        candidates.append((hms, "hms"))
    if mmss is not None:
        candidates.append((mmss, mmss_reason or "mmss"))
    if not candidates:
        raise ValueError(f"Invalid timestamp values: {raw}")

    if duration_seconds is not None:
        candidates = [c for c in candidates if c[0] <= duration_seconds]
    if not candidates:
        raise ValueError(f"Timestamp exceeds duration: {raw}")

    if prev_seconds is None:
        preferred = next((c for c in candidates if c[1].startswith("mmss_")), None)
        if preferred and preferred[1] in {"mmss_millis", "mmss_hours"}:
            return preferred
        for seconds, reason in candidates:
            if reason == "hms":
                return seconds, reason
        return candidates[0]

    monotonic = [(seconds, reason) for seconds, reason in candidates if seconds >= prev_seconds]
    if monotonic:
        monotonic.sort(key=lambda item: item[0] - prev_seconds)
        return monotonic[0]

    candidates.sort(key=lambda item: abs(item[0] - prev_seconds))
    return candidates[0][0], f"{candidates[0][1]}_non_monotonic"


def _assign_sentence_ids(
    youtube_video_id: str,
    seconds_list: list[int],
) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    for seconds in seconds_list:
        base_id = f"{youtube_video_id}:{seconds}"
        sentence_id = base_id
        suffix = 2
        while sentence_id in seen:
            sentence_id = f"{base_id}_{suffix}"
            suffix += 1
        seen.add(sentence_id)
        ids.append(sentence_id)
    return ids


def build_sentence_id_mapping(
    *,
    transcripts: list[dict[str, Any]],
    youtube_video_id: str,
    duration_seconds: int | None,
) -> SentenceIdMapping:
    """Build mapping from legacy sentence IDs to corrected IDs."""
    old_seconds: list[int] = []
    new_seconds: list[int] = []
    cleaned: list[dict[str, Any]] = []
    reasons: list[str] = []
    prev_seconds: int | None = None

    for entry in transcripts:
        raw = str(entry.get("start", "")).strip()
        parts = raw.split(":")
        if len(parts) == 3:
            hours, mins, secs = map(int, parts)
            legacy_seconds = hours * 3600 + mins * 60 + secs
        elif len(parts) == 2:
            mins, secs = map(int, parts)
            legacy_seconds = mins * 60 + secs
        else:
            raise ValueError(f"Invalid timestamp format: {raw}")
        old_seconds.append(legacy_seconds)

        fixed_seconds, reason = normalize_timestamp_str(raw, prev_seconds, duration_seconds)
        prev_seconds = fixed_seconds
        new_seconds.append(fixed_seconds)
        reasons.append(reason)

        cleaned_entry = dict(entry)
        cleaned_entry["start"] = format_seconds_to_timestamp(fixed_seconds)
        cleaned.append(cleaned_entry)

    old_ids = _assign_sentence_ids(youtube_video_id, old_seconds)
    new_ids = _assign_sentence_ids(youtube_video_id, new_seconds)

    old_to_new = dict(zip(old_ids, new_ids))
    new_seconds_by_old = dict(zip(old_ids, new_seconds))
    new_timestamp_by_old = dict(
        zip(old_ids, [format_seconds_to_timestamp(sec) for sec in new_seconds])
    )

    return SentenceIdMapping(
        old_to_new=old_to_new,
        new_seconds_by_old_id=new_seconds_by_old,
        new_timestamp_by_old_id=new_timestamp_by_old,
        cleaned_transcripts=cleaned,
        reasons=reasons,
    )


def build_sentence_id_mapping_from_existing_ids(
    *,
    transcripts: list[dict[str, Any]],
    youtube_video_id: str,
    duration_seconds: int | None,
    existing_ids: list[str],
) -> SentenceIdMapping:
    """Build mapping using existing sentence ID order."""
    if len(existing_ids) != len(transcripts):
        raise ValueError("Existing sentence count does not match transcript count")

    new_seconds: list[int] = []
    cleaned: list[dict[str, Any]] = []
    reasons: list[str] = []
    prev_seconds: int | None = None

    for entry in transcripts:
        raw = str(entry.get("start", "")).strip()
        fixed_seconds, reason = normalize_timestamp_str(raw, prev_seconds, duration_seconds)
        prev_seconds = fixed_seconds
        new_seconds.append(fixed_seconds)
        reasons.append(reason)

        cleaned_entry = dict(entry)
        cleaned_entry["start"] = format_seconds_to_timestamp(fixed_seconds)
        cleaned.append(cleaned_entry)

    new_ids = _assign_sentence_ids(youtube_video_id, new_seconds)
    old_to_new = dict(zip(existing_ids, new_ids))
    new_seconds_by_old = dict(zip(existing_ids, new_seconds))
    new_timestamp_by_old = dict(
        zip(existing_ids, [format_seconds_to_timestamp(sec) for sec in new_seconds])
    )

    return SentenceIdMapping(
        old_to_new=old_to_new,
        new_seconds_by_old_id=new_seconds_by_old,
        new_timestamp_by_old_id=new_timestamp_by_old,
        cleaned_transcripts=cleaned,
        reasons=reasons,
    )


def build_paragraph_id_mapping(
    *,
    transcripts: list[dict[str, Any]],
    youtube_video_id: str,
    duration_seconds: int | None,
) -> list[dict[str, Any]]:
    """Build paragraph ID updates based on corrected timestamps."""
    if not transcripts:
        return []

    legacy_seconds: list[int] = []
    fixed_seconds: list[int] = []
    fixed_timestamps: list[str] = []
    prev_seconds: int | None = None

    for entry in transcripts:
        raw = str(entry.get("start", "")).strip()
        parts = raw.split(":")
        if len(parts) == 3:
            hours, mins, secs = map(int, parts)
            legacy = hours * 3600 + mins * 60 + secs
        elif len(parts) == 2:
            mins, secs = map(int, parts)
            legacy = mins * 60 + secs
        else:
            raise ValueError(f"Invalid timestamp format: {raw}")
        legacy_seconds.append(legacy)

        fixed, _ = normalize_timestamp_str(raw, prev_seconds, duration_seconds)
        prev_seconds = fixed
        fixed_seconds.append(fixed)
        fixed_timestamps.append(format_seconds_to_timestamp(fixed))

    updates: list[dict[str, Any]] = []
    seen_old_ids: set[str] = set()
    seen_new_ids: set[str] = set()
    current_speaker = transcripts[0].get("speaker_id")
    start_idx = 0

    def flush(end_idx: int) -> None:
        old_start = legacy_seconds[start_idx]
        old_id = f"{youtube_video_id}:{old_start}"
        if old_id in seen_old_ids:
            return
        seen_old_ids.add(old_id)
        new_start = fixed_seconds[start_idx]
        new_end = fixed_seconds[end_idx]
        new_id = f"{youtube_video_id}:{new_start}"
        if new_id in seen_new_ids:
            return
        seen_new_ids.add(new_id)
        updates.append(
            {
                "old_id": old_id,
                "new_id": new_id,
                "new_start_seconds": new_start,
                "new_end_seconds": new_end,
                "new_start_timestamp": fixed_timestamps[start_idx],
                "new_end_timestamp": fixed_timestamps[end_idx],
            }
        )

    for idx, entry in enumerate(transcripts):
        speaker_id = entry.get("speaker_id")
        if speaker_id != current_speaker:
            flush(idx - 1)
            current_speaker = speaker_id
            start_idx = idx

    flush(len(transcripts) - 1)
    return updates
