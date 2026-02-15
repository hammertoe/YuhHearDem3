"""Generate speaker overrides from verification output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lib.id_generators import normalize_label
from lib.transcripts.speaker_verification import SpeakerVerificationResult


def load_verification_results(path: Path) -> list[dict[str, Any]]:
    """Load verification results from JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Invalid verification results payload")
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Invalid verification results format")
    return results


def build_override_entries(
    results: list[dict[str, Any]],
    min_confidence: float,
) -> list[dict[str, Any]]:
    """Build override entries from verification results."""
    overrides: list[dict[str, Any]] = []
    for row in results:
        result = row.get("result")
        if not isinstance(result, dict):
            continue
        try:
            parsed = SpeakerVerificationResult(**result)
        except Exception:
            continue

        if parsed.match_requested is True:
            continue
        if parsed.name is None:
            continue
        if parsed.confidence is None or parsed.confidence < min_confidence:
            continue

        overrides.append(
            {
                "video_id": row.get("video_id"),
                "start": row.get("timestamp"),
                "end": row.get("timestamp"),
                "old_speaker_id": row.get("speaker_id"),
                "new_speaker_name": parsed.name,
                "note": "Auto-generated from verification results",
            }
        )
    return overrides


def map_overrides_to_speaker_ids(
    overrides: list[dict[str, Any]],
    name_to_id: dict[str, str],
    normalized_to_id: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve override speaker names to speaker IDs."""
    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for entry in overrides:
        new_name = str(entry.get("new_speaker_name") or "").strip()
        new_id = name_to_id.get(new_name)
        if not new_id:
            new_id = normalized_to_id.get(normalize_label(new_name))
        if new_id:
            resolved.append({**entry, "new_speaker_id": new_id})
        else:
            unresolved.append(entry)

    return resolved, unresolved
