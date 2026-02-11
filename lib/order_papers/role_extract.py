"""Extract speaker role assignments from stored order paper JSON."""

from __future__ import annotations

from typing import Any


def extract_speaker_roles(parsed_json: Any) -> list[tuple[str, str]]:
    """Extract (speaker_name, role_label) pairs from an order paper parsed_json."""

    if not isinstance(parsed_json, dict):
        return []

    speakers = parsed_json.get("speakers")
    if not isinstance(speakers, list):
        return []

    out: list[tuple[str, str]] = []
    for s in speakers:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name", "") or "").strip()
        role = str(s.get("role", "") or "").strip()
        if name and role:
            out.append((name, role))

    return out
