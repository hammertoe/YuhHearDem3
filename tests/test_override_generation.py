"""Override generation tests."""

import json
from pathlib import Path

from lib.transcripts.override_generation import (
    build_override_entries,
    load_verification_results,
    map_overrides_to_speaker_ids,
)


def test_load_verification_results_parses_payload(tmp_path: Path) -> None:
    """Loads verification result payload."""
    payload = {
        "generated_at": "2026-02-14T00:00:00",
        "model": "gemini-2.5-flash",
        "results": [
            {
                "video_id": "01-AHbq9q1s",
                "speaker_id": "s_old",
                "timestamp": "00:38:56",
                "requested_name": "Old Name",
                "result": {
                    "name": "New Name",
                    "on_screen_text": "New Name",
                    "match_requested": False,
                    "confidence": 0.9,
                },
                "error": None,
            }
        ],
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    results = load_verification_results(path)

    assert len(results) == 1
    assert results[0]["video_id"] == "01-AHbq9q1s"


def test_build_override_entries_filters_non_actionable() -> None:
    """Only actionable mismatches become overrides."""
    results = [
        {
            "video_id": "01-AHbq9q1s",
            "speaker_id": "s_old",
            "timestamp": "00:38:56",
            "requested_name": "Old Name",
            "result": {
                "name": "New Name",
                "on_screen_text": "New Name",
                "match_requested": False,
                "confidence": 0.9,
            },
            "error": None,
        },
        {
            "video_id": "01-AHbq9q1s",
            "speaker_id": "s_old",
            "timestamp": "00:39:56",
            "requested_name": "Old Name",
            "result": {
                "name": "Old Name",
                "on_screen_text": "Old Name",
                "match_requested": True,
                "confidence": 0.9,
            },
            "error": None,
        },
    ]

    overrides = build_override_entries(results, 0.85)

    assert len(overrides) == 1
    assert overrides[0]["new_speaker_name"] == "New Name"


def test_map_overrides_to_speaker_ids() -> None:
    """Resolves speaker names into ids."""
    overrides = [
        {"new_speaker_name": "New Name", "video_id": "vid"},
        {"new_speaker_name": "Unknown", "video_id": "vid"},
    ]
    name_to_id = {"New Name": "s_new"}
    normalized_to_id = {"newname": "s_new"}

    resolved, unresolved = map_overrides_to_speaker_ids(overrides, name_to_id, normalized_to_id)

    assert resolved == [
        {"new_speaker_name": "New Name", "video_id": "vid", "new_speaker_id": "s_new"}
    ]
    assert unresolved == [{"new_speaker_name": "Unknown", "video_id": "vid"}]
