"""Speaker override tests."""

import json
from pathlib import Path

from lib.transcripts.speaker_override import load_overrides, parse_override_entry


def test_parse_override_entry_builds_seconds() -> None:
    """Parses override entries into seconds with validation."""
    entry = {
        "video_id": "CtEXU7pbfO0",
        "start": "00:45:08",
        "end": "00:45:38",
        "new_speaker_id": "s_dr_the_hon_william_duguid_jp_mp_1",
        "old_speaker_id": "s_santia_bradshaw_1",
    }

    override = parse_override_entry(entry)

    assert override.video_id == "CtEXU7pbfO0"
    assert override.start_seconds == 2708
    assert override.end_seconds == 2738
    assert override.old_speaker_id == "s_santia_bradshaw_1"


def test_load_overrides_from_file(tmp_path: Path) -> None:
    """Loads override file with expected entries."""
    payload = {
        "overrides": [
            {
                "video_id": "CtEXU7pbfO0",
                "start": "00:45:08",
                "end": "00:45:38",
                "new_speaker_id": "s_dr_the_hon_william_duguid_jp_mp_1",
            }
        ]
    }
    file_path = tmp_path / "overrides.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    overrides = load_overrides(file_path)

    assert len(overrides) == 1
    assert overrides[0].video_id == "CtEXU7pbfO0"
