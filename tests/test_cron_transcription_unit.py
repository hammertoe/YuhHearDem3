from __future__ import annotations

import json
from datetime import datetime, timedelta

from scripts.cron_transcription import CronJobManager


def test_process_video_passes_video_id_and_matched_order_paper(monkeypatch) -> None:
    manager = CronJobManager()

    captured_cmd: list[str] = []

    class _Result:
        returncode = 0

    def _fake_run(cmd, check, timeout):
        del check, timeout
        captured_cmd.extend(cmd)
        return _Result()

    monkeypatch.setattr(
        manager,
        "_auto_match_order_paper",
        lambda *_args, **_kwargs: "op_h_20260113_one",
    )
    monkeypatch.setattr("scripts.cron_transcription.subprocess.run", _fake_run)

    ok = manager.process_video("abc123", "Title", segment_minutes=30)

    assert ok is True
    assert "--video" in captured_cmd
    assert "abc123" in captured_cmd
    assert "--order-paper-id" in captured_cmd
    assert "op_h_20260113_one" in captured_cmd


def test_add_video_to_watchlist_fetches_title_when_missing(
    tmp_path, monkeypatch
) -> None:
    manager = CronJobManager()
    manager.watch_file = str(tmp_path / "watchlist.json")

    monkeypatch.setattr(
        manager,
        "_fetch_video_title",
        lambda _video_id: (
            "The Honourable The House - Tuesday 20th January, 2026 - Part 1"
        ),
    )

    manager.add_video_to_watchlist("abc123", None, 30, auto_process=True)

    data = json.loads((tmp_path / "watchlist.json").read_text())
    assert data["videos"]["abc123"]["title"].startswith("The Honourable The House")


def test_add_video_to_watchlist_falls_back_when_title_lookup_fails(
    tmp_path, monkeypatch
) -> None:
    manager = CronJobManager()
    manager.watch_file = str(tmp_path / "watchlist.json")

    monkeypatch.setattr(manager, "_fetch_video_title", lambda _video_id: None)

    manager.add_video_to_watchlist("abc123", None, 30, auto_process=True)

    data = json.loads((tmp_path / "watchlist.json").read_text())
    assert data["videos"]["abc123"]["title"] == "Video abc123"


def test_add_videos_from_file_supports_manual_and_fetched_titles(
    tmp_path, monkeypatch
) -> None:
    manager = CronJobManager()
    manager.watch_file = str(tmp_path / "watchlist.json")

    source_file = tmp_path / "videos.txt"
    source_file.write_text(
        "# comment\nabc123\ndef456|Custom Title\n\nghi789\n", encoding="utf-8"
    )

    monkeypatch.setattr(
        manager, "_fetch_video_title", lambda video_id: f"Fetched {video_id}"
    )

    manager.add_videos_from_file(
        str(source_file),
        segment_minutes=25,
        max_segments=4,
        auto_process=True,
    )

    data = json.loads((tmp_path / "watchlist.json").read_text())
    assert data["videos"]["abc123"]["title"] == "Fetched abc123"
    assert data["videos"]["def456"]["title"] == "Custom Title"
    assert data["videos"]["ghi789"]["title"] == "Fetched ghi789"
    assert data["videos"]["abc123"]["segment_minutes"] == 25
    assert data["videos"]["abc123"]["max_segments"] == 4
    assert data["videos"]["abc123"]["auto_process"] is True


def test_get_videos_to_process_includes_new_auto_process_entries() -> None:
    manager = CronJobManager()

    watchlist = {
        "videos": {
            "new_video": {
                "auto_process": True,
                "last_processed": None,
            },
            "old_video": {
                "auto_process": True,
                "last_processed": (datetime.now() - timedelta(days=2)).isoformat(),
            },
            "recent_video": {
                "auto_process": True,
                "last_processed": (datetime.now() - timedelta(hours=2)).isoformat(),
            },
            "manual_video": {
                "auto_process": False,
                "last_processed": None,
            },
        }
    }

    video_ids = manager.get_videos_to_process(watchlist)

    assert "new_video" in video_ids
    assert "old_video" in video_ids
    assert "recent_video" not in video_ids
    assert "manual_video" not in video_ids
