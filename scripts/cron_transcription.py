"""Cron job script for periodic transcript ingestion."""

# ruff: noqa: E402

import argparse
import json
from datetime import datetime, timedelta
import os
import subprocess
import sys
from typing import Any


_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from lib.db.postgres_client import PostgresClient
from lib.order_papers.video_matcher import (
    MatchStatus,
    match_order_paper_for_video_metadata,
)


class CronJobManager:
    """Manage cron jobs for periodic transcript ingestion."""

    def __init__(self):
        self.transcribe_script = "transcribe.py"
        self.watch_file = ".transcription_watchlist.json"
        self.processed_videos = set()

    def _fetch_video_title(self, video_id: str) -> str | None:
        """Fetch YouTube title for a video id using yt-dlp metadata."""
        try:
            import yt_dlp

            url = f"https://www.youtube.com/watch?v={video_id}"
            ydl_opts = {"quiet": True, "skip_download": True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[arg-type]
                info = ydl.extract_info(url, download=False)
            title = str((info or {}).get("title", "") or "").strip()
            if title:
                return title
            return None
        except Exception as e:
            print(f"⚠️ Could not fetch title for {video_id}: {e}")
            return None

    def _resolve_video_title(self, video_id: str, video_title: str | None) -> str:
        provided = (video_title or "").strip()
        if provided:
            return provided

        fetched = self._fetch_video_title(video_id)
        if fetched:
            print(f"✅ Fetched video title for {video_id}: {fetched}")
            return fetched

        fallback = f"Video {video_id}"
        print(f"⚠️ Using fallback title for {video_id}: {fallback}")
        return fallback

    def load_watchlist(self) -> dict[str, Any]:
        """Load video watchlist."""
        if not os.path.exists(self.watch_file):
            return {"videos": {}}

        with open(self.watch_file, "r") as f:
            return json.load(f)

    def save_watchlist(self, watchlist: dict[str, Any]) -> None:
        """Save video watchlist."""
        with open(self.watch_file, "w") as f:
            json.dump(watchlist, f, indent=2)

    def get_videos_to_process(self, watchlist: dict[str, Any]) -> list[str]:
        """Get videos that need processing."""
        videos = []

        for video_id, video_info in watchlist.get("videos", {}).items():
            last_processed = video_info.get("last_processed", None)
            auto_process = video_info.get("auto_process", False)

            if not auto_process:
                continue

            if not last_processed:
                videos.append(video_id)
                continue

            time_since_last = datetime.fromisoformat(last_processed)
            if time_since_last < datetime.now() - timedelta(hours=24):
                videos.append(video_id)

        return videos

    def process_video(
        self,
        video_id: str,
        video_title: str,
        segment_minutes: int = 30,
        max_segments: int | None = None,
    ) -> bool:
        """Process a single video."""
        print(f"\n{'=' * 80}")
        print(f"Processing: {video_id}")
        print(f"Title: {video_title}")
        print(f"{'=' * 80}")

        output_file = f"transcription_output_{video_id}.json"
        matched_order_paper_id = self._auto_match_order_paper(video_id, video_title)

        cmd = [
            "python",
            self.transcribe_script,
            "--video",
            video_id,
            "--output-file",
            output_file,
            "--segment-minutes",
            str(segment_minutes),
        ]

        if matched_order_paper_id:
            cmd.extend(["--order-paper-id", matched_order_paper_id])

        if max_segments:
            cmd.extend(["--max-segments", str(max_segments)])

        try:
            result = subprocess.run(
                cmd, check=True, timeout=timedelta(hours=4).total_seconds()
            )

            if result.returncode == 0:
                print(f"\n✅ Successfully processed: {video_id}")
                return True
            else:
                print(f"\n❌ Failed to process: {video_id}")
                return False

        except subprocess.TimeoutExpired:
            print(f"\n⏱️ Timeout processing: {video_id}")
            return False
        except Exception as e:
            print(f"\n❌ Error processing {video_id}: {e}")
            return False

    def _auto_match_order_paper(self, video_id: str, video_title: str) -> str | None:
        """Auto-match order paper for high-confidence candidates only."""
        try:
            with PostgresClient() as postgres:
                decision = match_order_paper_for_video_metadata(
                    postgres,
                    youtube_video_id=video_id,
                    video_title=video_title,
                    upload_date=None,
                    persist=True,
                )
            if decision.status == MatchStatus.AUTO_MATCHED and decision.order_paper_id:
                print(
                    f"✅ Auto-matched order paper {decision.order_paper_id} for {video_id} "
                    f"(score={decision.score:.1f})"
                )
                return decision.order_paper_id

            print(
                f"⚠️ No high-confidence order paper match for {video_id} "
                f"({video_title}); status={decision.status.value}, score={decision.score:.1f}"
            )
            return None
        except Exception as e:
            print(f"⚠️ Order paper auto-match failed for {video_id}: {e}")
            return None

    def run_scheduled_ingestion(self, watchlist: dict[str, Any]) -> None:
        """Run scheduled ingestion based on watchlist."""
        print("\n" + "=" * 80)
        print("Cron Job: Scheduled Transcript Ingestion")
        print("=" * 80)
        print(f"Started at: {datetime.now().isoformat()}")

        videos_to_process = self.get_videos_to_process(watchlist)

        if not videos_to_process:
            print("\n✅ No videos to process")
            return

        print(f"\nProcessing {len(videos_to_process)} video(s)...")

        results = []
        for video_id in videos_to_process:
            video_info = watchlist["videos"][video_id]
            video_title = video_info.get("title", f"Video {video_id}")

            success = self.process_video(
                video_id,
                video_title,
                segment_minutes=video_info.get("segment_minutes", 30),
                max_segments=video_info.get("max_segments", None),
            )

            results.append({"video_id": video_id, "success": success})

        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        print(f"\n{'=' * 80}")
        print(f"Results: {successful} succeeded, {failed} failed")
        print(f"{'=' * 80}")

        timestamp = datetime.now().isoformat()

        for result in results:
            video_id = result["video_id"]
            if result["success"]:
                watchlist["videos"][video_id]["last_processed"] = timestamp
                watchlist["videos"][video_id]["status"] = "processed"
            else:
                watchlist["videos"][video_id]["status"] = "failed"
                watchlist["videos"][video_id]["last_attempted"] = timestamp

        watchlist["last_run"] = timestamp
        self.save_watchlist(watchlist)

        print(f"\n✅ Completed at: {timestamp}")
        print(f"Watchlist saved to: {self.watch_file}")
        print("=" * 80)

    def add_video_to_watchlist(
        self,
        video_id: str,
        video_title: str | None,
        segment_minutes: int = 30,
        max_segments: int | None = None,
        auto_process: bool = False,
    ) -> None:
        """Add a video to the watchlist."""
        watchlist = self.load_watchlist()
        resolved_title = self._resolve_video_title(video_id, video_title)

        if "videos" not in watchlist:
            watchlist["videos"] = {}

        watchlist["videos"][video_id] = {
            "id": video_id,
            "title": resolved_title,
            "segment_minutes": segment_minutes,
            "max_segments": max_segments,
            "auto_process": auto_process,
            "added_at": datetime.now().isoformat(),
            "status": "pending",
            "last_processed": None,
        }

        self.save_watchlist(watchlist)

        print(f"✅ Added to watchlist: {video_id}")
        print(f"   Auto-process: {auto_process}")
        print(f"   Segment minutes: {segment_minutes}")
        print(f"   Max segments: {max_segments}")

    def add_videos_from_file(
        self,
        file_path: str,
        *,
        segment_minutes: int = 30,
        max_segments: int | None = None,
        auto_process: bool = False,
    ) -> None:
        """Add multiple videos from a text file.

        Each non-empty, non-comment line supports:
          - VIDEO_ID
          - VIDEO_ID|TITLE
        """
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        added = 0
        for line in lines:
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue

            if "|" in entry:
                video_id_raw, title_raw = entry.split("|", 1)
                video_id = video_id_raw.strip()
                title = title_raw.strip() or None
            else:
                video_id = entry
                title = None

            if not video_id:
                continue

            self.add_video_to_watchlist(
                video_id,
                title,
                segment_minutes=segment_minutes,
                max_segments=max_segments,
                auto_process=auto_process,
            )
            added += 1

        print(f"✅ Added {added} video(s) from {file_path}")

    def list_watchlist(self, watchlist: dict[str, Any] | None = None) -> None:
        """List all videos in watchlist."""
        watchlist = watchlist or self.load_watchlist()
        videos = watchlist.get("videos", {})

        if not videos:
            print("\nWatchlist is empty")
            return

        print("\n" + "=" * 80)
        print("Transcription Watchlist")
        print("=" * 80)

        for video_id, video_info in videos.items():
            status = video_info.get("status", "pending")
            last_processed = video_info.get("last_processed", "N/A")
            auto_process = video_info.get("auto_process", False)

            status_icon = {
                "pending": "⏳",
                "processing": "⏳",
                "processed": "✅",
                "failed": "❌",
            }

            print(f"\n{status_icon[status]} {video_id}")
            print(f"   Title: {video_info.get('title', 'N/A')}")
            print(f"   Auto-process: {auto_process}")
            print(f"   Last processed: {last_processed}")

        print("=" * 80)

    def update_watchlist_status(self, video_id: str, status: str) -> None:
        """Update video status in watchlist."""
        watchlist = self.load_watchlist()

        if "videos" not in watchlist:
            watchlist["videos"] = {}

        if video_id in watchlist["videos"]:
            watchlist["videos"][video_id]["status"] = status

            if status == "processed":
                watchlist["videos"][video_id]["last_processed"] = (
                    datetime.now().isoformat()
                )

        self.save_watchlist(watchlist)

        print(f"✅ Updated {video_id} status to: {status}")

    def remove_from_watchlist(self, video_id: str) -> None:
        """Remove video from watchlist."""
        watchlist = self.load_watchlist()

        if "videos" in watchlist and video_id in watchlist["videos"]:
            del watchlist["videos"][video_id]
            self.save_watchlist(watchlist)
            print(f"✅ Removed {video_id} from watchlist")
        else:
            print(f"⚠️ Video {video_id} not in watchlist")


def main():
    parser = argparse.ArgumentParser(
        description="Cron job manager for periodic transcript ingestion"
    )
    parser.add_argument(
        "--add-video",
        nargs="+",
        metavar="ADD_VIDEO_ARGS",
        help="Add video to watchlist (ID [title words] segment_minutes)",
    )
    parser.add_argument(
        "--add-from-file",
        help="Add videos from file (one VIDEO_ID or VIDEO_ID|TITLE per line)",
    )
    parser.add_argument(
        "--segment-minutes",
        type=int,
        default=30,
        help="Segment minutes for --add-from-file (default: 30)",
    )
    parser.add_argument(
        "--auto-process",
        action="store_true",
        help="Enable auto-processing for added video",
    )
    parser.add_argument(
        "--max-segments", type=int, help="Max segments for video processing"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all videos in watchlist"
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Run scheduled ingestion for all pending videos",
    )
    parser.add_argument(
        "--watch-file",
        default=".transcription_watchlist.json",
        help="Path to watchlist file",
    )
    parser.add_argument(
        "--status",
        nargs=2,
        metavar=("VIDEO_ID", "STATUS"),
        choices=["pending", "processing", "processed", "failed"],
        help="Update video status",
    )
    parser.add_argument(
        "--remove", help="Remove video from watchlist (requires --video-id)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Cron Job Manager - Phase 3: Real-time Ingestion")
    print("=" * 80)

    manager = CronJobManager()
    manager.watch_file = args.watch_file

    if args.add_video:
        if len(args.add_video) < 2:
            raise ValueError(
                "--add-video requires at least VIDEO_ID and SEGMENT_MINUTES"
            )

        video_id = args.add_video[0]
        segment_minutes = args.add_video[-1]
        title = " ".join(args.add_video[1:-1]).strip() or None

        manager.add_video_to_watchlist(
            video_id,
            title,
            int(segment_minutes),
            max_segments=args.max_segments,
            auto_process=args.auto_process,
        )

    elif args.add_from_file:
        manager.add_videos_from_file(
            args.add_from_file,
            segment_minutes=args.segment_minutes,
            max_segments=args.max_segments,
            auto_process=args.auto_process,
        )

    elif args.list:
        manager.list_watchlist()

    elif args.process:
        watchlist = manager.load_watchlist()
        manager.run_scheduled_ingestion(watchlist)

    elif args.status:
        manager.update_watchlist_status(args.status[0], args.status[1])

    elif args.remove:
        manager.remove_from_watchlist(args.remove)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
