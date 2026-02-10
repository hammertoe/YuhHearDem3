"""Cron job script for periodic transcript ingestion."""

import argparse
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
import subprocess
import sys

from lib.processors.three_tier_transcription_processor import (
    ThreeTierTranscriptionProcessor,
)
from lib.id_generators import parse_timestamp_to_seconds


class CronJobManager:
    """Manage cron jobs for periodic transcript ingestion."""

    def __init__(self):
        self.processor = ThreeTierTranscriptionProcessor()
        self.transcribe_script = "transcribe.py"
        self.watch_file = ".transcription_watchlist.json"
        self.processed_videos = set()

    def load_watchlist(self) -> Dict[str, Any]:
        """Load video watchlist."""
        if not os.path.exists(self.watch_file):
            return {"videos": {}}

        with open(self.watch_file, "r") as f:
            return json.load(f)

    def save_watchlist(self, watchlist: Dict[str, Any]) -> None:
        """Save video watchlist."""
        with open(self.watch_file, "w") as f:
            json.dump(watchlist, f, indent=2)

    def get_videos_to_process(self, watchlist: Dict[str, Any]) -> List[str]:
        """Get videos that need processing."""
        videos = []

        for video_id, video_info in watchlist.get("videos", {}).items():
            last_processed = video_info.get("last_processed", None)
            auto_process = video_info.get("auto_process", False)

            if auto_process and last_processed:
                time_since_last = datetime.fromisoformat(last_processed)
                if time_since_last < datetime.now() - timedelta(hours=24):
                    videos.append(video_id)

        return videos

    def process_video(
        self,
        video_id: str,
        video_title: str,
        segment_minutes: int = 30,
        max_segments: Optional[int] = None,
    ) -> bool:
        """Process a single video."""
        print(f"\n{'=' * 80}")
        print(f"Processing: {video_id}")
        print(f"Title: {video_title}")
        print(f"{'=' * 80}")

        output_file = f"transcription_output_{video_id}.json"

        cmd = [
            "python",
            self.transcribe_script,
            "--output-file",
            output_file,
            "--segment-minutes",
            str(segment_minutes),
        ]

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

    def run_scheduled_ingestion(self, watchlist: Dict[str, Any]) -> None:
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
        video_title: str,
        segment_minutes: int = 30,
        max_segments: Optional[int] = None,
        auto_process: bool = False,
    ) -> None:
        """Add a video to the watchlist."""
        watchlist = self.load_watchlist()

        if "videos" not in watchlist:
            watchlist["videos"] = {}

        watchlist["videos"][video_id] = {
            "id": video_id,
            "title": video_title,
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

    def list_watchlist(self, watchlist: Optional[Dict[str, Any]] = None) -> None:
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
        nargs=3,
        metavar=("VIDEO_ID", "TITLE", "SEGMENT_MINUTES"),
        help="Add video to watchlist (ID title segment_minutes)",
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

    if args.add_video:
        video_id, title, segment_minutes = args.add_video
        manager.add_video_to_watchlist(
            video_id,
            title,
            segment_minutes,
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
