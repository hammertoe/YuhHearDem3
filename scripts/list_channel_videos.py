#!/usr/bin/env python3
"""List all videos and live streams from a YouTube channel published in 2025."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import yt_dlp


def fetch_video_details(
    video_id: str, year: int, include_livestreams: bool
) -> dict[str, Any] | None:
    """Fetch details for a single video. Returns None if filtered out or error."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
    }

    video_url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            video_info = ydl.extract_info(video_url, download=False)
    except Exception:
        return None

    if not video_info:
        return None

    title = video_info.get("title", "Unknown")
    upload_date_str = video_info.get("upload_date")
    duration = video_info.get("duration")

    if not upload_date_str:
        return None

    try:
        upload_date = datetime.strptime(upload_date_str, "%Y%m%d")
        upload_year = upload_date.year
    except (ValueError, TypeError):
        return None

    if upload_year != year:
        return None

    is_live = video_info.get("is_live", False)
    was_live = video_info.get("was_live", False)

    if not include_livestreams and (is_live or was_live):
        return None

    return {
        "video_id": video_id,
        "title": title,
        "url": video_url,
        "upload_date": upload_date.strftime("%Y-%m-%d"),
        "duration_seconds": duration,
        "duration_formatted": format_duration(duration) if duration else "N/A",
        "is_live": is_live,
        "was_live": was_live,
        "live_status": video_info.get("live_status", "unknown"),
    }


def list_channel_videos(
    channel_url: str,
    year: int = 2025,
    include_livestreams: bool = True,
    plain: bool = False,
    max_workers: int = 5,
) -> list[dict[str, Any]]:
    """List all videos from a channel filtered by year.

    Args:
        channel_url: YouTube channel URL (e.g., https://www.youtube.com/@channel/streams)
        year: Filter videos by this year (default: 2025)
        include_livestreams: Include live streams and premiered videos
        plain: Suppress progress output
        max_workers: Number of parallel workers for fetching video details

    Returns:
        List of video metadata dictionaries
    """
    videos = []

    def log(msg: str) -> None:
        if not plain:
            print(msg, flush=True)

    # First pass: get flat list quickly
    ydl_opts_flat = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlistend": None,
    }

    log(f"Fetching video list from: {channel_url}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts_flat) as ydl:
            playlist_info = ydl.extract_info(channel_url, download=False)
    except Exception as e:
        log(f"Error fetching channel: {e}")
        return []

    if not playlist_info or "entries" not in playlist_info:
        log("No videos found or channel is private")
        return []

    entries = playlist_info["entries"]
    total_entries = len(entries) if entries else 0
    channel_title = playlist_info.get("title", "Unknown Channel")

    log(f"Channel: {channel_title}")
    log(f"Total videos in playlist: {total_entries}")

    if not entries:
        log("No entries found")
        return []

    # Get list of video IDs
    video_ids = [entry.get("id") for entry in entries if entry and entry.get("id")]

    log(
        f"Fetching details for {len(video_ids)} videos from {year} using {max_workers} workers..."
    )

    # Process videos in parallel
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_video_details, vid, year, include_livestreams): vid
            for vid in video_ids
        }

        for future in as_completed(futures):
            completed += 1
            if not plain and completed % 10 == 0:
                print(
                    f"  Processed {completed}/{len(video_ids)}...", end="\r", flush=True
                )

            result = future.result()
            if result:
                videos.append(result)

    if not plain:
        print(f"  Processed {completed} videos.{' ' * 20}", flush=True)
        log(f"Found {len(videos)} videos from {year}")

    # Sort by upload date
    videos.sort(key=lambda x: x["upload_date"], reverse=True)

    return videos


def format_duration(seconds: int | None) -> str:
    """Format duration in seconds to HH:MM:SS."""
    if not seconds:
        return "N/A"

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def print_videos(videos: list[dict[str, Any]]) -> None:
    """Print videos in a readable format."""
    if not videos:
        print("No videos found for the specified criteria.", flush=True)
        return

    print(f"\n{'=' * 100}", flush=True)
    print(f"Found {len(videos)} video(s) from 2025", flush=True)
    print(f"{'=' * 100}\n", flush=True)

    for i, video in enumerate(videos, 1):
        live_indicator = ""
        if video.get("was_live"):
            live_indicator = " [LIVE STREAM]"
        elif video.get("is_live"):
            live_indicator = " [CURRENTLY LIVE]"

        print(f"{i}. {video['title']}{live_indicator}", flush=True)
        print(f"   ID: {video['video_id']}", flush=True)
        print(f"   URL: {video['url']}", flush=True)
        print(f"   Upload Date: {video['upload_date']}", flush=True)
        print(f"   Duration: {video['duration_formatted']}", flush=True)
        print(flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="List all videos and live streams from a YouTube channel published in 2025"
    )
    parser.add_argument(
        "--channel-url",
        default="https://www.youtube.com/@barbadosparliamentchannel/streams",
        help="YouTube channel URL (default: Barbados Parliament streams)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Filter by year (default: 2025)",
    )
    parser.add_argument(
        "--exclude-livestreams",
        action="store_true",
        help="Exclude live streams and premiered videos",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file (JSON format)",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Output only video IDs, one per line",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5)",
    )

    args = parser.parse_args()

    if not args.plain:
        print("=" * 100, flush=True)
        print("YouTube Channel Video Lister", flush=True)
        print("=" * 100, flush=True)

    videos = list_channel_videos(
        channel_url=args.channel_url,
        year=args.year,
        include_livestreams=not args.exclude_livestreams,
        plain=args.plain,
        max_workers=args.workers,
    )

    if args.plain:
        for video in videos:
            print(video["video_id"], flush=True)
        return

    print_videos(videos)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(videos, f, indent=2)
        print(f"✅ Saved {len(videos)} videos to: {args.output}", flush=True)

    # Also print as video IDs (useful for watchlist)
    if videos:
        print(f"\n{'=' * 100}", flush=True)
        print("Video IDs (for use with cron_transcription.py --add-video):", flush=True)
        print(f"{'=' * 100}", flush=True)
        for video in videos:
            print(
                f"python scripts/cron_transcription.py --add-video {video['video_id']} 30",
                flush=True,
            )


if __name__ == "__main__":
    main()
