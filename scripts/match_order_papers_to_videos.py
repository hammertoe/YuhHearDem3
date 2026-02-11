#!/usr/bin/env python3
"""Match order papers to videos with high-confidence auto-attach policy."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from lib.db.postgres_client import PostgresClient
from lib.order_papers.video_matcher import MatchStatus, match_order_paper_for_video


def _load_target_video_ids(
    postgres: PostgresClient,
    *,
    youtube_video_id: str | None,
    all_unmatched: bool,
    limit: int,
) -> list[str]:
    if youtube_video_id:
        return [youtube_video_id]

    if not all_unmatched:
        raise ValueError("Either --youtube-video-id or --all-unmatched is required")

    rows = postgres.execute_query(
        """
        SELECT v.youtube_id
        FROM videos v
        LEFT JOIN video_order_paper_matches m ON m.youtube_video_id = v.youtube_id
        WHERE m.youtube_video_id IS NULL
        ORDER BY v.upload_date DESC NULLS LAST, v.created_at DESC
        LIMIT %s
        """,
        (int(limit),),
    )
    return [str(row[0]) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-match order papers to videos (high-confidence only)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--youtube-video-id", help="Match one video")
    group.add_argument(
        "--all-unmatched",
        action="store_true",
        help="Match all videos that do not yet have a match decision",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum videos to process with --all-unmatched (default: 200)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute decisions without writing video_order_paper_matches",
    )
    args = parser.parse_args()

    try:
        with PostgresClient() as postgres:
            video_ids = _load_target_video_ids(
                postgres,
                youtube_video_id=args.youtube_video_id,
                all_unmatched=args.all_unmatched,
                limit=args.limit,
            )

            if not video_ids:
                print("✅ No videos to match")
                return 0

            auto_count = 0
            review_count = 0
            for video_id in video_ids:
                decision = match_order_paper_for_video(
                    postgres,
                    youtube_video_id=video_id,
                    persist=not args.dry_run,
                )

                if decision.status == MatchStatus.AUTO_MATCHED:
                    auto_count += 1
                    print(
                        f"✅ {video_id} -> {decision.order_paper_id} "
                        f"(score={decision.score:.1f}, confidence={decision.confidence.value})"
                    )
                else:
                    review_count += 1
                    print(
                        f"⚠️ {video_id} -> needs review "
                        f"(top={decision.order_paper_id}, score={decision.score:.1f}, "
                        f"confidence={decision.confidence.value})"
                    )

            mode = "DRY RUN" if args.dry_run else "APPLIED"
            print(
                f"\n{mode}: processed={len(video_ids)} auto_matched={auto_count} "
                f"needs_review={review_count}"
            )
            return 0
    except Exception as e:
        print(f"❌ Matching failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
