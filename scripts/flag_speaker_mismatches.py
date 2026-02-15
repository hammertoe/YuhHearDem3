"""Flag likely speaker attribution issues and emit CSV."""

from __future__ import annotations

import argparse
import csv
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.db.postgres_client import PostgresClient  # noqa: E402
from lib.id_generators import format_seconds_to_timestamp  # noqa: E402
from lib.transcripts.speaker_mismatch import summarize_speaker_blocks  # noqa: E402


def _parse_video_ids(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {v.strip() for v in raw.split(",") if v.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Flag likely speaker mismatches")
    parser.add_argument("--output", default="speaker_mismatch_flags.csv")
    parser.add_argument("--video-id", help="Comma-separated list of video IDs")
    parser.add_argument("--min-seconds", type=int, default=120)
    parser.add_argument("--limit-per-video", type=int, default=3)
    parser.add_argument("--only-missing-roles", action="store_true")
    args = parser.parse_args()

    filter_ids = _parse_video_ids(args.video_id)

    with PostgresClient() as postgres:
        if filter_ids:
            video_ids = sorted(filter_ids)
        else:
            rows = postgres.execute_query("SELECT DISTINCT youtube_video_id FROM sentences")
            video_ids = sorted(row[0] for row in rows if row[0])

        role_rows = postgres.execute_query(
            "SELECT youtube_video_id, speaker_id, role_label FROM speaker_video_roles"
        )
        role_index: dict[tuple[str, str], str] = {}
        for video_id, speaker_id, role_label in role_rows:
            if not video_id or not speaker_id:
                continue
            role_index[(video_id, speaker_id)] = role_label or ""

        output_rows: list[dict[str, str]] = []

        for video_id in video_ids:
            sentences = postgres.execute_query(
                """
                SELECT speaker_id, seconds_since_start, voice_id
                FROM sentences
                WHERE youtube_video_id = %s
                ORDER BY seconds_since_start ASC, id ASC
                """,
                (video_id,),
            )
            if not sentences:
                continue

            sentence_dicts = [
                {
                    "speaker_id": row[0],
                    "seconds_since_start": int(row[1] or 0),
                    "voice_id": int(row[2]) if row[2] is not None else None,
                }
                for row in sentences
            ]

            summary = summarize_speaker_blocks(sentence_dicts)
            voice_to_speakers: dict[int, set[str]] = {}
            for row in sentence_dicts:
                if row["voice_id"] is None:
                    continue
                voice_to_speakers.setdefault(row["voice_id"], set()).add(row["speaker_id"])

            if summary:
                rows = postgres.execute_query(
                    "SELECT id, full_name FROM speakers WHERE id = ANY(%s)",
                    (list(summary.keys()),),
                )
            else:
                rows = []
            name_index = {row[0]: row[1] or "" for row in rows}

            flags: list[tuple[str, dict[str, str]]] = []
            for speaker_id, data in summary.items():
                role_label = role_index.get((video_id, speaker_id), "")
                has_role = bool(role_label)
                if args.only_missing_roles and has_role:
                    continue

                voice_id = data.voice_id
                voice_conflict = False
                if voice_id is not None:
                    voice_conflict = len(voice_to_speakers.get(voice_id, set())) > 1

                if data.total_seconds < args.min_seconds:
                    continue

                sample_times = ",".join(
                    format_seconds_to_timestamp(s) for s in data.sample_timestamps
                )
                flags.append(
                    (
                        speaker_id,
                        {
                            "video_id": video_id,
                            "speaker_id": speaker_id,
                            "speaker_name": name_index.get(speaker_id, ""),
                            "voice_id": str(voice_id) if voice_id is not None else "",
                            "sentence_count": str(data.sentence_count),
                            "total_seconds": str(data.total_seconds),
                            "has_role": str(has_role).lower(),
                            "role_label": role_label,
                            "voice_id_conflict": str(voice_conflict).lower(),
                            "sample_timestamps": sample_times,
                        },
                    )
                )

            flags.sort(key=lambda item: int(item[1]["total_seconds"]), reverse=True)
            for _speaker_id, row in flags[: args.limit_per_video]:
                output_rows.append(row)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "speaker_id",
                "speaker_name",
                "voice_id",
                "sentence_count",
                "total_seconds",
                "has_role",
                "role_label",
                "voice_id_conflict",
                "sample_timestamps",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"✅ Wrote {len(output_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
