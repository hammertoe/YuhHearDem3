"""Repair speaker labels using override rules."""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.db.postgres_client import PostgresClient  # noqa: E402
from lib.transcripts.speaker_override import load_overrides, SpeakerOverride  # noqa: E402


def _parse_video_ids(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {v.strip() for v in raw.split(",") if v.strip()}


def _override_where_clause(override: SpeakerOverride) -> tuple[str, list[object]]:
    clauses = [
        "youtube_video_id = %s",
        "seconds_since_start BETWEEN %s AND %s",
    ]
    params: list[object] = [override.video_id, override.start_seconds, override.end_seconds]

    if override.old_speaker_id:
        clauses.append("speaker_id = %s")
        params.append(override.old_speaker_id)
    if override.voice_id is not None:
        clauses.append("voice_id = %s")
        params.append(override.voice_id)

    return " AND ".join(clauses), params


def _update_edge_speakers(cursor, video_id: str) -> None:
    cursor.execute(
        "UPDATE kg_edges e SET speaker_ids = sub.speaker_ids "
        "FROM ("
        "  SELECT edge_id, array_agg(speaker_id ORDER BY min_ord) AS speaker_ids "
        "  FROM ("
        "    SELECT e.id AS edge_id, s.speaker_id, MIN(u.ord) AS min_ord "
        "    FROM kg_edges e "
        "    CROSS JOIN LATERAL unnest(e.utterance_ids) WITH ORDINALITY AS u(utt, ord) "
        "    JOIN sentences s ON s.id = u.utt "
        "    WHERE e.youtube_video_id = %s "
        "    GROUP BY e.id, s.speaker_id"
        "  ) ranked "
        "  GROUP BY edge_id"
        ") sub "
        "WHERE e.id = sub.edge_id",
        (video_id,),
    )


def _apply_override(cursor, override: SpeakerOverride, dry_run: bool) -> None:
    where_clause, params = _override_where_clause(override)
    if dry_run:
        cursor.execute(
            f"SELECT count(*) FROM sentences WHERE {where_clause}",
            params,
        )
        count = cursor.fetchone()[0]
        print(
            f"   {override.video_id} {override.start_seconds}-{override.end_seconds} "
            f"=> {override.new_speaker_id} (matches {count})"
        )
        return

    cursor.execute(
        f"SELECT DISTINCT paragraph_id FROM sentences WHERE {where_clause}",
        params,
    )
    paragraph_ids = [row[0] for row in cursor.fetchall()]

    cursor.execute(
        f"UPDATE sentences SET speaker_id = %s WHERE {where_clause}",
        [override.new_speaker_id, *params],
    )

    if paragraph_ids:
        placeholders = ",".join(["%s"] * len(paragraph_ids))
        cursor.execute(
            f"UPDATE paragraphs SET speaker_id = %s WHERE id IN ({placeholders})",
            [override.new_speaker_id, *paragraph_ids],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply speaker overrides to DB")
    parser.add_argument("--overrides-file", required=True, help="Path to overrides JSON")
    parser.add_argument("--video-id", help="Comma-separated list of video IDs to apply")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes only")
    args = parser.parse_args()

    overrides = load_overrides(args.overrides_file)
    filter_ids = _parse_video_ids(args.video_id)
    if filter_ids:
        overrides = [o for o in overrides if o.video_id in filter_ids]

    if not overrides:
        print("⚠️  No overrides to apply")
        return 0

    with PostgresClient() as postgres:
        with postgres.get_cursor() as cursor:
            for override in overrides:
                cursor.execute(
                    "SELECT 1 FROM speakers WHERE id = %s",
                    (override.new_speaker_id,),
                )
                if not cursor.fetchone():
                    raise ValueError(
                        f"Unknown new_speaker_id: {override.new_speaker_id} for {override.video_id}"
                    )

            print("Applying overrides:")
            for override in overrides:
                _apply_override(cursor, override, args.dry_run)

            if not args.dry_run:
                for video_id in {o.video_id for o in overrides}:
                    _update_edge_speakers(cursor, video_id)

    if args.dry_run:
        print("✅ Dry run complete")
    else:
        print("✅ Speaker overrides applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
