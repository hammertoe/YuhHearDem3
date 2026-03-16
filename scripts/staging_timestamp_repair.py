"""Repair collapsed sentence timestamps using transcript JSON files.

This script matches rows by (text, speaker_id, voice_id) and applies timestamp
values from transcript files in order of appearance per key.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from lib.db.postgres_client import PostgresClient
from lib.id_generators import format_seconds_to_timestamp, parse_timestamp_to_seconds


def repair_video(video_id: str, transcript_path: Path) -> int:
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    transcripts = data.get("transcripts", [])

    transcript_groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for entry in transcripts:
        key = (
            str(entry.get("text") or "").strip(),
            str(entry.get("speaker_id") or "").strip(),
            int(entry.get("voice_id") or 0),
        )
        seconds = parse_timestamp_to_seconds(str(entry.get("start") or "0:00:00"))
        transcript_groups[key].append(seconds)

    updates: list[tuple[int, str, str]] = []

    with PostgresClient() as postgres:
        with postgres.get_cursor() as cursor:
            cursor.execute(
                "SELECT id, text, speaker_id, voice_id, seconds_since_start, timestamp_str "
                "FROM sentences WHERE youtube_video_id = %s "
                "ORDER BY seconds_since_start, id",
                (video_id,),
            )
            rows = cursor.fetchall()

            db_groups: dict[tuple[str, str, int], list[tuple[str, int, str]]] = defaultdict(list)
            for sentence_id, text, speaker_id, voice_id, seconds, timestamp_str in rows:
                key = (
                    str(text or "").strip(),
                    str(speaker_id or "").strip(),
                    int(voice_id or 0),
                )
                db_groups[key].append((sentence_id, int(seconds or 0), str(timestamp_str or "")))

            for key, db_list in db_groups.items():
                transcript_list = transcript_groups.get(key, [])
                count = min(len(db_list), len(transcript_list))
                for i in range(count):
                    sentence_id, old_seconds, old_timestamp = db_list[i]
                    new_seconds = transcript_list[i]
                    new_timestamp = format_seconds_to_timestamp(new_seconds)
                    if old_seconds != new_seconds or old_timestamp != new_timestamp:
                        updates.append((new_seconds, new_timestamp, sentence_id))

            if updates:
                cursor.executemany(
                    "UPDATE sentences SET seconds_since_start = %s, timestamp_str = %s WHERE id = %s",
                    updates,
                )

            cursor.execute(
                "UPDATE paragraphs p SET "
                "start_seconds = sub.min_sec, end_seconds = sub.max_sec, "
                "start_timestamp = sub.min_ts, end_timestamp = sub.max_ts "
                "FROM ("
                "  SELECT paragraph_id, "
                "         MIN(seconds_since_start) AS min_sec, "
                "         MAX(seconds_since_start) AS max_sec, "
                "         MIN(timestamp_str) AS min_ts, "
                "         MAX(timestamp_str) AS max_ts "
                "  FROM sentences WHERE youtube_video_id = %s GROUP BY paragraph_id"
                ") sub WHERE p.id = sub.paragraph_id",
                (video_id,),
            )

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

            cursor.execute(
                "UPDATE kg_edges e SET "
                "earliest_seconds = sub.min_seconds, "
                "earliest_timestamp_str = sub.min_timestamp "
                "FROM ("
                "  SELECT e.id AS edge_id, "
                "         MIN(s.seconds_since_start) AS min_seconds, "
                "         MIN(s.timestamp_str) AS min_timestamp "
                "  FROM kg_edges e "
                "  CROSS JOIN LATERAL unnest(e.utterance_ids) AS u(utt) "
                "  JOIN sentences s ON s.id = u.utt "
                "  WHERE e.youtube_video_id = %s "
                "  GROUP BY e.id"
                ") sub "
                "WHERE e.id = sub.edge_id",
                (video_id,),
            )

    return len(updates)


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair collapsed timestamps from transcript JSON")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--transcript-file", required=True)
    args = parser.parse_args()

    updated = repair_video(args.video_id, Path(args.transcript_file))
    print(f"✅ {args.video_id}: updated_sentences={updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
