"""Repair transcript-derived rows after timestamp normalization."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.db.postgres_client import PostgresClient  # noqa: E402
from lib.transcripts.db_repair_helpers import (  # noqa: E402
    build_temp_id,
    build_temp_table_ddl,
    filter_paragraph_updates_by_existing,
    filter_sentence_mapping_by_existing,
    should_use_existing_order,
)
from lib.transcripts.timestamp_fix import (  # noqa: E402
    build_paragraph_id_mapping,
    build_sentence_id_mapping,
    build_sentence_id_mapping_from_existing_ids,
    extract_video_id_from_filename,
    parse_duration_to_seconds,
)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _safe_table_suffix(video_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", video_id)
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{safe}_{stamp}"


def _backup_tables(cursor, video_id: str) -> None:
    suffix = _safe_table_suffix(video_id)
    cursor.execute(
        f"CREATE TABLE sentences_backup_{suffix} AS "
        "SELECT * FROM sentences WHERE youtube_video_id = %s",
        (video_id,),
    )
    cursor.execute(
        f"CREATE TABLE paragraphs_backup_{suffix} AS "
        "SELECT * FROM paragraphs WHERE youtube_video_id = %s",
        (video_id,),
    )
    cursor.execute(
        f"CREATE TABLE sentence_entities_backup_{suffix} AS "
        "SELECT se.* FROM sentence_entities se "
        "JOIN sentences s ON se.sentence_id = s.id "
        "WHERE s.youtube_video_id = %s",
        (video_id,),
    )
    cursor.execute(
        f"CREATE TABLE paragraph_entities_backup_{suffix} AS "
        "SELECT pe.* FROM paragraph_entities pe "
        "JOIN paragraphs p ON pe.paragraph_id = p.id "
        "WHERE p.youtube_video_id = %s",
        (video_id,),
    )
    cursor.execute(
        f"CREATE TABLE kg_edges_backup_{suffix} AS "
        "SELECT * FROM kg_edges WHERE youtube_video_id = %s",
        (video_id,),
    )


def _parse_video_ids(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {v.strip() for v in raw.split(",") if v.strip()}


def _repair_single_video(
    *,
    postgres: PostgresClient,
    video_id: str,
    transcript_file: Path,
    dry_run: bool,
    no_backup: bool,
) -> None:
    data = _load_json(transcript_file)
    transcripts = data.get("transcripts", [])
    duration = data.get("video_metadata", {}).get("duration", "")
    duration_seconds = parse_duration_to_seconds(duration)

    sentence_map = build_sentence_id_mapping(
        transcripts=transcripts,
        youtube_video_id=video_id,
        duration_seconds=duration_seconds,
    )
    paragraph_updates = build_paragraph_id_mapping(
        transcripts=transcripts,
        youtube_video_id=video_id,
        duration_seconds=duration_seconds,
    )

    with postgres.get_cursor() as cursor:
        cursor.execute(
            "SELECT id FROM sentences WHERE youtube_video_id = %s ORDER BY seconds_since_start ASC, id ASC",
            (video_id,),
        )
        sentence_id_rows = cursor.fetchall()
        sentence_ids = {row[0] for row in sentence_id_rows}
        sentence_ids_ordered = [row[0] for row in sentence_id_rows]
        cursor.execute(
            "SELECT id FROM paragraphs WHERE youtube_video_id = %s",
            (video_id,),
        )
        paragraph_ids = {row[0] for row in cursor.fetchall()}

        if len(sentence_ids) == len(transcripts) and should_use_existing_order(
            sentence_ids, set(sentence_map.old_to_new.keys())
        ):
            sentence_map = build_sentence_id_mapping_from_existing_ids(
                transcripts=transcripts,
                youtube_video_id=video_id,
                duration_seconds=duration_seconds,
                existing_ids=sentence_ids_ordered,
            )
        sentence_map = filter_sentence_mapping_by_existing(sentence_map, sentence_ids)
        paragraph_updates = filter_paragraph_updates_by_existing(paragraph_updates, paragraph_ids)

        existing_new_ids = {
            new_id for new_id in sentence_map.old_to_new.values() if new_id in sentence_ids
        }

        if dry_run:
            print(f"✅ Dry run: {video_id}")
            print(f"   sentences mapped: {len(sentence_map.old_to_new)}")
            print(f"   paragraphs mapped: {len(paragraph_updates)}")
            return

        if not no_backup:
            _backup_tables(cursor, video_id)

        for stmt in build_temp_table_ddl(
            "sentence_id_map",
            "old_id text primary key, new_id text, tmp_id text, conflict boolean, "
            "new_seconds int, new_timestamp text",
        ):
            cursor.execute(stmt)  # type: ignore[arg-type]
        sentence_rows = [
            (
                old_id,
                new_id,
                build_temp_id(old_id),
                new_id in existing_new_ids,
                sentence_map.new_seconds_by_old_id[old_id],
                sentence_map.new_timestamp_by_old_id[old_id],
            )
            for old_id, new_id in sentence_map.old_to_new.items()
        ]
        cursor.executemany(
            "INSERT INTO sentence_id_map (old_id, new_id, tmp_id, conflict, new_seconds, new_timestamp) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            sentence_rows,
        )

        for stmt in build_temp_table_ddl(
            "paragraph_id_map",
            "old_id text primary key, new_id text, new_start_seconds int, "
            "new_end_seconds int, new_start_timestamp text, new_end_timestamp text",
        ):
            cursor.execute(stmt)  # type: ignore[arg-type]
        paragraph_rows = [
            (
                row["old_id"],
                row["new_id"],
                row["new_start_seconds"],
                row["new_end_seconds"],
                row["new_start_timestamp"],
                row["new_end_timestamp"],
            )
            for row in paragraph_updates
        ]
        cursor.executemany(
            "INSERT INTO paragraph_id_map ("
            "old_id, new_id, new_start_seconds, new_end_seconds, "
            "new_start_timestamp, new_end_timestamp) VALUES (%s, %s, %s, %s, %s, %s)",
            paragraph_rows,
        )

        cursor.execute(
            "UPDATE sentence_entities se SET sentence_id = m.tmp_id "
            "FROM sentence_id_map m WHERE se.sentence_id = m.old_id"
        )
        cursor.execute(
            "UPDATE sentences s SET id = m.tmp_id FROM sentence_id_map m WHERE s.id = m.old_id"
        )
        cursor.execute(
            "UPDATE sentences s SET "
            "id = m.new_id, seconds_since_start = m.new_seconds, "
            "timestamp_str = m.new_timestamp "
            "FROM sentence_id_map m WHERE s.id = m.tmp_id AND m.conflict = false"
        )
        cursor.execute(
            "UPDATE sentence_entities se SET sentence_id = m.new_id "
            "FROM sentence_id_map m WHERE se.sentence_id = m.tmp_id"
        )
        cursor.execute(
            "DELETE FROM sentences s USING sentence_id_map m "
            "WHERE s.id = m.tmp_id AND m.conflict = true"
        )

        cursor.execute(
            "INSERT INTO paragraphs ("
            "id, youtube_video_id, start_seconds, end_seconds, text, speaker_id, "
            "voice_id, start_timestamp, end_timestamp, embedding, video_date, video_title, sentence_count"
            ") "
            "SELECT m.new_id, p.youtube_video_id, m.new_start_seconds, m.new_end_seconds, "
            "p.text, p.speaker_id, p.voice_id, m.new_start_timestamp, m.new_end_timestamp, "
            "p.embedding, p.video_date, p.video_title, p.sentence_count "
            "FROM paragraphs p JOIN paragraph_id_map m ON p.id = m.old_id "
            "ON CONFLICT (id) DO UPDATE SET "
            "start_seconds = EXCLUDED.start_seconds, "
            "end_seconds = EXCLUDED.end_seconds, "
            "start_timestamp = EXCLUDED.start_timestamp, "
            "end_timestamp = EXCLUDED.end_timestamp"
        )
        cursor.execute(
            "UPDATE sentences s SET paragraph_id = m.new_id "
            "FROM paragraph_id_map m WHERE s.paragraph_id = m.old_id"
        )
        cursor.execute(
            "UPDATE paragraph_entities pe SET paragraph_id = m.new_id "
            "FROM paragraph_id_map m WHERE pe.paragraph_id = m.old_id"
        )
        cursor.execute(
            "DELETE FROM paragraphs p USING paragraph_id_map m "
            "WHERE p.id = m.old_id AND m.old_id <> m.new_id"
        )

        cursor.execute(
            "UPDATE kg_edges e SET utterance_ids = sub.new_ids "
            "FROM ("
            "  SELECT e.id AS edge_id, "
            "         array_agg(COALESCE(m.new_id, u.utt) ORDER BY u.ord) AS new_ids "
            "  FROM kg_edges e "
            "  CROSS JOIN LATERAL unnest(e.utterance_ids) WITH ORDINALITY AS u(utt, ord) "
            "  LEFT JOIN sentence_id_map m ON m.old_id = u.utt "
            "  WHERE e.youtube_video_id = %s "
            "  GROUP BY e.id"
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

    print(f"✅ Repaired: {video_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair transcript timestamps in DB")
    parser.add_argument("--video-id", help="YouTube video id (comma-separated for multiple)")
    parser.add_argument(
        "--transcript-file",
        help="Original transcript JSON file",
    )
    parser.add_argument(
        "--input-dir",
        help="Directory with transcription_output_*.json files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report changes without writing",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup tables",
    )
    args = parser.parse_args()

    video_ids = _parse_video_ids(args.video_id)

    files: list[tuple[str, Path]] = []
    if args.input_dir:
        for path in sorted(Path(args.input_dir).glob("transcription_output_*.json")):
            video_id = extract_video_id_from_filename(path.name)
            if not video_id:
                continue
            if video_ids and video_id not in video_ids:
                continue
            files.append((video_id, path))
    elif args.transcript_file and args.video_id:
        files.append((args.video_id, Path(args.transcript_file)))
    else:
        raise ValueError("Provide --input-dir or --video-id with --transcript-file")

    if not files:
        print("⚠️  No matching transcript files found")
        return 0

    with PostgresClient() as postgres:
        for video_id, path in files:
            _repair_single_video(
                postgres=postgres,
                video_id=video_id,
                transcript_file=path,
                dry_run=args.dry_run,
                no_backup=args.no_backup,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
