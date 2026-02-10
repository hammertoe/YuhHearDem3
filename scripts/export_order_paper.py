# ruff: noqa: E402
# ruff: noqa: E402

/* ruff: noqa: E402 */#!/usr/bin/env python3
"""Export order paper from database to text file for use by transcribe.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.db.postgres_client import PostgresClient


def export_order_paper(order_paper_id: str, output_path: str) -> None:
    """Export order paper from database to text file.

    Args:
        order_paper_id: Order paper ID
        output_path: Path to output text file
    """
    with PostgresClient() as postgres:
        result = postgres.execute_query(
            """
            SELECT id, session_title, sitting_date, sitting_number,
                   speakers, agenda_items
            FROM order_papers
            WHERE id = %s
            """,
            (order_paper_id,),
        )

        if not result:
            raise ValueError(f"Order paper not found: {order_paper_id}")

        paper = result[0]
        session_title = paper[1] or ""
        sitting_date = str(paper[2]) or ""
        sitting_number = paper[3] or ""
        speakers_json = paper[4] or "[]"
        agenda_json = paper[5] or "[]"

        speakers = json.loads(speakers_json)
        agenda_items = json.loads(agenda_json)

    output = []

    output.append("=" * 80)
    output.append(session_title.upper())
    output.append("=" * 80)
    output.append(f"Sitting: {sitting_number}")
    output.append(f"Date: {sitting_date}")
    output.append()

    if speakers:
        output.append("SPEAKERS")
        output.append("-" * 40)
        for speaker in speakers:
            name = speaker.get("name", "")
            title = speaker.get("title", "")
            role = speaker.get("role", "")

            line = f"- {name}"
            if title:
                line += f" ({title})"
            if role:
                line += f" - {role}"
            output.append(line)
        output.append()

    if agenda_items:
        output.append("AGENDA ITEMS")
        output.append("-" * 40)
        for idx, item in enumerate(agenda_items, 1):
            topic = item.get("topic_title", "")
            mover = item.get("primary_speaker", "")
            description = item.get("description", "")

            output.append(f"{idx}. {topic}")
            if mover:
                output.append(f"   Mover: {mover}")
            if description:
                output.append(f"   Description: {description}")
            output.append()

    text = "\n".join(output)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Exported order paper to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export order paper from database to text file"
    )
    parser.add_argument("order_paper_id", help="Order paper ID")
    parser.add_argument("output_path", help="Output text file path")
    args = parser.parse_args()

    try:
        export_order_paper(args.order_paper_id, args.output_path)
        return 0
    except Exception as e:
        print(f"❌ Failed to export order paper: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
