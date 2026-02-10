#!/usr/bin/env python3
"""Ingest order paper PDFs using Gemini vision."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Get absolute path to project root
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.db.postgres_client import PostgresClient
from lib.google_client import GeminiClient
from lib.order_papers.pdf_parser import OrderPaperParser
from lib.id_generators import generate_order_paper_id


def ingest_order_paper(pdf_path: str, chamber: str = "house") -> str:
    """Parse an order paper PDF using Gemini vision and save it to database.

    Args:
        pdf_path: Path to PDF file
        chamber: Chamber type (house or senate, default: house)

    Returns:
        Order paper ID
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Parsing order paper: {pdf_path}")

    gemini_client = GeminiClient()
    parser = OrderPaperParser(gemini_client)
    parsed_paper = parser.parse(pdf_path)

    print(f"  Session: {parsed_paper.session_title}")
    print(f"  Date: {parsed_paper.session_date}")
    print(f"  Sitting: {parsed_paper.sitting_number}")
    print(f"  Speakers: {len(parsed_paper.speakers)}")
    print(f"  Agenda Items: {len(parsed_paper.agenda_items)}")

    chamber_code = "h" if chamber == "house" else "s"
    sitting_number = parsed_paper.sitting_number or ""
    order_paper_number = sitting_number

    order_paper_id = generate_order_paper_id(
        chamber_code=chamber_code,
        session_date=parsed_paper.session_date,
        order_paper_number=order_paper_number,
    )

    parsed_json = {
        "session_title": parsed_paper.session_title,
        "sitting_number": parsed_paper.sitting_number,
        "speakers": [
            {"name": s.name, "title": s.title, "role": s.role}
            for s in parsed_paper.speakers
        ],
        "agenda_items": [
            {
                "topic_title": i.topic_title,
                "primary_speaker": i.primary_speaker,
                "description": i.description,
            }
            for i in parsed_paper.agenda_items
        ],
    }

    raw_text = f"Session: {parsed_paper.session_title}\nDate: {parsed_paper.session_date}\nSitting: {parsed_paper.sitting_number}"

    with PostgresClient() as postgres:
        postgres.execute_update(
            """
            INSERT INTO order_papers (
                id, sitting_date, order_paper_number, session,
                sitting_number, raw_text, parsed_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                session = EXCLUDED.session,
                sitting_number = EXCLUDED.sitting_number,
                raw_text = EXCLUDED.raw_text,
                parsed_json = EXCLUDED.parsed_json,
                updated_at = NOW()
            """,
            (
                order_paper_id,
                parsed_paper.session_date,
                order_paper_number,
                parsed_paper.session_title,
                parsed_paper.sitting_number,
                raw_text,
                json.dumps(parsed_json),
            ),
        )

        for idx, item in enumerate(parsed_paper.agenda_items, 1):
            item_id = f"{order_paper_id}_{idx:03d}"
            postgres.execute_update(
                """
                INSERT INTO order_paper_items (
                    id, order_paper_id, sequence, item_type,
                    title, mover, action, status_text
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    order_paper_id = EXCLUDED.order_paper_id,
                    sequence = EXCLUDED.sequence,
                    item_type = EXCLUDED.item_type,
                    title = EXCLUDED.title,
                    mover = EXCLUDED.mover,
                    action = EXCLUDED.action,
                    status_text = EXCLUDED.status_text,
                    updated_at = NOW()
                """,
                (
                    item_id,
                    order_paper_id,
                    idx,
                    "agenda",
                    item.topic_title,
                    item.primary_speaker,
                    None,
                    None,
                ),
            )

        print(f"✅ Saved order paper ID: {order_paper_id}")
        print(f"   {len(parsed_paper.agenda_items)} agenda items")
        return order_paper_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest order paper PDF using Gemini vision"
    )
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument(
        "--chamber",
        choices=["house", "senate"],
        default="house",
        help="Chamber type (default: house)",
    )
    args = parser.parse_args()

    try:
        order_paper_id = ingest_order_paper(args.pdf_path, args.chamber)
        print(f"\nOrder Paper ID: {order_paper_id}")
        return 0
    except Exception as e:
        print(f"❌ Failed to ingest order paper: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
