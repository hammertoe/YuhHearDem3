from __future__ import annotations

import hashlib
import json

from lib.db.postgres_client import PostgresClient
from lib.order_papers.parser import OrderPaperParsed, parse_order_paper_text


def generate_order_paper_id(sitting_date: str, order_paper_number: str) -> str:
    """Generate a stable order paper ID."""
    unique = f"order_paper:{sitting_date}:{order_paper_number}".encode("utf-8")
    return "op_" + hashlib.md5(unique).hexdigest()[:12]


def generate_order_paper_item_id(order_paper_id: str, sequence: int) -> str:
    return f"{order_paper_id}:{sequence}"


class OrderPaperIngestor:
    def __init__(self, postgres: PostgresClient):
        self.postgres = postgres

    def ingest_order_paper_text(self, raw_text: str) -> str:
        parsed = parse_order_paper_text(raw_text)
        return self.ingest_parsed(parsed, raw_text)

    def ingest_parsed(self, parsed: OrderPaperParsed, raw_text: str) -> str:
        order_paper_id = generate_order_paper_id(
            parsed.sitting_date, parsed.order_paper_number
        )

        self.postgres.execute_update(
            """
            INSERT INTO order_papers (
                id, sitting_date, order_paper_number, session, sitting_number, raw_text, parsed_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (sitting_date, order_paper_number) DO UPDATE SET
                session = EXCLUDED.session,
                sitting_number = EXCLUDED.sitting_number,
                raw_text = EXCLUDED.raw_text,
                parsed_json = EXCLUDED.parsed_json,
                updated_at = NOW()
            """,
            (
                order_paper_id,
                parsed.sitting_date,
                parsed.order_paper_number,
                parsed.session,
                parsed.sitting_number,
                raw_text,
                json.dumps(
                    {
                        "session": parsed.session,
                        "sitting_number": parsed.sitting_number,
                        "sitting_date": parsed.sitting_date,
                        "order_paper_number": parsed.order_paper_number,
                        "items": [item.__dict__ for item in parsed.items],
                    }
                ),
            ),
        )

        for idx, item in enumerate(parsed.items, 1):
            item_id = generate_order_paper_item_id(order_paper_id, idx)
            self.postgres.execute_update(
                """
                INSERT INTO order_paper_items (
                    id, order_paper_id, sequence, item_type, title, mover, action, status_text, linked_bill_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    item_type = EXCLUDED.item_type,
                    title = EXCLUDED.title,
                    mover = EXCLUDED.mover,
                    action = EXCLUDED.action,
                    status_text = EXCLUDED.status_text,
                    linked_bill_id = EXCLUDED.linked_bill_id,
                    updated_at = NOW()
                """,
                (
                    item_id,
                    order_paper_id,
                    idx,
                    item.item_type,
                    item.title,
                    item.mover,
                    item.action,
                    item.status_text,
                    None,
                ),
            )

        return order_paper_id
