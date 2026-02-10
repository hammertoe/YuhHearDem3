from __future__ import annotations

from unittest.mock import Mock

from lib.order_papers.ingestor import OrderPaperIngestor


def test_ingest_order_paper_calls_db_with_items() -> None:
    postgres = Mock()
    ingestor = OrderPaperIngestor(postgres)

    raw_text = """
**Date:** Tuesday, 6th January, 2026
**Order Paper Number:** No. 125

**3. Bills on the Order Paper:**
*   **Barbados Citizenship Bill, 2025**
    *   Mover: Hon. W. A. Abrahams
    *   Action: Second Reading
""".strip()

    order_paper_id = ingestor.ingest_order_paper_text(raw_text)

    assert isinstance(order_paper_id, str)
    assert order_paper_id
    # One insert into order_papers + one into order_paper_items.
    assert postgres.execute_update.call_count >= 2
