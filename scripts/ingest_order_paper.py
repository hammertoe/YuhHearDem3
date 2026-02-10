from __future__ import annotations

import argparse
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from lib.db.postgres_client import PostgresClient
from lib.order_papers.ingestor import OrderPaperIngestor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest an order paper into PostgreSQL"
    )
    parser.add_argument(
        "--file", required=True, help="Path to an order paper text file"
    )
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    with PostgresClient() as postgres:
        ingestor = OrderPaperIngestor(postgres)
        order_paper_id = ingestor.ingest_order_paper_text(raw_text)

    print(f"✅ Ingested order paper: {order_paper_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
