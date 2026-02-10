"""Scrape, process, and ingest bills into Postgres and Memgraph."""

from __future__ import annotations

import argparse
import json
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from lib.db.memgraph_client import MemgraphClient
from lib.db.postgres_client import PostgresClient
from lib.embeddings.google_client import GoogleEmbeddingClient
from lib.processors.bill_entity_extractor import BillEntityExtractor
from lib.processors.bill_ingestor import BillIngestor
from lib.scraping.bill_scraper import BillScraper


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest bills as first-class entities")
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Scrape bills from website before ingesting",
    )
    parser.add_argument(
        "--scraped-file",
        default="bills_scraped.json",
        help="Input file with scraped bills (skip scraping)",
    )
    parser.add_argument(
        "--processed-file",
        default="bills_processed.json",
        help="Output file with processed bills",
    )
    parser.add_argument(
        "--max-bills",
        type=int,
        default=None,
        help="Maximum number of bills to ingest",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip generating/storing bill embeddings in Postgres",
    )
    parser.add_argument(
        "--source-url",
        default="https://www.parliament.gov.bb/legislation",
        help="URL to scrape bills from",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Bill Ingestion")
    print("=" * 80)
    print(f"Scrape: {args.scrape}")
    print(f"Source URL: {args.source_url}")
    print(f"Scraped File: {args.scraped_file}")
    print(f"Processed File: {args.processed_file}")
    print(f"Max Bills: {args.max_bills or 'All'}")
    print("=" * 80)

    if args.scrape:
        scraper = BillScraper()
        bills = scraper.scrape_all_bills(max_bills=args.max_bills)
        with open(args.scraped_file, "w", encoding="utf-8") as f:
            json.dump(bills, f, indent=2)
    else:
        with open(args.scraped_file, "r", encoding="utf-8") as f:
            bills = json.load(f)

    if args.max_bills:
        bills = bills[: args.max_bills]

    extractor = BillEntityExtractor()
    bills = extractor.process_bills(bills)

    with open(args.processed_file, "w", encoding="utf-8") as f:
        json.dump(bills, f, indent=2)

    print(f"\n✅ Saved {len(bills)} processed bills to {args.processed_file}")

    embeddings = None if args.skip_embeddings else GoogleEmbeddingClient()
    with PostgresClient() as postgres, MemgraphClient() as memgraph:
        ingestor = BillIngestor(
            postgres=postgres,
            memgraph=memgraph,
            embedding_client=embeddings,
        )
        ingestor.ingest_bills(bills, embed=not args.skip_embeddings)

    print("\n" + "=" * 80)
    print("✅ Bill ingestion complete")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
