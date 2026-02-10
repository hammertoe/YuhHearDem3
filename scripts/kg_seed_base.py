"""Script to seed base KG from existing database tables."""

from __future__ import annotations

import argparse
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.db.postgres_client import PostgresClient
from lib.embeddings.google_client import GoogleEmbeddingClient
from lib.knowledge_graph.base_kg_seeder import BaseKGSeeder


def main():
    parser = argparse.ArgumentParser(
        description="Seed base KG from speakers, order papers, and bills"
    )
    parser.add_argument(
        "--skip-speakers", action="store_true", help="Skip seeding speakers"
    )
    parser.add_argument(
        "--skip-order-papers",
        action="store_true",
        help="Skip seeding order paper items",
    )
    parser.add_argument("--skip-bills", action="store_true", help="Skip seeding bills")
    args = parser.parse_args()

    print("=" * 60)
    print("Base KG Seeding")
    print("=" * 60)

    try:
        with PostgresClient() as pg_client:
            embedding_client = GoogleEmbeddingClient()
            seeder = BaseKGSeeder(pg_client, embedding_client)

            total_counts = {
                "speakers": 0,
                "order_paper_items": 0,
                "bills": 0,
            }

            if not args.skip_speakers:
                print("\nSeeding speakers...")
                count = seeder._seed_speakers()
                total_counts["speakers"] = count
                print(f"✅ Seeded {count} speaker nodes")

            if not args.skip_order_papers:
                print("\nSeeding order paper items...")
                count = seeder._seed_order_paper_items()
                total_counts["order_paper_items"] = count
                print(f"✅ Seeded {count} order paper item nodes")

            if not args.skip_bills:
                print("\nSeeding bills...")
                count = seeder._seed_bills()
                total_counts["bills"] = count
                print(f"✅ Seeded {count} bill nodes")

            print("\n" + "=" * 60)
            print("Summary")
            print("=" * 60)
            total = sum(total_counts.values())
            print(f"Total nodes seeded: {total}")
            for source, count in total_counts.items():
                print(f"  {source}: {count}")
            print("=" * 60)

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
