#!/usr/bin/env python3
"""Clear all knowledge graph data for fresh extraction."""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.db.postgres_client import PostgresClient


def clear_kg_tables():
    """Clear all KG tables."""
    with PostgresClient() as pg:
        print("Clearing knowledge graph tables...")

        # Clear edges first (due to foreign key constraints)
        result = pg.execute_query("DELETE FROM kg_edges RETURNING 1")
        count = len(result) if result else 0
        print(f"  Deleted {count} edges")

        # Clear aliases
        result = pg.execute_query("DELETE FROM kg_aliases RETURNING 1")
        count = len(result) if result else 0
        print(f"  Deleted {count} aliases")

        # Clear nodes
        result = pg.execute_query("DELETE FROM kg_nodes RETURNING 1")
        count = len(result) if result else 0
        print(f"  Deleted {count} nodes")

        print("✅ Knowledge graph cleared successfully")


def main():
    parser = argparse.ArgumentParser(description="Clear knowledge graph tables")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    if not args.yes:
        confirm = input(
            "This will delete ALL nodes, edges, and aliases from the knowledge graph. "
            "Are you sure? (type 'yes' to confirm): "
        )
        if confirm.lower() != "yes":
            print("Aborted.")
            return

    try:
        clear_kg_tables()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
