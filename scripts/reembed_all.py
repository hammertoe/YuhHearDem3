#!/usr/bin/env python3
"""Re-embed all stored vectors using the local BGE embedding model.

Migrates `paragraphs`, `entities`, `bill_excerpts`, and `kg_nodes` from
Google Gemini 768-dim embeddings to local BAAI/bge-base-en-v1.5 768-dim
embeddings. Both models produce 768-dim vectors so the pgvector column
type is unchanged.

Usage:
    python scripts/reembed_all.py --dry-run            # count rows, no writes
    python scripts/reembed_all.py --batch-size 64      # default 64
    python scripts/reembed_all.py --only paragraphs    # restrict to one table
"""

import argparse
import os
import sys
import time
from collections.abc import Iterator

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.db.pgvector import vector_literal  # noqa: E402
from lib.db.postgres_client import PostgresClient  # noqa: E402


# (table, id_column, text_column, supports_label)
TABLES: list[tuple[str, str, str]] = [
    ("paragraphs", "id", "text"),
    ("entities", "id", "text"),
    ("bill_excerpts", "id", "text"),
    ("kg_nodes", "id", "label"),
]


def _stream_rows(
    pg: PostgresClient,
    table: str,
    id_column: str,
    text_column: str,
    batch_size: int,
    only_missing: bool,
) -> Iterator[tuple[list[str], list[str]]]:
    """Yield (ids, texts) batches from a table."""
    where = "WHERE embedding IS NULL" if only_missing else ""
    offset = 0
    while True:
        sql = (
            f"SELECT {id_column}, {text_column} FROM {table} "
            f"{where} ORDER BY {id_column} LIMIT %s OFFSET %s"
        )
        rows = pg.execute_query(sql, (batch_size, offset))
        if not rows:
            return
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        yield ids, texts
        if len(rows) < batch_size:
            return
        offset += batch_size


def _update_embeddings(
    pg: PostgresClient,
    table: str,
    id_column: str,
    ids: list[str],
    vectors: list[list[float]],
) -> None:
    """Write vectors back into the table using a single UPDATE...FROM(VALUES)."""
    if not ids:
        return
    values_sql = ",".join(["(%s, %s::vector)"] * len(ids))
    params: list = []
    for vid, vec in zip(ids, vectors):
        params.extend([vid, vector_literal(vec)])
    sql = (
        f"UPDATE {table} SET embedding = v.emb, updated_at = NOW() "
        f"FROM (VALUES {values_sql}) AS v(id, emb) "
        f"WHERE {table}.{id_column} = v.id"
    )
    with pg.connection.cursor() as cur:
        cur.execute(sql, params)
        pg.connection.commit()


def reembed_table(
    pg: PostgresClient,
    *,
    table: str,
    id_column: str,
    text_column: str,
    batch_size: int,
    only_missing: bool,
    dry_run: bool,
) -> int:
    from lib.embeddings.bge_local_client import LocalBGEEmbeddingClient

    print(f"\n→ Re-embedding {table}.{text_column}")
    client = LocalBGEEmbeddingClient()

    total = 0
    start = time.time()
    for ids, texts in _stream_rows(pg, table, id_column, text_column, batch_size, only_missing):
        vectors = client.generate_embeddings_batch(texts)
        if dry_run:
            total += len(ids)
            continue
        _update_embeddings(pg, table, id_column, ids, vectors)
        total += len(ids)
        elapsed = time.time() - start
        rate = total / elapsed if elapsed > 0 else 0.0
        print(f"  {table}: {total} rows ({rate:.1f}/s)")
    elapsed = time.time() - start
    print(
        f"✓ {table}: {total} rows in {elapsed:.1f}s" + (" (DRY RUN — no writes)" if dry_run else "")
    )
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-embed all stored vectors with BGE.")
    parser.add_argument(
        "--only",
        choices=[t[0] for t in TABLES],
        help="Re-embed only this table.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Rows per embedding batch.")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only process rows where embedding IS NULL.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Count rows without writing.")
    args = parser.parse_args()

    targets = [t for t in TABLES if not args.only or t[0] == args.only]
    print(
        f"Re-embedding {len(targets)} table(s) "
        f"(batch_size={args.batch_size}, only_missing={args.only_missing}, "
        f"dry_run={args.dry_run})"
    )

    with PostgresClient() as pg:
        for table, id_col, text_col in targets:
            reembed_table(
                pg,
                table=table,
                id_column=id_col,
                text_column=text_col,
                batch_size=args.batch_size,
                only_missing=args.only_missing,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
