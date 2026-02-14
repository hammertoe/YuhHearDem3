"""Bill ingestion into Postgres (three-tier storage)."""

from __future__ import annotations

from typing import Any

from lib.bills.excerpt_chunker import chunk_bill_text, generate_chunk_id
from lib.db.postgres_client import PostgresClient
from lib.embeddings.google_client import GoogleEmbeddingClient
from lib.id_generators import generate_bill_id, generate_entity_id


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


class BillIngestor:
    """Ingest bills as first-class entities with embeddings."""

    def __init__(
        self,
        postgres: PostgresClient | None = None,
        embedding_client: GoogleEmbeddingClient | None = None,
    ):
        self.postgres = postgres or PostgresClient()
        self.embedding_client = embedding_client or GoogleEmbeddingClient()

    def ingest_bills(
        self,
        bills: list[dict[str, Any]],
        *,
        embed: bool = True,
        ingest_excerpts: bool = True,
    ) -> int:
        """Ingest multiple bills with entity, metadata, and optionally excerpts.

        Args:
            bills: List of bill data dicts
            embed: Whether to generate entity embeddings
            ingest_excerpts: Whether to chunk and embed bill text

        Returns:
            Total number of excerpts created across all bills
        """
        existing_bill_ids: set[str] = set()
        total_excerpts = 0

        for bill in bills:
            bill_number = bill.get("bill_number", "")
            bill_id = generate_bill_id(bill_number, existing_bill_ids)
            existing_bill_ids.add(bill_id)
            entity_id = generate_entity_id(bill.get("title", bill_number), "BILL")
            excerpt_count = self.ingest_bill(
                bill,
                bill_id=bill_id,
                entity_id=entity_id,
                embed=embed,
                ingest_excerpts=ingest_excerpts,
            )
            total_excerpts += excerpt_count

        return total_excerpts

    def ingest_bill(
        self,
        bill: dict[str, Any],
        bill_id: str,
        entity_id: str,
        *,
        embed: bool = True,
        ingest_excerpts: bool = True,
    ) -> int:
        """Ingest a bill with entity, metadata, and optionally excerpt chunks.

        Args:
            bill: Bill data dict
            bill_id: Generated bill ID
            entity_id: Generated entity ID
            embed: Whether to generate entity embedding
            ingest_excerpts: Whether to chunk and embed bill text

        Returns:
            Number of excerpts created
        """
        self._upsert_bill_entity_postgres(bill, entity_id)
        self._upsert_bill_postgres(bill, bill_id, entity_id)

        excerpt_count = 0

        if embed:
            self._store_bill_embedding_postgres(bill, entity_id)

        if ingest_excerpts:
            excerpt_count = self.ingest_bill_excerpts(bill, bill_id)

        return excerpt_count

    def _upsert_bill_entity_postgres(self, bill: dict[str, Any], entity_id: str) -> None:
        """Upsert a row in entities representing the bill."""
        self.postgres.execute_update(
            """
            INSERT INTO entities (
                id, text, type, bill_number, bill_status, category
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                text = EXCLUDED.text,
                bill_number = EXCLUDED.bill_number,
                bill_status = EXCLUDED.bill_status,
                category = EXCLUDED.category,
                updated_at = NOW()
            """,
            (
                entity_id,
                bill.get("title", ""),
                "BILL",
                bill.get("bill_number", ""),
                bill.get("status", ""),
                bill.get("category", ""),
            ),
        )

    def _upsert_bill_postgres(self, bill: dict[str, Any], bill_id: str, entity_id: str) -> None:
        """Insert/update the bill record."""
        keywords = bill.get("keywords")
        if not isinstance(keywords, list):
            keywords = []

        self.postgres.execute_update(
            """
            INSERT INTO bills (
                id, bill_number, title, description,
                bill_type, status, introduced_date, passed_date,
                source_url, source_text, entity_id,
                category, keywords
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (bill_number) DO UPDATE SET
                title = EXCLUDED.title,
                description = EXCLUDED.description,
                status = EXCLUDED.status,
                category = EXCLUDED.category,
                keywords = EXCLUDED.keywords,
                updated_at = NOW()
            """,
            (
                bill_id,
                bill.get("bill_number", ""),
                bill.get("title", ""),
                bill.get("description", ""),
                bill.get("bill_type", ""),
                bill.get("status", "Unknown"),
                bill.get("introduced_date"),
                bill.get("passed_date"),
                bill.get("source_url", ""),
                bill.get("source_text", ""),
                entity_id,
                bill.get("category", ""),
                keywords,
            ),
        )

    def _store_bill_embedding_postgres(self, bill: dict[str, Any], entity_id: str) -> None:
        """Generate and store bill embedding in entities."""
        text = (
            (bill.get("title", "") or "").strip()
            + " "
            + (bill.get("description", "") or "").strip()
        )
        text = text.strip()
        if not text:
            return

        try:
            embedding = self.embedding_client.generate_embedding(text)
        except Exception as e:
            print(f"⚠️ Error generating embedding for bill {bill.get('bill_number', '')}: {e}")
            return

        self.postgres.execute_update(
            """
            UPDATE entities
            SET embedding = (%s)::vector, updated_at = NOW()
            WHERE id = %s
            """,
            (_vector_literal(embedding), entity_id),
        )

    def ingest_bill_excerpts(
        self,
        bill: dict[str, Any],
        bill_id: str,
        chunk_size: int = 900,
        overlap: int = 150,
    ) -> int:
        """Generate and store bill excerpt chunks with embeddings.

        Args:
            bill: Bill data dict with source_text and/or description
            bill_id: The bill ID
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks

        Returns:
            Number of excerpts created
        """
        source_text = bill.get("source_text", "")
        description = bill.get("description")
        title = bill.get("title")

        chunks = chunk_bill_text(
            bill_id=bill_id,
            source_text=source_text,
            description=description,
            title=title,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        if not chunks:
            return 0

        chunk_texts = [c.text for c in chunks]

        try:
            embeddings = self.embedding_client.generate_embeddings_batch(chunk_texts)
        except Exception as e:
            print(
                f"⚠️ Error generating embeddings for bill excerpts {bill.get('bill_number', '')}: {e}"
            )
            return 0

        source_url = bill.get("source_url", "")

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = generate_chunk_id(bill_id, chunk.chunk_index)
            self.postgres.execute_update(
                """
                INSERT INTO bill_excerpts (
                    id, bill_id, chunk_index, text, char_start, char_end,
                    embedding, source_url
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (bill_id, chunk_index) DO UPDATE SET
                    text = EXCLUDED.text,
                    char_start = EXCLUDED.char_start,
                    char_end = EXCLUDED.char_end,
                    embedding = EXCLUDED.embedding,
                    source_url = EXCLUDED.source_url,
                    updated_at = NOW()
                """,
                (
                    chunk_id,
                    bill_id,
                    chunk.chunk_index,
                    chunk.text,
                    chunk.char_start,
                    chunk.char_end,
                    _vector_literal(embedding),
                    source_url,
                ),
            )

        return len(chunks)

    def upsert_bill_with_excerpts(
        self,
        bill: dict[str, Any],
        bill_id: str,
        entity_id: str,
        *,
        embed: bool = True,
        ingest_excerpts: bool = True,
    ) -> int:
        """Ingest a bill with entity, metadata, and excerpt chunks.

        Args:
            bill: Bill data dict
            bill_id: Generated bill ID
            entity_id: Generated entity ID
            embed: Whether to generate entity embedding
            ingest_excerpts: Whether to chunk and embed bill text

        Returns:
            Number of excerpts created
        """
        self._upsert_bill_entity_postgres(bill, entity_id)
        self._upsert_bill_postgres(bill, bill_id, entity_id)

        excerpt_count = 0

        if embed:
            self._store_bill_embedding_postgres(bill, entity_id)

        if ingest_excerpts:
            excerpt_count = self.ingest_bill_excerpts(bill, bill_id)

        return excerpt_count
