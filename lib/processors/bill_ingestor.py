"""Bill ingestion into Postgres (three-tier storage)."""

from __future__ import annotations

from typing import Any

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

    def ingest_bills(self, bills: list[dict[str, Any]], *, embed: bool = True) -> None:
        existing_bill_ids: set[str] = set()

        for bill in bills:
            bill_number = bill.get("bill_number", "")
            bill_id = generate_bill_id(bill_number, existing_bill_ids)
            existing_bill_ids.add(bill_id)
            entity_id = generate_entity_id(bill.get("title", bill_number), "BILL")
            self.ingest_bill(bill, bill_id=bill_id, entity_id=entity_id, embed=embed)

    def ingest_bill(
        self, bill: dict[str, Any], bill_id: str, entity_id: str, *, embed: bool = True
    ) -> None:
        self._upsert_bill_entity_postgres(bill, entity_id)
        self._upsert_bill_postgres(bill, bill_id, entity_id)
        if embed:
            self._store_bill_embedding_postgres(bill, entity_id)

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
