"""Base KG seeder for seeding canonical KG from existing data."""

from __future__ import annotations

from typing import Any

from lib.db.postgres_client import PostgresClient
from lib.embeddings.google_client import GoogleEmbeddingClient
from lib.id_generators import generate_kg_node_id, normalize_label
from lib.db.pgvector import vector_literal


class BaseKGSeeder:
    """Seed base KG from speakers, order papers, and bills."""

    NODE_TYPE_PERSON = "foaf:Person"
    NODE_TYPE_LEGISLATION = "schema:Legislation"

    SOURCE_SPEAKER = "speaker_seed"
    SOURCE_ORDER_PAPER = "order_paper"
    SOURCE_BILL = "bill_seed"

    def __init__(
        self,
        postgres_client: PostgresClient,
        embedding_client: GoogleEmbeddingClient,
    ):
        self.postgres = postgres_client
        self.embedding = embedding_client

    def seed_all(self) -> dict[str, int]:
        """Seed all base KG sources and return counts."""
        counts = {
            "speakers": self._seed_speakers(),
            "order_paper_items": self._seed_order_paper_items(),
            "bills": self._seed_bills(),
        }
        return counts

    def _seed_speakers(self) -> int:
        """Seed speakers from speakers table as foaf:Person nodes."""
        query = """
            SELECT id, normalized_name, full_name, title, position, constituency, party
            FROM speakers
        """
        rows = self.postgres.execute_query(query)

        node_data = []
        alias_data = []

        for row in rows:
            (
                speaker_id,
                normalized_name,
                full_name,
                title,
                position,
                constituency,
                party,
            ) = row

            node_id = f"speaker_{speaker_id}"
            label = full_name or normalized_name

            aliases = []
            if full_name:
                aliases.append(normalize_label(full_name))
            if normalized_name:
                aliases.append(normalize_label(normalized_name))
            if title:
                aliases.append(normalize_label(title))

            node_data.append(
                (
                    node_id,
                    label,
                    self.NODE_TYPE_PERSON,
                    aliases,
                )
            )

            for alias in aliases:
                if alias:
                    alias_data.append(
                        (
                            normalize_label(alias),
                            alias,
                            node_id,
                            self.NODE_TYPE_PERSON,
                            self.SOURCE_SPEAKER,
                            None,
                        )
                    )

        if node_data:
            node_query = """
                INSERT INTO kg_nodes (id, label, type, aliases)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET label = EXCLUDED.label,
                    aliases = EXCLUDED.aliases,
                    updated_at = NOW()
            """
            self.postgres.execute_batch(node_query, node_data)

        if alias_data:
            alias_query = """
                INSERT INTO kg_aliases (alias_norm, alias_raw, node_id, type, source, confidence)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (alias_norm) DO NOTHING
            """
            self.postgres.execute_batch(alias_query, alias_data)

        self._generate_embeddings_for_nodes(node_id for node_id, *_ in node_data)

        return len(node_data)

    def _seed_order_paper_items(self) -> int:
        """Seed order paper items (bills/acts) as schema:Legislation nodes."""
        query = """
            SELECT opi.id, opi.title, opi.item_type, opi.linked_bill_id
            FROM order_paper_items opi
            WHERE UPPER(opi.item_type) IN ('BILL', 'ACT')
        """
        rows = self.postgres.execute_query(query)

        node_data = []
        alias_data = []

        for row in rows:
            item_id, title, item_type, linked_bill_id = row

            if linked_bill_id:
                continue

            node_id = generate_kg_node_id(self.NODE_TYPE_LEGISLATION, title)
            label = title
            aliases = [normalize_label(title)]

            node_data.append(
                (
                    node_id,
                    label,
                    self.NODE_TYPE_LEGISLATION,
                    aliases,
                )
            )

            for alias in aliases:
                if alias:
                    alias_data.append(
                        (
                            alias,
                            alias,
                            node_id,
                            self.NODE_TYPE_LEGISLATION,
                            self.SOURCE_ORDER_PAPER,
                            None,
                        )
                    )

        if node_data:
            node_query = """
                INSERT INTO kg_nodes (id, label, type, aliases)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET label = EXCLUDED.label,
                    aliases = EXCLUDED.aliases,
                    updated_at = NOW()
            """
            self.postgres.execute_batch(node_query, node_data)

        if alias_data:
            alias_query = """
                INSERT INTO kg_aliases (alias_norm, alias_raw, node_id, type, source, confidence)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (alias_norm) DO NOTHING
            """
            self.postgres.execute_batch(alias_query, alias_data)

        self._generate_embeddings_for_nodes(node_id for node_id, *_ in node_data)

        return len(node_data)

    def _seed_bills(self) -> int:
        """Seed bills from bills table as schema:Legislation nodes."""
        query = """
            SELECT id, bill_number, title, description
            FROM bills
            WHERE title IS NOT NULL AND title != ''
        """
        rows = self.postgres.execute_query(query)

        node_data = []
        alias_data = []

        for row in rows:
            bill_id, bill_number, title, description = row

            node_id = generate_kg_node_id(self.NODE_TYPE_LEGISLATION, title)
            label = title
            aliases = [normalize_label(title)]

            if bill_number:
                aliases.append(normalize_label(bill_number))

            node_data.append(
                (
                    node_id,
                    label,
                    self.NODE_TYPE_LEGISLATION,
                    aliases,
                )
            )

            for alias in aliases:
                if alias:
                    alias_data.append(
                        (
                            alias,
                            alias,
                            node_id,
                            self.NODE_TYPE_LEGISLATION,
                            self.SOURCE_BILL,
                            None,
                        )
                    )

        if node_data:
            node_query = """
                INSERT INTO kg_nodes (id, label, type, aliases)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET label = EXCLUDED.label,
                    aliases = EXCLUDED.aliases,
                    updated_at = NOW()
            """
            self.postgres.execute_batch(node_query, node_data)

        if alias_data:
            alias_query = """
                INSERT INTO kg_aliases (alias_norm, alias_raw, node_id, type, source, confidence)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (alias_norm) DO NOTHING
            """
            self.postgres.execute_batch(alias_query, alias_data)

        self._generate_embeddings_for_nodes(node_id for node_id, *_ in node_data)

        return len(node_data)

    def _generate_embeddings_for_nodes(self, node_ids: Any) -> None:
        """Generate embeddings for nodes that don't have them."""
        node_id_list = list(node_ids)

        if not node_id_list:
            return

        check_query = (
            "SELECT id, label FROM kg_nodes WHERE id = ANY(%s) AND embedding IS NULL"
        )
        rows = self.postgres.execute_query(check_query, (node_id_list,))

        labels_to_embed = {row[0]: row[1] for row in rows}

        if labels_to_embed:
            print(f"Generating embeddings for {len(labels_to_embed)} nodes...")

            node_ids = list(labels_to_embed.keys())
            texts = [labels_to_embed[nid] for nid in node_ids]

            try:
                embeddings = self.embedding.generate_embeddings_batch(
                    texts, task_type="RETRIEVAL_DOCUMENT"
                )
            except Exception as e:
                print(f"Error generating embeddings batch: {e}")
                return

            update_rows = [
                (vector_literal(vec), nid) for nid, vec in zip(node_ids, embeddings)
            ]

            update_query = """
                UPDATE kg_nodes
                SET embedding = (%s)::vector, updated_at = NOW()
                WHERE id = %s
            """
            self.postgres.execute_batch(update_query, update_rows)
