from __future__ import annotations

from unittest.mock import Mock

from lib.id_generators import generate_bill_id, generate_entity_id
from lib.processors.bill_ingestor import BillIngestor


def test_bill_ingestor_inserts_entity_and_bill_and_graph() -> None:
    postgres = Mock()
    memgraph = Mock()
    embeddings = Mock()
    embeddings.generate_embedding.return_value = [0.0] * 768

    ingestor = BillIngestor(
        postgres=postgres,
        memgraph=memgraph,
        embedding_client=embeddings,
    )

    bill = {
        "bill_number": "HR 1234",
        "title": "Road Traffic (Amendment) Bill",
        "description": "Amends Road Traffic Act.",
        "status": "Introduced",
        "category": "Transport",
        "keywords": ["road", "traffic"],
        "extracted_entities": {"topics": [{"text": "traffic safety"}]},
    }

    bill_id = generate_bill_id(bill["bill_number"], set())
    entity_id = generate_entity_id(bill["title"], "BILL")

    ingestor.ingest_bill(bill, bill_id=bill_id, entity_id=entity_id)

    # Postgres should be asked to upsert entity, insert/update bill, and store embeddings.
    assert postgres.execute_update.call_count >= 3
    # Memgraph should be asked to merge a bill node.
    assert memgraph.merge_entity.call_count >= 1
