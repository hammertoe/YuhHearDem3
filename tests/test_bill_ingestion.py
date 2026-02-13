# Tests for bill ingestion.
import pytest

from lib.db.postgres_client import PostgresClient
from lib.embeddings.google_client import GoogleEmbeddingClient

from lib.id_generators import generate_bill_id, generate_entity_id


pytestmark = pytest.mark.integration


@pytest.fixture
def sample_bills():
    """Sample bills for ingestion testing."""
    return [
        {
            "bill_number": "HR 1234",
            "title": "Road Traffic (Amendment) Bill",
            "description": "Amends Road Traffic Act to increase penalties.",
            "status": "Introduced",
            "introduced_date": "2026-01-15",
            "source_url": "https://www.parliament.gov.bb/bill/HR1234",
            "source_text": "THE ROAD TRAFFIC ACT...",
            "category": "Transport",
            "extracted_entities": {
                "organizations": [{"text": "Ministry of Transport", "start": 0, "end": 21}],
                "topics": [{"text": "traffic safety", "start": 22, "end": 35}],
                "persons": [{"text": "John Smith", "start": 36, "end": 46}],
            },
        },
        {
            "bill_number": "SB 5678",
            "title": "Health Services Improvement Act",
            "description": "Improves healthcare services.",
            "status": "Passed",
            "passed_date": "2026-02-20",
            "source_url": "https://www.parliament.gov.bb/bill/SB5678",
            "source_text": "THE HEALTH SERVICES ACT...",
            "category": "Health",
            "extracted_entities": {
                "organizations": [{"text": "Ministry of Health", "start": 0, "end": 18}],
                "topics": [{"text": "healthcare services", "start": 19, "end": 40}],
            },
        },
    ]


def test_bill_ingestor_init():
    """Test bill ingestor initialization."""
    from lib.processors.bill_ingestor import BillIngestor

    with (
        PostgresClient() as postgres,
        GoogleEmbeddingClient() as embeddings,
    ):
        ingestor = BillIngestor()

        assert ingestor.postgres is not None
        assert ingestor.embedding_client is not None
        assert ingestor.entity_extractor is not None

        print("✅ Bill ingestor initialization works")


def test_ingest_bill_postgres():
    """Test bill ingestion into PostgreSQL."""
    from lib.processors.bill_ingestor import BillIngestor

    with (
        PostgresClient() as postgres,
        GoogleEmbeddingClient() as embeddings,
    ):
        ingestor = BillIngestor()

        bill_data = {
            "bill_number": "HR 1234",
            "title": "Road Traffic (Amendment) Bill",
            "description": "Amends Road Traffic Act.",
            "status": "Introduced",
            "category": "Transport",
        }

        bill_id = generate_bill_id("HR 1234", set())
        entity_id = generate_entity_id("Road Traffic (Amendment) Bill", "BILL")

        ingestor._ingest_bill_postgres(bill_data, bill_id, entity_id)

        postgres.execute_query(
            """
            SELECT id, bill_number, title
            FROM bills
            WHERE bill_number = %s
        """,
            ("HR 1234",),
        )

        result = postgres.execute_query("SELECT COUNT(*) FROM bills")
        assert result[0][0] > 0

        print("✅ Bill PostgreSQL ingestion works")


def test_ingest_bill_embeddings():
    """Test bill embedding generation and storage."""
    from lib.processors.bill_ingestor import BillIngestor
    from unittest.mock import patch

    with PostgresClient() as postgres, GoogleEmbeddingClient() as embeddings:
        ingestor = BillIngestor()

        bill_data = {
            "title": "Test Bill",
            "description": "Test description",
            "id": "test_bill_id",
        }

        bill_id = generate_bill_id("TEST", set())
        entity_id = generate_entity_id("Test Bill", "BILL")

        with patch.object(ingestor.embedding_client, "generate_embedding") as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3] * 256

            ingestor._ingest_bill_embeddings(bill_data, entity_id)

            result = postgres.execute_query(
                """
                SELECT embedding FROM entities WHERE id = %s
            """,
                (entity_id,),
            )

            if len(result) > 0 and result[0][0] is not None:
                print("✅ Bill embedding storage works")
            else:
                pytest.fail("Embedding not stored in entities table")


def test_ingest_bills_batch():
    """Test batch bill ingestion."""
    from lib.processors.bill_ingestor import BillIngestor

    bills = [
        {
            "bill_number": "HR 1234",
            "title": "Road Traffic (Amendment) Bill",
            "status": "Introduced",
        },
        {"bill_number": "SB 5678", "title": "Health Services Act", "status": "Passed"},
    ]

    with (
        PostgresClient() as postgres,
        GoogleEmbeddingClient() as embeddings,
    ):
        ingestor = BillIngestor()

        result = postgres.execute_query("SELECT COUNT(*) FROM bills")
        initial_count = result[0][0] if result and result[0] else 0

        ingestor.ingest_bills(bills)

        result = postgres.execute_query("SELECT COUNT(*) FROM bills")
        final_count = result[0][0] if result and result[0] else 0

        assert final_count == initial_count + len(bills)

        print("✅ Batch bill ingestion works")


def test_update_existing_bills():
    """Test updating existing bills (ON CONFLICT)."""
    from lib.processors.bill_ingestor import BillIngestor

    with PostgresClient() as postgres:
        ingestor = BillIngestor()

        bill_data = {
            "bill_number": "HR 1234",
            "title": "Road Traffic (Amendment) Bill (Updated)",
            "description": "Updated description.",
            "status": "Passed",
        }

        bill_id = generate_bill_id("HR 1234", set())
        entity_id = generate_entity_id("Road Traffic (Amendment) Bill (Updated)", "BILL")

        ingestor._ingest_bill_postgres(bill_data, bill_id, entity_id)

        result = postgres.execute_query(
            """
            SELECT title, status FROM bills WHERE bill_number = %s
        """,
            ("HR 1234",),
        )

        assert len(result) > 0
        assert result[0][0] == ("Road Traffic (Amendment) Bill (Updated)", "Passed")

        print("✅ Bill update works")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
