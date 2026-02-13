"""Database connection tests."""

import pytest
from lib.db.postgres_client import PostgresClient


pytestmark = pytest.mark.integration


@pytest.fixture
def postgres_client():
    """Create a PostgreSQL client for testing."""
    with PostgresClient() as client:
        yield client


def test_postgres_connection(postgres_client):
    """Test PostgreSQL connection."""
    result = postgres_client.execute_query("SELECT 1")
    assert result == [(1,)]
    print("✅ PostgreSQL connection successful")


def test_postgres_tables_exist(postgres_client):
    """Test that all tables were created."""
    tables = postgres_client.execute_query("""
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public'
    """)
    table_names = {row[0] for row in tables}

    required_tables = {
        "paragraphs",
        "entities",
        "sentences",
        "speakers",
        "bills",
        "paragraph_entities",
        "sentence_entities",
        "videos",
    }

    missing_tables = required_tables - table_names
    assert not missing_tables, f"Missing tables: {missing_tables}"
    print(f"✅ All {len(required_tables)} tables exist")


def test_postgres_indexes_exist(postgres_client):
    """Test that vector indexes exist."""
    indexes = postgres_client.execute_query("""
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = 'public'
        AND indexname LIKE 'idx_%'
    """)
    index_names = {row[0] for row in indexes}

    vector_indexes = {"idx_paragraphs_embedding", "idx_entities_embedding"}

    missing_indexes = vector_indexes - index_names
    assert not missing_indexes, f"Missing indexes: {missing_indexes}"
    print(f"✅ Vector indexes exist: {len(index_names)} indexes total")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
