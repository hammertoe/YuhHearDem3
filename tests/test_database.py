"""Database connection tests."""

import pytest
from lib.db.postgres_client import PostgresClient
from lib.db.memgraph_client import MemgraphClient


pytestmark = pytest.mark.integration


@pytest.fixture
def postgres_client():
    """Create a PostgreSQL client for testing."""
    with PostgresClient() as client:
        yield client


@pytest.fixture
def memgraph_client():
    """Create a Memgraph client for testing."""
    with MemgraphClient() as client:
        yield client


def test_postgres_connection(postgres_client):
    """Test PostgreSQL connection."""
    result = postgres_client.execute_query("SELECT 1")
    assert result == [(1,)]
    print("✅ PostgreSQL connection successful")


def test_memgraph_connection(memgraph_client):
    """Test Memgraph connection."""
    result = memgraph_client.execute_query("RETURN 1 AS test")
    assert result == [{"test": 1}]
    print("✅ Memgraph connection successful")


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


def test_memgraph_constraints_exist(memgraph_client):
    """Test that Memgraph constraints exist."""
    try:
        memgraph_client.execute_query("""
            SHOW CONSTRAINTS
        """)
        print("✅ Memgraph constraints accessible")
    except Exception as e:
        pytest.fail(f"Failed to query constraints: {e}")


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
