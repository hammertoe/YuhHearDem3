"""Test KG cleanup ranking migration."""

import pytest

from lib.db.postgres_client import PostgresClient


@pytest.fixture
def postgres():
    return PostgresClient()


def test_kg_nodes_has_ranking_columns(postgres: PostgresClient):
    """Test that kg_nodes table has ranking columns."""
    query = """
    SELECT column_name, data_type, is_nullable, column_default
    FROM information_schema.columns
    WHERE table_name = 'kg_nodes'
    AND column_name IN ('pagerank_score', 'merge_cluster_id', 'merged_from_count')
    ORDER BY column_name
    """
    rows = postgres.execute_query(query)
    columns = {row[0]: {"type": row[1], "nullable": row[2], "default": row[3]} for row in rows}

    assert "pagerank_score" in columns
    assert columns["pagerank_score"]["type"] == "double precision"
    assert columns["pagerank_score"]["nullable"] == "YES"

    assert "merge_cluster_id" in columns
    assert columns["merge_cluster_id"]["type"] == "text"
    assert columns["merge_cluster_id"]["nullable"] == "YES"

    assert "merged_from_count" in columns
    assert columns["merged_from_count"]["type"] == "integer"
    assert columns["merged_from_count"]["nullable"] == "NO"


def test_kg_edges_has_support_and_ranking_columns(postgres: PostgresClient):
    """Test that kg_edges table has support and ranking columns."""
    query = """
    SELECT column_name, data_type, is_nullable, column_default
    FROM information_schema.columns
    WHERE table_name = 'kg_edges'
    AND column_name IN ('support_count', 'edge_weight', 'edge_rank_score')
    ORDER BY column_name
    """
    rows = postgres.execute_query(query)
    columns = {row[0]: {"type": row[1], "nullable": row[2], "default": row[3]} for row in rows}

    assert "support_count" in columns
    assert columns["support_count"]["type"] == "integer"
    assert columns["support_count"]["nullable"] == "NO"

    assert "edge_weight" in columns
    assert columns["edge_weight"]["type"] == "double precision"
    assert columns["edge_weight"]["nullable"] == "YES"

    assert "edge_rank_score" in columns
    assert columns["edge_rank_score"]["type"] == "double precision"
    assert columns["edge_rank_score"]["nullable"] == "YES"


def test_kg_nodes_ranking_indexes_exist(postgres: PostgresClient):
    """Test that ranking indexes exist on kg_nodes."""
    query = """
    SELECT indexname
    FROM pg_indexes
    WHERE tablename = 'kg_nodes'
    AND indexname IN ('idx_kg_nodes_pagerank_score', 'idx_kg_nodes_merge_cluster_id')
    ORDER BY indexname
    """
    rows = postgres.execute_query(query)
    index_names = {row[0] for row in rows}

    assert "idx_kg_nodes_pagerank_score" in index_names
    assert "idx_kg_nodes_merge_cluster_id" in index_names


def test_kg_edges_ranking_indexes_exist(postgres: PostgresClient):
    """Test that ranking indexes exist on kg_edges."""
    query = """
    SELECT indexname
    FROM pg_indexes
    WHERE tablename = 'kg_edges'
    AND indexname IN ('idx_kg_edges_support_count', 'idx_kg_edges_edge_rank_score')
    ORDER BY indexname
    """
    rows = postgres.execute_query(query)
    index_names = {row[0] for row in rows}

    assert "idx_kg_edges_support_count" in index_names
    assert "idx_kg_edges_edge_rank_score" in index_names


def test_kg_edges_with_details_view_includes_new_columns(postgres: PostgresClient):
    """Test that kg_edges_with_details view includes new ranking columns."""
    query = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'kg_edges_with_details'
    AND table_schema = 'public'
    ORDER BY column_name
    """
    rows = postgres.execute_query(query)
    column_names = {row[0] for row in rows}

    assert "support_count" in column_names
    assert "edge_weight" in column_names
    assert "edge_rank_score" in column_names
