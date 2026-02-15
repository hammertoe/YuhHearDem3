"""Tests for KG cleanup rewrite module."""

import pytest

from lib.knowledge_graph.cleanup.rewrite import (
    rewrite_edge_endpoints,
    drop_invalid_edges,
    filter_discourse_edges,
    collapse_duplicate_edges,
    aggregate_provenance,
    rewrite_and_clean_edges,
)


def test_rewrite_edge_endpoints():
    """Test rewriting edge endpoints via merge map."""
    edges = [
        {"id": "edge1", "source_id": "node1", "target_id": "node2"},
        {"id": "edge2", "source_id": "node2", "target_id": "node3"},
        {"id": "edge3", "source_id": "node3", "target_id": "node4"},
    ]
    merge_map = {"node1": "nodeA", "node3": "nodeA"}

    rewritten = rewrite_edge_endpoints(edges, merge_map)
    assert rewritten[0]["source_id"] == "nodeA"
    assert rewritten[0]["target_id"] == "node2"
    assert rewritten[1]["source_id"] == "node2"
    assert rewritten[1]["target_id"] == "nodeA"
    assert rewritten[2]["source_id"] == "nodeA"
    assert rewritten[2]["target_id"] == "node4"


def test_drop_invalid_edges():
    """Test dropping edges with missing endpoints."""
    edges = [
        {"id": "edge1", "source_id": "node1", "target_id": "node2"},
        {"id": "edge2", "source_id": "node_missing", "target_id": "node3"},
        {"id": "edge3", "source_id": "node3", "target_id": "node_missing"},
    ]
    valid_nodes = {"node1", "node2", "node3"}

    filtered = drop_invalid_edges(edges, valid_nodes)
    assert len(filtered) == 1
    assert filtered[0]["id"] == "edge1"


def test_filter_discourse_edges():
    """Test filtering discourse edges to Person->Person only."""
    edges = [
        {
            "id": "edge1",
            "source_id": "node1",
            "target_id": "node2",
            "predicate": "RESPONDS_TO",
            "source_type": "foaf:Person",
            "target_type": "foaf:Person",
        },
        {
            "id": "edge2",
            "source_id": "node1",
            "target_id": "node3",
            "predicate": "RESPONDS_TO",
            "source_type": "foaf:Person",
            "target_type": "schema:Legislation",
        },
        {
            "id": "edge3",
            "source_id": "node1",
            "target_id": "node2",
            "predicate": "PROPOSES",
            "source_type": "foaf:Person",
            "target_type": "foaf:Person",
        },
    ]

    filtered = filter_discourse_edges(edges)
    assert len(filtered) == 2
    assert filtered[0]["id"] == "edge1"
    assert filtered[1]["id"] == "edge3"


def test_collapse_duplicate_edges():
    """Test collapsing duplicate edges by key."""
    edges = [
        {
            "id": "edge1",
            "source_id": "node1",
            "target_id": "node2",
            "predicate": "GOVERNS",
            "youtube_video_id": "video1",
            "earliest_timestamp_str": "00:10:00",
            "earliest_seconds": 600,
            "utterance_ids": ["u1"],
            "evidence": "evidence A",
            "confidence": 0.8,
            "edge_weight": 0.2,
            "edge_rank_score": 0.3,
        },
        {
            "id": "edge2",
            "source_id": "node1",
            "target_id": "node2",
            "predicate": "GOVERNS",
            "youtube_video_id": "video1",
            "earliest_timestamp_str": "00:05:00",
            "earliest_seconds": 300,
            "utterance_ids": ["u2"],
            "evidence": "evidence B",
            "confidence": 0.9,
            "edge_weight": 0.4,
            "edge_rank_score": 0.5,
        },
        {
            "id": "edge3",
            "source_id": "node1",
            "target_id": "node2",
            "predicate": "GOVERNS",
            "youtube_video_id": "video2",
            "earliest_timestamp_str": "00:10:00",
            "earliest_seconds": 600,
            "utterance_ids": ["u3"],
            "evidence": "evidence C",
            "confidence": 0.7,
            "edge_weight": 0.1,
            "edge_rank_score": 0.2,
        },
    ]

    collapsed = collapse_duplicate_edges(edges)
    assert len(collapsed) == 2
    assert collapsed[0]["earliest_seconds"] == 300
    assert "u1" in collapsed[0]["utterance_ids"]
    assert "u2" in collapsed[0]["utterance_ids"]
    assert collapsed[0]["support_count"] == 2
    assert collapsed[0]["confidence"] == pytest.approx(0.9, rel=0.01)
    assert collapsed[0]["edge_weight"] == pytest.approx(0.4, rel=0.01)
    assert collapsed[0]["edge_rank_score"] == pytest.approx(0.5, rel=0.01)


def test_aggregate_provenance():
    """Test provenance aggregation."""
    edges = [
        {
            "id": "edge1",
            "earliest_timestamp_str": "00:10:00",
            "earliest_seconds": 600,
            "utterance_ids": ["u1", "u2"],
            "evidence": "evidence A",
            "confidence": 0.8,
        },
        {
            "id": "edge2",
            "earliest_timestamp_str": "00:05:00",
            "earliest_seconds": 300,
            "utterance_ids": ["u3"],
            "evidence": "evidence B",
            "confidence": 0.9,
        },
    ]

    aggregated = aggregate_provenance(edges)
    assert aggregated["earliest_timestamp_str"] == "00:05:00"
    assert aggregated["earliest_seconds"] == 300
    assert "u1" in aggregated["utterance_ids"]
    assert "u2" in aggregated["utterance_ids"]
    assert "u3" in aggregated["utterance_ids"]
    assert aggregated["confidence"] == pytest.approx(0.9, rel=0.01)
    assert aggregated["support_count"] == 2


def test_rewrite_and_clean_edges():
    """Test full edge rewriting and cleaning pipeline."""
    edges = [
        {
            "id": "edge1",
            "source_id": "node1",
            "target_id": "node2",
            "predicate": "GOVERNS",
            "youtube_video_id": "video1",
            "source_type": "foaf:Person",
            "target_type": "schema:Organization",
            "earliest_timestamp_str": "00:10:00",
            "earliest_seconds": 600,
            "utterance_ids": ["u1"],
            "evidence": "evidence A",
            "confidence": 0.8,
        },
        {
            "id": "edge2",
            "source_id": "node1",
            "target_id": "node2",
            "predicate": "RESPONDS_TO",
            "youtube_video_id": "video1",
            "source_type": "foaf:Person",
            "target_type": "schema:Legislation",
            "earliest_timestamp_str": "00:05:00",
            "earliest_seconds": 300,
            "utterance_ids": ["u2"],
            "evidence": "evidence B",
            "confidence": 0.9,
        },
    ]
    merge_map = {}
    valid_nodes = {"node1", "node2"}

    result = rewrite_and_clean_edges(edges, merge_map, valid_nodes)
    assert "dropped_edges" in result
    assert "clean_edges" in result
    assert len(result["dropped_edges"]) == 1
    assert result["dropped_edges"][0] == ("edge2", "discourse_constraint")
    assert len(result["clean_edges"]) == 1
    assert result["clean_edges"][0]["predicate"] == "GOVERNS"


def test_rewrite_and_clean_edges_keeps_original_invalid_edge_id_in_log():
    """Test dropped invalid edges preserve original edge IDs."""
    edges = [
        {
            "id": "edge_missing_target",
            "source_id": "node1",
            "target_id": "missing",
            "predicate": "GOVERNS",
            "source_type": "foaf:Person",
            "target_type": "schema:Organization",
        }
    ]
    result = rewrite_and_clean_edges(edges, {}, {"node1"})

    assert result["clean_edges"] == []
    assert result["dropped_edges"] == [("edge_missing_target", "invalid_endpoint")]
