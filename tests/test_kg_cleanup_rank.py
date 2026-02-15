"""Tests for KG cleanup ranking module."""

import math

import pytest

from lib.knowledge_graph.cleanup.rank import (
    normalize_scores,
    compute_edge_weight,
    compute_edge_rank_score,
    compute_pagerank_scores,
    compute_all_ranking_scores,
)


def test_normalize_scores_empty():
    """Test normalization with empty input."""
    scores = {}
    normalized = normalize_scores(scores)
    assert normalized == {}


def test_normalize_scores_single_value():
    """Test normalization with single value."""
    scores = {"node1": 1.0}
    normalized = normalize_scores(scores)
    assert normalized["node1"] == 1.0


def test_normalize_scores_multiple_values():
    """Test normalization with multiple values."""
    scores = {"node1": 1.0, "node2": 2.0, "node3": 3.0}
    normalized = normalize_scores(scores)
    assert normalized["node1"] == pytest.approx(1.0 / 3.0, rel=0.01)
    assert normalized["node2"] == pytest.approx(2.0 / 3.0, rel=0.01)
    assert normalized["node3"] == 1.0


def test_normalize_scores_zero_values():
    """Test normalization with zero values."""
    scores = {"node1": 0.0, "node2": 1.0}
    normalized = normalize_scores(scores)
    assert normalized["node1"] == 0.0
    assert normalized["node2"] == 1.0


def test_compute_edge_weight():
    """Test edge weight computation."""
    from lib.knowledge_graph.cleanup.contracts import PREDICATE_PRIOR_WEIGHTS

    support_count = 5
    confidence = 0.8
    predicate = "GOVERNS"
    predicate_weight = PREDICATE_PRIOR_WEIGHTS.get(predicate, 0.1)

    weight = compute_edge_weight(support_count, confidence, predicate_weight)
    expected = (
        0.50 * (math.log1p(support_count) / 10.0) + 0.35 * confidence + 0.15 * predicate_weight
    )
    assert weight == pytest.approx(expected, rel=0.01)


def test_compute_edge_rank_score():
    """Test edge rank score computation."""
    edge_weight = 0.5
    source_pr = 0.1
    target_pr = 0.2
    support_count = 5

    score = compute_edge_rank_score(edge_weight, source_pr, target_pr, support_count)
    avg_pr = (0.1 + 0.2) / 2
    expected = edge_weight * avg_pr * math.log1p(support_count)
    assert score == pytest.approx(expected, rel=0.01)


def test_compute_pagerank_simple_graph():
    """Test PageRank on simple graph."""
    edges = [
        {"source_id": "node1", "target_id": "node2"},
        {"source_id": "node2", "target_id": "node3"},
        {"source_id": "node3", "target_id": "node1"},
    ]
    pagerank = compute_pagerank_scores(edges)
    assert "node1" in pagerank
    assert "node2" in pagerank
    assert "node3" in pagerank
    assert all(pr > 0 for pr in pagerank.values())
    assert abs(sum(pagerank.values()) - 1.0) < 0.01


def test_compute_pagerank_empty_graph():
    """Test PageRank on empty graph."""
    edges = []
    pagerank = compute_pagerank_scores(edges)
    assert pagerank == {}


def test_compute_pagerank_single_node():
    """Test PageRank on graph with single node."""
    edges = []
    nodes = ["node1"]
    pagerank = compute_pagerank_scores(edges, node_ids=nodes)
    assert pagerank["node1"] == 1.0


def test_compute_pagerank_disconnected_components():
    """Test PageRank on graph with disconnected components."""
    edges = [
        {"source_id": "node1", "target_id": "node2"},
        {"source_id": "node3", "target_id": "node4"},
    ]
    nodes = ["node1", "node2", "node3", "node4"]
    pagerank = compute_pagerank_scores(edges, node_ids=nodes)
    assert len(pagerank) == 4
    assert abs(sum(pagerank.values()) - 1.0) < 0.01


def test_compute_all_ranking_scores():
    """Test computing all ranking scores."""
    edges = [
        {
            "id": "edge1",
            "source_id": "node1",
            "target_id": "node2",
            "predicate": "GOVERNS",
            "support_count": 5,
            "confidence": 0.8,
        },
        {
            "id": "edge2",
            "source_id": "node2",
            "target_id": "node3",
            "predicate": "PROPOSES",
            "support_count": 3,
            "confidence": 0.6,
        },
    ]
    nodes = {"node1": {}, "node2": {}, "node3": {}}

    node_scores, edge_scores = compute_all_ranking_scores(edges, nodes)

    assert "node1" in node_scores
    assert "node2" in node_scores
    assert "node3" in node_scores

    assert "edge1" in edge_scores
    assert "edge2" in edge_scores

    assert all(pr > 0 for pr in node_scores.values())
    assert all(ew > 0 for ew in edge_scores.values())


def test_compute_all_ranking_scores_empty():
    """Test computing all scores with empty input."""
    node_scores, edge_scores = compute_all_ranking_scores([], {})
    assert node_scores == {}
    assert edge_scores == {}
