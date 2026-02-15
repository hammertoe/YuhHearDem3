"""KG cleanup ranking functions."""

import math
from typing import Any

import networkx as nx

from lib.knowledge_graph.cleanup.contracts import PREDICATE_PRIOR_WEIGHTS


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Normalize scores to [0, 1] range by dividing by max."""
    if not scores:
        return {}

    max_score = max(scores.values())
    if max_score == 0:
        return {k: 0.0 for k in scores}

    return {k: v / max_score for k, v in scores.items()}


def compute_edge_weight(support_count: int, confidence: float, predicate_weight: float) -> float:
    """Compute edge weight from support, confidence, and predicate prior.

    Formula: 0.50 * support_norm + 0.35 * confidence + 0.15 * predicate_prior
    """
    support_norm = math.log1p(support_count) / 10.0
    support_norm = min(support_norm, 1.0)

    weight = 0.50 * support_norm + 0.35 * confidence + 0.15 * predicate_weight
    return weight


def compute_edge_rank_score(
    edge_weight: float, source_pr: float, target_pr: float, support_count: int
) -> float:
    """Compute edge rank score combining edge weight and endpoint PageRank.

    Formula: edge_weight * avg(PR(source), PR(target)) * log1p(support_count)
    """
    avg_pr = (source_pr + target_pr) / 2.0
    score = edge_weight * avg_pr * math.log1p(support_count)
    return score


def compute_pagerank_scores(
    edges: list[dict[str, Any]], node_ids: list[str] | None = None
) -> dict[str, float]:
    """Compute PageRank scores for nodes in the graph.

    Args:
        edges: List of edge dictionaries with source_id and target_id
        node_ids: Optional list of node IDs to include (for isolated nodes)

    Returns:
        Dictionary mapping node_id to PageRank score
    """
    G = nx.DiGraph()

    for edge in edges:
        source_id = edge.get("source_id")
        target_id = edge.get("target_id")
        if source_id and target_id:
            G.add_edge(source_id, target_id)

    if node_ids:
        for node_id in node_ids:
            if node_id not in G:
                G.add_node(node_id)

    if G.number_of_nodes() == 0:
        return {}

    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
    return {k: float(v) for k, v in pagerank.items()}


def compute_all_ranking_scores(
    edges: list[dict[str, Any]], nodes: dict[str, dict[str, Any]]
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute all ranking scores for nodes and edges.

    Args:
        edges: List of edge dictionaries
        nodes: Dictionary of node_id -> node_info

    Returns:
        Tuple of (node_pagerank_scores, edge_rank_scores)
    """
    node_ids = list(nodes.keys())
    pagerank_scores = compute_pagerank_scores(edges, node_ids)

    edge_rank_scores: dict[str, float] = {}

    for edge in edges:
        edge_id = edge.get("id")
        source_id = edge.get("source_id")
        target_id = edge.get("target_id")
        predicate = edge.get("predicate")
        support_count = edge.get("support_count", 1)
        confidence = edge.get("confidence", 0.5)

        if not edge_id or not source_id or not target_id or not predicate:
            continue

        predicate_weight = PREDICATE_PRIOR_WEIGHTS.get(predicate, 0.1)
        edge_weight = compute_edge_weight(support_count, confidence, predicate_weight)

        source_pr = pagerank_scores.get(source_id, 0.0)
        target_pr = pagerank_scores.get(target_id, 0.0)

        rank_score = compute_edge_rank_score(edge_weight, source_pr, target_pr, support_count)
        edge_rank_scores[edge_id] = rank_score

    return pagerank_scores, edge_rank_scores
