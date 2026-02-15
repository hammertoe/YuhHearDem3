"""Tests for KG cleanup candidates module."""

import pytest

from lib.knowledge_graph.cleanup.candidates import (
    generate_person_blocking_key,
    generate_legislation_blocking_key,
    generate_generic_blocking_key,
    build_type_blocks,
    compute_label_similarity,
    compute_embedding_similarity,
    compute_neighbor_jaccard,
    compute_alias_overlap,
    compute_merge_score,
    generate_candidate_pairs,
)


def test_generate_person_blocking_key():
    """Test person blocking key generation."""
    assert generate_person_blocking_key("John Smith") == ("smith", "js")
    assert generate_person_blocking_key("The Honourable Mary Jones") == ("jones", "mj")
    assert generate_person_blocking_key("Dr. A B C") == ("c", "abc")


def test_generate_legislation_blocking_key():
    """Test legislation blocking key generation."""
    assert generate_legislation_blocking_key("Climate Change Act 2008") == (
        "climate change act",
        "cca2008",
    )
    assert generate_legislation_blocking_key("Road Traffic Act 1991") == (
        "road traffic act",
        "rta1991",
    )


def test_generate_generic_blocking_key():
    """Test generic blocking key generation."""
    assert generate_generic_blocking_key("Environment Agency") == ("environment", "agency")
    assert generate_generic_blocking_key("London Borough") == ("london", "borough")


def test_build_type_blocks_person():
    """Test building blocks for person nodes."""
    nodes = {
        "node1": {
            "id": "node1",
            "type": "foaf:Person",
            "label": "John Smith",
            "aliases": ["J. Smith"],
        },
        "node2": {
            "id": "node2",
            "type": "foaf:Person",
            "label": "Jane Doe",
            "aliases": [],
        },
        "node3": {
            "id": "node3",
            "type": "foaf:Person",
            "label": "Mary Smith",
            "aliases": [],
        },
    }
    blocks = build_type_blocks(nodes, "foaf:Person")
    assert "smith" in blocks
    assert "doe" in blocks
    assert "node1" in blocks["smith"]
    assert "node3" in blocks["smith"]
    assert "node2" in blocks["doe"]


def test_build_type_blocks_legislation():
    """Test building blocks for legislation nodes."""
    nodes = {
        "node1": {
            "id": "node1",
            "type": "schema:Legislation",
            "label": "Climate Change Act 2008",
            "aliases": [],
        },
        "node2": {
            "id": "node2",
            "type": "schema:Legislation",
            "label": "Climate Change Act",
            "aliases": [],
        },
    }
    blocks = build_type_blocks(nodes, "schema:Legislation")
    assert "climate change act" in blocks
    assert "node1" in blocks["climate change act"]
    assert "node2" in blocks["climate change act"]


def test_compute_label_similarity():
    """Test label similarity computation."""
    sim = compute_label_similarity("John Smith", "Johnny Smith")
    assert sim > 0.7
    assert sim <= 1.0

    sim = compute_label_similarity("Alice", "Bob")
    assert sim < 0.3


def test_compute_embedding_similarity_none():
    """Test embedding similarity when embeddings are missing."""
    sim = compute_embedding_similarity(None, None)
    assert sim == 0.0

    sim = compute_embedding_similarity([0.1, 0.2], None)
    assert sim == 0.0


def test_compute_embedding_similarity_string_vectors():
    """Test embedding similarity when DB returns vector strings."""
    sim = compute_embedding_similarity("[0.1, 0.2, 0.3]", "[0.1, 0.2, 0.3]")
    assert sim == pytest.approx(1.0, rel=0.01)


def test_compute_embedding_similarity_invalid_string_vector():
    """Test embedding similarity returns 0 for invalid vector strings."""
    sim = compute_embedding_similarity("not-a-vector", "[0.1, 0.2]")
    assert sim == 0.0


def test_compute_neighbor_jaccard():
    """Test neighbor Jaccard similarity."""
    neighbors1 = {"node1", "node2", "node3"}
    neighbors2 = {"node1", "node2", "node4"}

    jaccard = compute_neighbor_jaccard(neighbors1, neighbors2)
    expected = len(neighbors1 & neighbors2) / len(neighbors1 | neighbors2)
    assert jaccard == pytest.approx(expected, rel=0.01)


def test_compute_neighbor_jaccard_empty():
    """Test neighbor Jaccard with empty sets."""
    jaccard = compute_neighbor_jaccard(set(), set())
    assert jaccard == 0.0


def test_compute_alias_overlap():
    """Test alias overlap computation."""
    aliases1 = ["john", "jonathan", "j smith"]
    aliases2 = ["johnny", "jonathan", "j smith"]

    overlap = compute_alias_overlap(aliases1, aliases2)
    assert overlap >= 0.5
    assert overlap <= 1.0


def test_compute_alias_overlap_empty():
    """Test alias overlap with empty lists."""
    overlap = compute_alias_overlap([], ["john"])
    assert overlap == 0.0


def test_compute_merge_score():
    """Test merge score computation."""
    label_sim = 0.9
    embedding_sim = 0.8
    neighbor_jaccard = 0.5
    alias_overlap = 0.6

    score = compute_merge_score(label_sim, embedding_sim, neighbor_jaccard, alias_overlap)
    expected = 0.45 * 0.9 + 0.30 * 0.8 + 0.15 * 0.5 + 0.10 * 0.6
    assert score == pytest.approx(expected, rel=0.01)


def test_generate_candidate_pairs_basic():
    """Test candidate pair generation from blocks."""
    nodes = {
        "node1": {
            "id": "node1",
            "type": "foaf:Person",
            "label": "John Smith",
            "aliases": ["J. Smith"],
            "neighbors": {"node2"},
            "embedding": [0.5, 0.6],
        },
        "node2": {
            "id": "node2",
            "type": "foaf:Person",
            "label": "Johnny Smith",
            "aliases": ["J. Smith"],
            "neighbors": {"node1"},
            "embedding": [0.5, 0.6],
        },
    }
    blocks = {"smith": ["node1", "node2"]}

    pairs = generate_candidate_pairs(nodes, blocks, "foaf:Person")
    assert len(pairs) == 1
    assert pairs[0][:2] == ("node1", "node2")


def test_generate_candidate_pairs_type_filtered():
    """Test candidate pairs are type-filtered."""
    nodes = {
        "node1": {
            "id": "node1",
            "type": "foaf:Person",
            "label": "John Smith",
            "aliases": [],
            "neighbors": set(),
            "embedding": None,
        },
        "node2": {
            "id": "node2",
            "type": "schema:Legislation",
            "label": "Act",
            "aliases": [],
            "neighbors": set(),
            "embedding": None,
        },
    }
    blocks = {"smith": ["node1"], "act": ["node2"]}

    pairs = generate_candidate_pairs(nodes, blocks, "foaf:Person")
    assert len(pairs) == 0


def test_generate_candidate_pairs_generic_guard():
    """Test generic label guardrail prevents auto-merge."""
    nodes = {
        "node1": {
            "id": "node1",
            "type": "foaf:Person",
            "label": "Government",
            "aliases": [],
            "neighbors": {"node2"},
            "embedding": None,
        },
        "node2": {
            "id": "node2",
            "type": "foaf:Person",
            "label": "Government",
            "aliases": [],
            "neighbors": {"node1"},
            "embedding": None,
        },
    }
    blocks = {"government": ["node1", "node2"]}

    pairs = generate_candidate_pairs(nodes, blocks, "foaf:Person")
    pairs_with_scores = [p for p in pairs if len(p) == 3 and p[2] > 0]

    assert len(pairs_with_scores) == 0


def test_generate_candidate_pairs_self_avoidance():
    """Test candidate pairs don't include self-comparisons."""
    nodes = {
        "node1": {
            "id": "node1",
            "type": "foaf:Person",
            "label": "John Smith",
            "aliases": [],
            "neighbors": set(),
            "embedding": None,
        },
    }
    blocks = {"smith": ["node1"]}

    pairs = generate_candidate_pairs(nodes, blocks, "foaf:Person")
    assert len(pairs) == 0
