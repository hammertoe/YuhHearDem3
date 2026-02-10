"""Unit tests for ID generators."""

from lib.id_generators import (
    generate_kg_edge_id,
    generate_kg_node_id,
    normalize_label,
)


def test_normalize_label():
    """Test label normalization."""
    assert normalize_label("Fixed Penalty Regime") == "fixed penalty regime"
    assert normalize_label("  Fixed Penalty Regime  ") == "fixed penalty regime"
    assert normalize_label("Fixed   Penalty   Regime") == "fixed penalty regime"
    assert normalize_label("FIXED PENALTY REGIME") == "fixed penalty regime"


def test_generate_kg_node_id():
    """Test KG node ID generation."""
    id1 = generate_kg_node_id("skos:Concept", "Fixed Penalty Regime")
    assert id1.startswith("kg_")
    assert len(id1) == 3 + 12

    id2 = generate_kg_node_id("skos:Concept", "fixed penalty regime")
    assert id1 == id2

    id3 = generate_kg_node_id("schema:Legislation", "Fixed Penalty Regime")
    assert id1 != id3


def test_generate_kg_edge_id():
    """Test KG edge ID generation."""
    id1 = generate_kg_edge_id(
        "kg_abc123",
        "PROPOSES",
        "kg_def456",
        "video123",
        100,
        "I propose this bill...",
    )
    assert id1.startswith("kge_")
    assert len(id1) == 4 + 12

    id2 = generate_kg_edge_id(
        "kg_abc123",
        "PROPOSES",
        "kg_def456",
        "video123",
        100,
        "I propose this bill...",
    )
    assert id1 == id2

    id3 = generate_kg_edge_id(
        "kg_abc123",
        "PROPOSES",
        "kg_def456",
        "video123",
        100,
        "I propose this act...",
    )
    assert id1 != id3
