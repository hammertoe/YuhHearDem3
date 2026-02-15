"""Tests for KG cleanup contracts module."""

from lib.knowledge_graph.cleanup.contracts import (
    ALLOWED_NODE_TYPES,
    ALLOWED_PREDICATES,
    DISCOURSE_PREDICATES,
    NODE_TYPE_REMAPS,
    PREDICATE_REMAPS,
    GENERIC_LABEL_GUARDS,
    PREDICATE_PRIOR_WEIGHTS,
    get_remapped_node_type,
    get_remapped_predicate,
    is_node_type_allowed,
    is_predicate_allowed,
    is_discourse_predicate,
    is_generic_guarded_label,
)


def test_allowed_node_types_is_complete():
    """Test that all expected node types are in allowlist."""
    expected = {
        "foaf:Person",
        "schema:Legislation",
        "schema:Organization",
        "schema:Place",
        "skos:Concept",
    }
    assert ALLOWED_NODE_TYPES == expected


def test_allowed_predicates_is_complete():
    """Test that all expected predicates are in allowlist."""
    expected_predicates = {
        "AMENDS",
        "GOVERNS",
        "MODERNIZES",
        "AIMS_TO_REDUCE",
        "REQUIRES_APPROVAL",
        "IMPLEMENTED_BY",
        "RESPONSIBLE_FOR",
        "ASSOCIATED_WITH",
        "CAUSES",
        "ADDRESSES",
        "PROPOSES",
        "RESPONDS_TO",
        "AGREES_WITH",
        "DISAGREES_WITH",
        "QUESTIONS",
    }
    assert ALLOWED_PREDICATES == expected_predicates


def test_discourse_predicates_subset_of_allowed():
    """Test that discourse predicates are a subset of allowed predicates."""
    assert DISCOURSE_PREDICATES.issubset(ALLOWED_PREDICATES)
    expected = {"RESPONDS_TO", "AGREES_WITH", "DISAGREES_WITH", "QUESTIONS"}
    assert DISCOURSE_PREDICATES == expected


def test_is_node_type_allowed():
    """Test node type allowlist checking."""
    assert is_node_type_allowed("foaf:Person") is True
    assert is_node_type_allowed("schema:Legislation") is True
    assert is_node_type_allowed("skos:Concept") is True
    assert is_node_type_allowed("foaf:Organization") is False
    assert is_node_type_allowed("schema:Person") is False
    assert is_node_type_allowed("unknown:Type") is False


def test_is_predicate_allowed():
    """Test predicate allowlist checking."""
    assert is_predicate_allowed("PROPOSES") is True
    assert is_predicate_allowed("AGREES_WITH") is True
    assert is_predicate_allowed("QUESTION") is False
    assert is_predicate_allowed("unknown:Predicate") is False


def test_is_discourse_predicate():
    """Test discourse predicate identification."""
    assert is_discourse_predicate("RESPONDS_TO") is True
    assert is_discourse_predicate("AGREES_WITH") is True
    assert is_discourse_predicate("DISAGREES_WITH") is True
    assert is_discourse_predicate("QUESTIONS") is True
    assert is_discourse_predicate("PROPOSES") is False
    assert is_discourse_predicate("GOVERNS") is False


def test_get_remapped_node_type():
    """Test node type remapping."""
    assert get_remapped_node_type("foaf:Person") == "foaf:Person"
    assert get_remapped_node_type("schema:Person") == "foaf:Person"
    assert get_remapped_node_type("foaf:Organization") == "schema:Organization"
    assert get_remapped_node_type("schema:Organization") == "schema:Organization"
    assert get_remapped_node_type("unknown:Type") is None


def test_get_remapped_predicate():
    """Test predicate remapping."""
    assert get_remapped_predicate("AGREES_WITH") == "AGREES_WITH"
    assert get_remapped_predicate("AGREE_WITH") == "AGREES_WITH"
    assert get_remapped_predicate("DISAGREES_WITH") == "DISAGREES_WITH"
    assert get_remapped_predicate("DISAGREE_WITH") == "DISAGREES_WITH"
    assert get_remapped_predicate("AIMS_TO") == "AIMS_TO_REDUCE"
    assert get_remapped_predicate("QUESTION") == "QUESTIONS"
    assert get_remapped_predicate("unknown:Predicate") is None


def test_is_generic_guarded_label():
    """Test generic label guardrails."""
    assert is_generic_guarded_label("government") is True
    assert is_generic_guarded_label("member") is True
    assert is_generic_guarded_label("bill") is True
    assert is_generic_guarded_label("John Smith") is False
    assert is_generic_guarded_label("Climate Change Act") is False


def test_generic_label_guards_is_frozenset():
    """Test that generic label guards is a frozenset for immutability."""
    assert isinstance(GENERIC_LABEL_GUARDS, frozenset)


def test_predicate_prior_weights_sum_to_one():
    """Test that predicate prior weights sum to approximately 1.0."""
    total_weight = sum(PREDICATE_PRIOR_WEIGHTS.values())
    assert abs(total_weight - 1.0) < 0.01


def test_predicate_prior_weights_for_allowed_predicates():
    """Test that all predicate prior weights are for allowed predicates."""
    for predicate in PREDICATE_PRIOR_WEIGHTS:
        assert predicate in ALLOWED_PREDICATES


def test_predicate_prior_weights_are_positive():
    """Test that all predicate prior weights are positive."""
    for weight in PREDICATE_PRIOR_WEIGHTS.values():
        assert weight > 0


def test_remap_tables_are_dicts():
    """Test that remap tables are dictionaries."""
    assert isinstance(NODE_TYPE_REMAPS, dict)
    assert isinstance(PREDICATE_REMAPS, dict)


def test_remap_values_are_in_allowlists():
    """Test that all remapped values are in allowlists."""
    for target_type in NODE_TYPE_REMAPS.values():
        assert target_type in ALLOWED_NODE_TYPES

    for target_pred in PREDICATE_REMAPS.values():
        assert target_pred in ALLOWED_PREDICATES
