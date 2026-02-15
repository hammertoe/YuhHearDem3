"""KG cleanup contracts: type/predicate allowlists and remaps."""

from typing import Final

ALLOWED_NODE_TYPES: Final = frozenset(
    {
        "foaf:Person",
        "schema:Legislation",
        "schema:Organization",
        "schema:Place",
        "skos:Concept",
    }
)

ALLOWED_PREDICATES: Final = frozenset(
    {
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
)

DISCOURSE_PREDICATES: Final = frozenset(
    {
        "RESPONDS_TO",
        "AGREES_WITH",
        "DISAGREES_WITH",
        "QUESTIONS",
    }
)

NODE_TYPE_REMAPS: Final = {
    "schema:Person": "foaf:Person",
    "foaf:Organization": "schema:Organization",
}

PREDICATE_REMAPS: Final = {
    "AGREE_WITH": "AGREES_WITH",
    "DISAGREE_WITH": "DISAGREES_WITH",
    "AIMS_TO": "AIMS_TO_REDUCE",
    "QUESTION": "QUESTIONS",
}

GENERIC_LABEL_GUARDS: Final = frozenset(
    {
        "government",
        "member",
        "bill",
        "act",
        "law",
        "legislation",
        "parliament",
        "committee",
        "minister",
        "house",
        "scheme",
        "system",
        "program",
        "policy",
    }
)

PREDICATE_PRIOR_WEIGHTS: Final = {
    "GOVERNS": 0.15,
    "RESPONSIBLE_FOR": 0.12,
    "IMPLEMENTED_BY": 0.12,
    "AMENDS": 0.10,
    "AIMS_TO_REDUCE": 0.10,
    "PROPOSES": 0.08,
    "ADDRESSES": 0.08,
    "ASSOCIATED_WITH": 0.07,
    "CAUSES": 0.06,
    "MODERNIZES": 0.05,
    "REQUIRES_APPROVAL": 0.04,
    "RESPONDS_TO": 0.02,
    "AGREES_WITH": 0.005,
    "DISAGREES_WITH": 0.005,
    "QUESTIONS": 0.005,
}


def is_node_type_allowed(node_type: str) -> bool:
    """Check if node type is in allowlist."""
    return node_type in ALLOWED_NODE_TYPES


def is_predicate_allowed(predicate: str) -> bool:
    """Check if predicate is in allowlist."""
    return predicate in ALLOWED_PREDICATES


def is_discourse_predicate(predicate: str) -> bool:
    """Check if predicate is a discourse predicate (Person->Person only)."""
    return predicate in DISCOURSE_PREDICATES


def get_remapped_node_type(node_type: str) -> str | None:
    """Get remapped node type or None if no remap exists."""
    if node_type in ALLOWED_NODE_TYPES:
        return node_type
    return NODE_TYPE_REMAPS.get(node_type)


def get_remapped_predicate(predicate: str) -> str | None:
    """Get remapped predicate or None if no remap exists."""
    if predicate in ALLOWED_PREDICATES:
        return predicate
    return PREDICATE_REMAPS.get(predicate)


def is_generic_guarded_label(label: str) -> bool:
    """Check if label is a generic guarded label."""
    normalized = label.lower().strip()
    return normalized in GENERIC_LABEL_GUARDS
