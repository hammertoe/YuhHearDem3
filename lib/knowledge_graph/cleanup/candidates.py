"""KG cleanup candidate generation and blocking."""

import json
from typing import Any

import numpy as np
from rapidfuzz import fuzz

from lib.knowledge_graph.cleanup.contracts import is_generic_guarded_label
from lib.knowledge_graph.cleanup.normalize import (
    extract_initials,
    extract_surname,
    strip_honorifics,
    normalize_for_matching,
)


def generate_person_blocking_key(label: str) -> tuple[str, str]:
    """Generate blocking key for Person nodes (surname, initials)."""
    label_stripped = strip_honorifics(label)
    surname = extract_surname(label_stripped)
    initials = extract_initials(label_stripped)
    return (surname.lower(), initials.lower())


def generate_legislation_blocking_key(label: str) -> tuple[str, str]:
    """Generate blocking key for Legislation nodes (title, abbreviated)."""
    from lib.knowledge_graph.cleanup.normalize import normalize_legislation_key

    title = normalize_legislation_key(label)
    parts = title.split()
    if len(parts) >= 3:
        abb = "".join(p[0].lower() for p in parts[:3])
    elif len(parts) >= 2:
        abb = (parts[0][0] + parts[1][0]).lower()
    else:
        abb = (parts[0][:3] if parts else "").lower()

    year_match = ""
    import re

    year_match_obj = re.search(r"\b\d{4}\b", label)
    if year_match_obj:
        year = year_match_obj.group()
        year_match = year

    return (title, f"{abb}{year_match}")


def generate_generic_blocking_key(label: str) -> tuple[str, str]:
    """Generate blocking key for Organization/Place/Concept (first token, last token)."""
    normalized = normalize_for_matching(label)
    tokens = normalized.split()
    if len(tokens) >= 2:
        return (tokens[0], tokens[-1])
    elif tokens:
        return (tokens[0], tokens[0])
    return ("", "")


def build_type_blocks(nodes: dict[str, dict[str, Any]], node_type: str) -> dict[str, list[str]]:
    """Build blocking buckets for nodes of a specific type."""
    blocks: dict[str, list[str]] = {}

    for node_id, node in nodes.items():
        if node.get("type") != node_type:
            continue

        label = node.get("label", "")
        if not label:
            continue

        if node_type == "foaf:Person":
            surname, _ = generate_person_blocking_key(label)
            if surname:
                block_key = surname
            else:
                continue
        elif node_type == "schema:Legislation":
            title, _ = generate_legislation_blocking_key(label)
            block_key = title
        else:
            first, last = generate_generic_blocking_key(label)
            block_key = f"{first}|{last}"

        if block_key not in blocks:
            blocks[block_key] = []
        blocks[block_key].append(node_id)

    return blocks


def compute_label_similarity(label1: str, label2: str) -> float:
    """Compute label similarity using rapidfuzz."""
    if not label1 or not label2:
        return 0.0

    normalized1 = normalize_for_matching(label1)
    normalized2 = normalize_for_matching(label2)

    return fuzz.ratio(normalized1, normalized2) / 100.0


def compute_embedding_similarity(
    emb1: list[float] | str | None, emb2: list[float] | str | None
) -> float:
    """Compute cosine similarity between embeddings."""
    vec1 = _coerce_embedding_vector(emb1)
    vec2 = _coerce_embedding_vector(emb2)

    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    arr1 = np.array(vec1, dtype=float)
    arr2 = np.array(vec2, dtype=float)

    dot_product = np.dot(arr1, arr2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def _coerce_embedding_vector(embedding: list[float] | str | None) -> list[float]:
    """Coerce embedding value from DB into a list of floats."""
    if embedding is None:
        return []

    if isinstance(embedding, list):
        try:
            return [float(value) for value in embedding]
        except (TypeError, ValueError):
            return []

    if isinstance(embedding, str):
        text = embedding.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []

        if not isinstance(parsed, list):
            return []

        try:
            return [float(value) for value in parsed]
        except (TypeError, ValueError):
            return []

    return []


def compute_neighbor_jaccard(neighbors1: set[str], neighbors2: set[str]) -> float:
    """Compute Jaccard similarity between neighbor sets."""
    if not neighbors1 or not neighbors2:
        return 0.0

    intersection = len(neighbors1 & neighbors2)
    union = len(neighbors1 | neighbors2)

    return intersection / union if union > 0 else 0.0


def compute_alias_overlap(aliases1: list[str], aliases2: list[str]) -> float:
    """Compute overlap score between alias lists."""
    if not aliases1 or not aliases2:
        return 0.0

    set1 = {normalize_for_matching(a) for a in aliases1}
    set2 = {normalize_for_matching(a) for a in aliases2}

    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def compute_merge_score(
    label_sim: float,
    embedding_sim: float,
    neighbor_jaccard: float,
    alias_overlap: float,
) -> float:
    """Compute merge score from similarity components.

    Formula: 0.45 * label_sim + 0.30 * embedding_sim +
             0.15 * neighbor_jaccard + 0.10 * alias_overlap
    """
    return 0.45 * label_sim + 0.30 * embedding_sim + 0.15 * neighbor_jaccard + 0.10 * alias_overlap


def generate_candidate_pairs(
    nodes: dict[str, dict[str, Any]],
    blocks: dict[str, list[str]],
    node_type: str,
    threshold: float = 0.7,
) -> list[tuple[str, str, float]]:
    """Generate candidate pairs within type using blocking and scoring.

    Args:
        nodes: Dictionary of node_id -> node_info
        blocks: Blocking buckets from build_type_blocks
        node_type: Type of nodes to generate candidates for
        threshold: Minimum similarity score to include candidate

    Returns:
        List of (node_id_1, node_id_2, similarity_score) tuples
    """
    candidate_pairs: list[tuple[str, str, float]] = []

    for block_key, node_ids in blocks.items():
        if len(node_ids) < 2:
            continue

        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                node_id_1 = node_ids[i]
                node_id_2 = node_ids[j]

                node1 = nodes.get(node_id_1, {})
                node2 = nodes.get(node_id_2, {})

                if node1.get("type") != node_type or node2.get("type") != node_type:
                    continue

                label1 = node1.get("label", "")
                label2 = node2.get("label", "")

                if is_generic_guarded_label(label1) or is_generic_guarded_label(label2):
                    neighbors1 = node1.get("neighbors", set())
                    neighbors2 = node2.get("neighbors", set())

                    if len(neighbors1) == 0 or len(neighbors2) == 0:
                        continue

                label_sim = compute_label_similarity(label1, label2)

                emb1 = node1.get("embedding")
                emb2 = node2.get("embedding")
                embedding_sim = compute_embedding_similarity(emb1, emb2)

                neighbors1 = node1.get("neighbors", set())
                neighbors2 = node2.get("neighbors", set())
                neighbor_jaccard = compute_neighbor_jaccard(neighbors1, neighbors2)

                aliases1 = node1.get("aliases", [])
                aliases2 = node2.get("aliases", [])
                alias_overlap = compute_alias_overlap(aliases1, aliases2)

                score = compute_merge_score(
                    label_sim, embedding_sim, neighbor_jaccard, alias_overlap
                )

                if score >= threshold:
                    candidate_pairs.append((node_id_1, node_id_2, score))

    return candidate_pairs
