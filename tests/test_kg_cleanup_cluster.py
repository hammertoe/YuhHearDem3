"""Tests for KG cleanup cluster module (union-find)."""

from lib.knowledge_graph.cleanup.cluster import (
    UnionFind,
    find_canonical_node,
    build_merge_clusters,
    cluster_from_candidate_pairs,
)


def test_union_find_initial_state():
    """Test UnionFind initial state."""
    uf = UnionFind()
    assert uf.find("node1") == "node1"
    assert uf.find("node2") == "node2"


def test_union_find_union():
    """Test UnionFind union operation."""
    uf = UnionFind()
    uf.union("node1", "node2")
    assert uf.find("node1") == uf.find("node2")


def test_union_find_transitive():
    """Test UnionFind transitive property."""
    uf = UnionFind()
    uf.union("node1", "node2")
    uf.union("node2", "node3")
    root1 = uf.find("node1")
    root2 = uf.find("node2")
    root3 = uf.find("node3")
    assert root1 == root2 == root3


def test_union_find_independent_sets():
    """Test UnionFind maintains independent sets."""
    uf = UnionFind()
    uf.union("node1", "node2")
    uf.union("node3", "node4")
    set1_root = uf.find("node1")
    set2_root = uf.find("node3")
    assert set1_root != set2_root


def test_union_find_get_clusters():
    """Test UnionFind get_clusters returns correct groups."""
    uf = UnionFind()
    uf.union("node1", "node2")
    uf.union("node3", "node4")
    uf.union("node5", "node6")
    clusters = uf.get_clusters()
    assert len(clusters) == 3
    assert {"node1", "node2"} in clusters
    assert {"node3", "node4"} in clusters
    assert {"node5", "node6"} in clusters


def test_union_find_size():
    """Test UnionFind tracks cluster sizes."""
    uf = UnionFind()
    uf.union("node1", "node2")
    uf.union("node2", "node3")
    assert uf.size("node1") == 3
    assert uf.size("node4") == 1


def test_find_canonical_node_basic():
    """Test canonical node selection with basic preference."""
    nodes = [
        {"id": "node1", "label": "Label 1", "degree": 5, "alias_count": 2, "is_speaker": False},
        {"id": "node2", "label": "Label 2", "degree": 10, "alias_count": 3, "is_speaker": False},
        {"id": "node3", "label": "Label 3", "degree": 3, "alias_count": 1, "is_speaker": False},
    ]
    canonical = find_canonical_node(nodes)
    assert canonical["id"] == "node2"


def test_find_canonical_node_speaker_priority():
    """Test canonical node selection prioritizes speaker nodes."""
    nodes = [
        {"id": "node1", "label": "Label 1", "degree": 5, "alias_count": 2, "is_speaker": False},
        {
            "id": "speaker_node",
            "label": "Speaker Label",
            "degree": 2,
            "alias_count": 1,
            "is_speaker": True,
        },
    ]
    canonical = find_canonical_node(nodes)
    assert canonical["id"] == "speaker_node"


def test_find_canonical_node_deterministic_tiebreak():
    """Test canonical node selection has deterministic tiebreak."""
    nodes = [
        {"id": "node2", "label": "Label", "degree": 5, "alias_count": 2, "is_speaker": False},
        {"id": "node1", "label": "Label", "degree": 5, "alias_count": 2, "is_speaker": False},
    ]
    canonical = find_canonical_node(nodes)
    assert canonical["id"] == "node1"


def test_find_canonical_node_prefer_longer_label():
    """Test canonical node selection prefers longer labels."""
    nodes = [
        {
            "id": "node1",
            "label": "R R Straughn",
            "degree": 5,
            "alias_count": 2,
            "is_speaker": False,
        },
        {
            "id": "node2",
            "label": "The Honourable Ryan Straughn Mp",
            "degree": 5,
            "alias_count": 2,
            "is_speaker": False,
        },
    ]
    canonical = find_canonical_node(nodes)
    assert canonical["id"] == "node2"
    assert canonical["label"] == "The Honourable Ryan Straughn Mp"


def test_build_merge_clusters():
    """Test building merge clusters from UnionFind."""
    uf = UnionFind()
    uf.union("node1", "node2")
    uf.union("node2", "node3")
    uf.union("node4", "node5")

    node_data = {
        "node1": {
            "id": "node1",
            "label": "Label 1",
            "degree": 5,
            "alias_count": 2,
            "is_speaker": False,
        },
        "node2": {
            "id": "node2",
            "label": "Label 2",
            "degree": 3,
            "alias_count": 1,
            "is_speaker": False,
        },
        "node3": {
            "id": "node3",
            "label": "Label 3",
            "degree": 2,
            "alias_count": 1,
            "is_speaker": False,
        },
        "node4": {
            "id": "node4",
            "label": "Label 4",
            "degree": 4,
            "alias_count": 2,
            "is_speaker": False,
        },
        "node5": {
            "id": "node5",
            "label": "Label 5",
            "degree": 1,
            "alias_count": 0,
            "is_speaker": False,
        },
    }

    merge_map = build_merge_clusters(uf, node_data)
    assert merge_map["node1"] == "node1"
    assert merge_map["node2"] == "node1"
    assert merge_map["node3"] == "node1"
    assert merge_map["node4"] == "node4"
    assert merge_map["node5"] == "node4"


def test_cluster_from_candidate_pairs():
    """Test clustering from candidate pairs."""
    candidate_pairs = [
        ("node1", "node2", 0.8),
        ("node2", "node3", 0.75),
        ("node4", "node5", 0.9),
        ("node3", "node4", 0.3),
    ]
    threshold = 0.5

    node_data = {
        "node1": {
            "id": "node1",
            "label": "Label 1",
            "degree": 5,
            "alias_count": 2,
            "is_speaker": False,
        },
        "node2": {
            "id": "node2",
            "label": "Label 2",
            "degree": 3,
            "alias_count": 1,
            "is_speaker": False,
        },
        "node3": {
            "id": "node3",
            "label": "Label 3",
            "degree": 2,
            "alias_count": 1,
            "is_speaker": False,
        },
        "node4": {
            "id": "node4",
            "label": "Label 4",
            "degree": 4,
            "alias_count": 2,
            "is_speaker": False,
        },
        "node5": {
            "id": "node5",
            "label": "Label 5",
            "degree": 1,
            "alias_count": 0,
            "is_speaker": False,
        },
    }

    merge_map = cluster_from_candidate_pairs(candidate_pairs, node_data, threshold)
    assert merge_map["node1"] == "node1"
    assert merge_map["node2"] == "node1"
    assert merge_map["node3"] == "node1"
    assert merge_map["node4"] == "node4"
    assert merge_map["node5"] == "node4"


def test_cluster_from_candidate_pairs_high_threshold():
    """Test clustering with high threshold (fewer merges)."""
    candidate_pairs = [
        ("node1", "node2", 0.95),
        ("node2", "node3", 0.75),
        ("node4", "node5", 0.9),
    ]
    threshold = 0.9

    node_data = {
        "node1": {
            "id": "node1",
            "label": "Label 1",
            "degree": 5,
            "alias_count": 2,
            "is_speaker": False,
        },
        "node2": {
            "id": "node2",
            "label": "Label 2",
            "degree": 3,
            "alias_count": 1,
            "is_speaker": False,
        },
        "node3": {
            "id": "node3",
            "label": "Label 3",
            "degree": 2,
            "alias_count": 1,
            "is_speaker": False,
        },
        "node4": {
            "id": "node4",
            "label": "Label 4",
            "degree": 4,
            "alias_count": 2,
            "is_speaker": False,
        },
        "node5": {
            "id": "node5",
            "label": "Label 5",
            "degree": 1,
            "alias_count": 0,
            "is_speaker": False,
        },
    }

    merge_map = cluster_from_candidate_pairs(candidate_pairs, node_data, threshold)
    assert merge_map["node1"] == "node1"
    assert merge_map["node2"] == "node1"
    assert merge_map["node3"] == "node3"
    assert merge_map["node4"] == "node4"
    assert merge_map["node5"] == "node4"
