"""KG cleanup clustering using union-find."""

from dataclasses import dataclass
from typing import Any


@dataclass
class NodeInfo:
    """Node information for canonical node selection."""

    id: str
    label: str
    degree: int
    alias_count: int
    is_speaker: bool


class UnionFind:
    """Union-Find (Disjoint Set) data structure for node clustering."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self._size: dict[str, int] = {}

    def find(self, x: str) -> str:
        """Find root of x with path compression."""
        if x not in self.parent:
            self.parent[x] = x
            self._size[x] = 1
            return x

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        """Union sets containing x and y with union by size."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        if self._size[root_x] < self._size[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x
        self._size[root_x] += self._size[root_y]

    def size(self, x: str) -> int:
        """Get size of cluster containing x."""
        return self._size[self.find(x)]

    def get_clusters(self) -> list[set[str]]:
        """Get all clusters as sets of node IDs."""
        root_to_cluster: dict[str, set[str]] = {}
        for node in self.parent:
            root = self.find(node)
            if root not in root_to_cluster:
                root_to_cluster[root] = set()
            root_to_cluster[root].add(node)
        return list(root_to_cluster.values())


def find_canonical_node(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Select canonical node from a cluster using priority rules.

    Priority:
    1. Prefer speaker_* nodes for Person type
    2. Higher degree
    3. Richer alias/evidence footprint (alias_count)
    4. Longer label (most descriptive)
    5. Deterministic tie-break (id lexical)
    """
    if not nodes:
        raise ValueError("Cannot select canonical node from empty list")

    def sort_key(node: dict[str, Any]) -> tuple:
        is_speaker = node.get("is_speaker", False)
        degree = node.get("degree", 0)
        alias_count = node.get("alias_count", 0)
        label = node.get("label", "")
        node_id = node.get("id", "")
        return (
            -1 if is_speaker else 0,
            -degree,
            -alias_count,
            -len(label),
            node_id,
        )

    return sorted(nodes, key=sort_key)[0]


def build_merge_clusters(uf: UnionFind, node_data: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Build merge map from UnionFind structure.

    Returns a dictionary mapping each node to its canonical node.
    Nodes not in any cluster map to themselves.
    """
    merge_map: dict[str, str] = {}
    clusters = uf.get_clusters()
    clustered_nodes: set[str] = set()

    for cluster in clusters:
        if not cluster:
            continue

        cluster_nodes = [node_data[nid] for nid in cluster if nid in node_data]
        if not cluster_nodes:
            continue

        canonical = find_canonical_node(cluster_nodes)
        canonical_id = canonical["id"]

        for node_id in cluster:
            merge_map[node_id] = canonical_id
            clustered_nodes.add(node_id)

    for node_id in node_data:
        if node_id not in clustered_nodes:
            merge_map[node_id] = node_id

    return merge_map


def cluster_from_candidate_pairs(
    candidate_pairs: list[tuple[str, str, float]],
    node_data: dict[str, dict[str, Any]],
    threshold: float,
) -> dict[str, str]:
    """Build merge clusters from candidate pairs above threshold.

    Args:
        candidate_pairs: List of (node_id_1, node_id_2, similarity_score)
        node_data: Dictionary of node_id -> node_info
        threshold: Minimum similarity score to consider merging

    Returns:
        Dictionary mapping each node to its canonical node
    """
    uf = UnionFind()

    for node_id_1, node_id_2, score in candidate_pairs:
        if score >= threshold:
            uf.union(node_id_1, node_id_2)

    return build_merge_clusters(uf, node_data)
