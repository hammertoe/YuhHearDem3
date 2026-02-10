from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


def _sanitize_rel_type(value: str) -> str:
    rel = re.sub(r"[^A-Za-z0-9_]", "_", (value or "").upper()).strip("_")
    if not rel:
        rel = "RELATED_TO"
    if rel[0].isdigit():
        rel = "REL_" + rel
    return rel


def _label_for_node_type(node_type: str) -> str:
    mapping = {
        "ORG": "Organization",
        "PERSON": "Person",
        "GPE": "Location",
        "LAW": "Law",
        "DATE": "Date",
        "SPEAKER": "Speaker",
        "LEGISLATION": "Legislation",
    }
    return mapping.get(node_type, "Entity")


@dataclass(frozen=True)
class KnowledgeGraphIngestStats:
    nodes: int
    edges: int


class KnowledgeGraphMemgraphIngestor:
    def __init__(self, memgraph: Any):
        self.memgraph = memgraph

    def delete_run(self, run_id: str) -> int:
        result = self.memgraph.execute_update(
            """
            MATCH ()-[r]->()
            WHERE r.kg_run_id = $run_id
            DELETE r
            """,
            {"run_id": run_id},
        )
        return int(result)

    def ingest(
        self, kg: dict[str, Any], *, run_id: str, youtube_video_id: str | None = None
    ) -> KnowledgeGraphIngestStats:
        nodes = kg.get("nodes") or []
        edges = kg.get("edges") or []

        for node in nodes:
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue

            node_type = str(node.get("type", "Entity"))
            label = _label_for_node_type(node_type)
            props: dict[str, Any] = {
                "id": node_id,
                "text": node.get("text", ""),
                "type": node_type,
                "count": int(node.get("count", 1) or 1),
                "speaker_context": node.get("speaker_context", []) or [],
                "timestamps": node.get("timestamps", []) or [],
                "source": "knowledge_graph",
            }
            if youtube_video_id:
                props["youtube_video_id"] = youtube_video_id

            # Optional date normalization fields.
            if "resolved_date" in node:
                props["resolved_date"] = node.get("resolved_date")
            if "is_relative" in node:
                props["is_relative"] = bool(node.get("is_relative"))

            self.memgraph.merge_entity(label, "id", props)

        seen_edges: set[tuple[str, str, str, str, str]] = set()
        for edge in edges:
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            if not source or not target:
                continue

            rel_type = _sanitize_rel_type(str(edge.get("relationship", "RELATED_TO")))
            ts = str(edge.get("timestamp", ""))
            speaker_id = str(edge.get("speaker_id", ""))
            key = (source, target, rel_type, ts, speaker_id)
            if key in seen_edges:
                continue
            seen_edges.add(key)

            rel_props: dict[str, Any] = {
                "context": edge.get("context", ""),
                "speaker_id": edge.get("speaker_id", ""),
                "timestamp": edge.get("timestamp", ""),
                "kg_run_id": run_id,
                "source": "knowledge_graph",
            }
            if youtube_video_id:
                rel_props["youtube_video_id"] = youtube_video_id

            self.memgraph.merge_relationship(
                source,
                target,
                rel_type,
                match_properties=None,
                set_properties=rel_props,
            )

        return KnowledgeGraphIngestStats(nodes=len(nodes), edges=len(edges))
