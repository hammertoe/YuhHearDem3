"""KG extractor for LLM-based knowledge graph extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai.types import GenerateContentConfig
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from lib.db.postgres_client import PostgresClient
from lib.embeddings.google_client import GoogleEmbeddingClient
from lib.id_generators import generate_kg_edge_id, generate_kg_node_id, normalize_label
from lib.db.pgvector import vector_literal
from lib.knowledge_graph.window_builder import (
    ConceptWindow,
    Window,
    WindowBuilder,
)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


@dataclass
class ExtractedNode:
    """A new node to be created from LLM extraction."""

    temp_id: str
    type: str
    label: str
    aliases: list[str]


@dataclass
class ExtractedEdge:
    """An edge extracted from LLM."""

    source_ref: str
    predicate: str
    target_ref: str
    evidence: str
    utterance_ids: list[str]
    earliest_timestamp: str | None
    earliest_seconds: int | None
    confidence: float


@dataclass
class ExtractionResult:
    """Result of extraction from a single window."""

    window: Window
    nodes_new: list[ExtractedNode]
    edges: list[ExtractedEdge]
    raw_response: str
    parse_success: bool
    error: str | None = None


class KGExtractor:
    """Extract knowledge graph entities and relationships using Gemini."""

    DEFAULT_CONFIG = GenerateContentConfig(temperature=0.0)

    PREDICATES = [
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
    ]

    NODE_TYPES = [
        "foaf:Person",
        "schema:Legislation",
        "schema:Organization",
        "schema:Place",
        "skos:Concept",
    ]

    def __init__(
        self,
        postgres_client: PostgresClient,
        embedding_client: GoogleEmbeddingClient,
        model: str = DEFAULT_GEMINI_MODEL,
    ):
        self.postgres = postgres_client
        self.embedding = embedding_client
        self.window_builder = WindowBuilder(postgres_client, embedding_client)

        api_key = self._get_api_key()
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _get_api_key(self) -> str:
        """Get Google API key from environment."""
        import os

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        return api_key

    def _build_prompt(self, window: ConceptWindow, known_nodes_table: str) -> str:
        """Build prompt for extraction window."""
        predicates = ", ".join(self.PREDICATES)
        node_types = ", ".join(self.NODE_TYPES)

        prompt = f"""You are extracting knowledge graph entities and relationships from parliamentary transcripts.

TRANSCRIPT WINDOW:
{window.text}

KNOWN NODES (use these IDs when possible):
{known_nodes_table}

RULES:
1. If a node matches a Known Node, you MUST use the existing id (do not create a new node).
2. For new nodes, assign a temporary id like "n1", "n2", etc.
3. Predicate must be from this list: {predicates}
4. Node type must be from this list: {node_types} (use "skos:Concept" for abstract concepts)
5. Evidence must be a direct substring quote from the transcript window.
6. Utterance IDs must refer to the provided utterances (from utterance_id=...).
7. Return valid JSON only - no markdown, no comments.
8. Focus on substantive relationships - avoid trivial connections.
9. For discourse relationships (RESPONDS_TO, AGREES_WITH, DISAGREES_WITH, QUESTIONS), focus only on speaker-to-speaker connections with clear evidence. Generic acknowledgments should be avoided.

OUTPUT FORMAT:
{{
  "nodes_new": [
    {{"temp_id": "n1", "type": "skos:Concept", "label": "Fixed penalty regime", "aliases": ["fixed penalties"]}}
  ],
  "edges": [
    {{
      "source_ref": "speaker_s_mr_ralph_thorne_1",
      "predicate": "PROPOSES",
      "target_ref": "n1",
      "evidence": "I want today ... to offer prescriptions in relation to the Road Traffic Act...",
      "utterance_ids": ["Syxyah7QIaM:2564", "Syxyah7QIaM:2589"],
      "confidence": 0.72
    }}
  ]
}}

Extract entities and relationships from the transcript window above. Return JSON only."""
        return prompt

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API with retry logic."""
        response = self.client.models.generate_content(
            model=self.model, contents=prompt, config=self.DEFAULT_CONFIG
        )
        if response.text is None:
            raise ValueError("Gemini returned empty response")
        return response.text

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON response from Gemini, removing markdown if present."""
        response = response.strip()

        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]

        if response.endswith("```"):
            response = response[:-3]

        response = response.strip()
        return json.loads(response)

    def _parse_edges_from_llm_data(
        self,
        data: dict[str, Any],
        utterance_timestamps: dict[str, tuple[str | None, int]],
        window: Window,
    ) -> list[ExtractedEdge]:
        edges: list[ExtractedEdge] = []

        for edge_data in data.get("edges", []):
            try:
                source_ref = edge_data["source_ref"]
                predicate = edge_data["predicate"]
                target_ref = edge_data["target_ref"]
                evidence = edge_data["evidence"]
            except Exception:
                continue

            utterance_ids = edge_data.get("utterance_ids")
            if not isinstance(utterance_ids, list) or not utterance_ids:
                continue

            earliest_timestamp_str = None
            earliest_seconds = None

            for uid in utterance_ids:
                if uid in utterance_timestamps:
                    ts_str, ts_seconds = utterance_timestamps[uid]
                    if earliest_seconds is None or ts_seconds < earliest_seconds:
                        earliest_timestamp_str = ts_str
                        earliest_seconds = ts_seconds

            edges.append(
                ExtractedEdge(
                    source_ref=source_ref,
                    predicate=predicate,
                    target_ref=target_ref,
                    evidence=evidence,
                    utterance_ids=utterance_ids,
                    earliest_timestamp=earliest_timestamp_str
                    or window.earliest_timestamp,
                    earliest_seconds=earliest_seconds or window.earliest_seconds,
                    confidence=float(edge_data.get("confidence", 0.5)),
                )
            )

        return edges

    def extract_from_concept_window(
        self, window: ConceptWindow, youtube_video_id: str, top_k: int = 25
    ) -> ExtractionResult:
        """Extract knowledge graph from a concept window."""
        candidates = self.window_builder.get_candidate_nodes(
            window.text, window.speaker_ids, youtube_video_id, top_k
        )
        known_nodes_table = self.window_builder.format_known_nodes(candidates)

        utterance_timestamps = {
            u.id: (u.timestamp_str, u.seconds_since_start) for u in window.utterances
        }

        prompt = self._build_prompt(window, known_nodes_table)

        try:
            raw_response = self._call_gemini(prompt)
            data = self._parse_json_response(raw_response)

            nodes_new = []
            for node_data in data.get("nodes_new", []):
                nodes_new.append(
                    ExtractedNode(
                        temp_id=node_data["temp_id"],
                        type=node_data["type"],
                        label=node_data["label"],
                        aliases=node_data.get("aliases", []),
                    )
                )

            edges = self._parse_edges_from_llm_data(data, utterance_timestamps, window)

            return ExtractionResult(
                window=window,
                nodes_new=nodes_new,
                edges=edges,
                raw_response=raw_response,
                parse_success=True,
            )

        except Exception as e:
            return ExtractionResult(
                window=window,
                nodes_new=[],
                edges=[],
                raw_response=prompt if "prompt" in locals() else "",
                parse_success=False,
                error=str(e),
            )

    def canonicalize_and_store(
        self,
        results: list[ExtractionResult],
        youtube_video_id: str,
        kg_run_id: str,
        extractor_model: str,
    ) -> dict[str, Any]:
        """Canonicalize nodes and edges and store them in Postgres."""

        def _normalize_speaker_ref(
            ref: str, window_speaker_ids: list[str]
        ) -> str | None:
            ref = (ref or "").strip()
            if not ref:
                return None

            if ref.startswith("speaker_"):
                sid = ref.removeprefix("speaker_")
                return ref if sid in window_speaker_ids else None

            if ref.startswith("s_"):
                return f"speaker_{ref}" if ref in window_speaker_ids else None

            return ref

        temp_to_canonical = {}
        new_nodes_data = []
        new_aliases_data = []
        edges_data = []
        stats = {
            "windows_processed": len(results),
            "windows_successful": 0,
            "windows_failed": 0,
            "new_nodes": 0,
            "edges": 0,
            "links_to_known": 0,
            "edges_skipped_invalid_speaker_ref": 0,
            "edges_skipped_missing_nodes": 0,
        }

        # Ensure speaker nodes exist for all speakers observed in successful windows.
        speaker_ids_seen: set[str] = set()
        for result in results:
            if result.parse_success:
                speaker_ids_seen.update(result.window.speaker_ids)

        if speaker_ids_seen:
            speaker_rows = self.postgres.execute_query(
                """
                SELECT id, normalized_name, full_name, title
                FROM speakers
                WHERE id = ANY(%s)
                """,
                (list(speaker_ids_seen),),
            )
            speaker_meta = {
                row[0]: {
                    "normalized_name": row[1] or "",
                    "full_name": row[2] or "",
                    "title": row[3] or "",
                }
                for row in speaker_rows
            }

            speaker_nodes_data = []
            for speaker_id in speaker_ids_seen:
                meta = speaker_meta.get(speaker_id, {})
                label = (
                    meta.get("full_name") or meta.get("normalized_name") or speaker_id
                )
                aliases = []
                for candidate in (
                    meta.get("full_name"),
                    meta.get("normalized_name"),
                    meta.get("title"),
                    speaker_id,
                ):
                    if candidate:
                        aliases.append(normalize_label(candidate))

                speaker_nodes_data.append(
                    (
                        f"speaker_{speaker_id}",
                        label,
                        "foaf:Person",
                        aliases,
                    )
                )

            node_query = """
                INSERT INTO kg_nodes (id, label, type, aliases)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET label = EXCLUDED.label,
                    aliases = EXCLUDED.aliases,
                    updated_at = NOW()
            """
            self.postgres.execute_batch(node_query, speaker_nodes_data)

        for result in results:
            if not result.parse_success:
                stats["windows_failed"] += 1
                continue

            stats["windows_successful"] += 1

            for node in result.nodes_new:
                node_id = generate_kg_node_id(node.type, node.label)

                temp_to_canonical[node.temp_id] = node_id

                new_nodes_data.append(
                    (
                        node_id,
                        node.label,
                        node.type,
                        node.aliases,
                    )
                )

                for alias in node.aliases:
                    if alias:
                        new_aliases_data.append(
                            (
                                normalize_label(alias),
                                alias,
                                node_id,
                                node.type,
                                "llm",
                                None,
                            )
                        )

                stats["new_nodes"] += 1

            for edge in result.edges:
                window_speaker_ids = result.window.speaker_ids

                source_ref = _normalize_speaker_ref(edge.source_ref, window_speaker_ids)
                target_ref = _normalize_speaker_ref(edge.target_ref, window_speaker_ids)

                if source_ref is None or target_ref is None:
                    stats["edges_skipped_invalid_speaker_ref"] += 1
                    continue

                source_id = temp_to_canonical.get(source_ref, source_ref)
                target_id = temp_to_canonical.get(target_ref, target_ref)

                if not (
                    edge.source_ref.startswith("speaker_")
                    or edge.source_ref in temp_to_canonical
                ):
                    stats["links_to_known"] += 1
                if not (
                    edge.target_ref.startswith("speaker_")
                    or edge.target_ref in temp_to_canonical
                ):
                    stats["links_to_known"] += 1

                edge_id = generate_kg_edge_id(
                    source_id,
                    edge.predicate,
                    target_id,
                    youtube_video_id,
                    edge.earliest_seconds or result.window.earliest_seconds or 0,
                    edge.evidence,
                )

                edges_data.append(
                    (
                        edge_id,
                        source_id,
                        edge.predicate,
                        target_id,
                        youtube_video_id,
                        edge.earliest_timestamp,
                        edge.earliest_seconds,
                        edge.utterance_ids,
                        edge.evidence,
                        result.window.speaker_ids,
                        edge.confidence,
                        extractor_model,
                        kg_run_id,
                    )
                )

                stats["edges"] += 1

        if new_nodes_data:
            node_query = """
                INSERT INTO kg_nodes (id, label, type, aliases)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET label = EXCLUDED.label,
                    aliases = EXCLUDED.aliases,
                    updated_at = NOW()
            """
            self.postgres.execute_batch(node_query, new_nodes_data)

        if new_aliases_data:
            alias_query = """
                INSERT INTO kg_aliases (alias_norm, alias_raw, node_id, type, source, confidence)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (alias_norm) DO NOTHING
            """
            self.postgres.execute_batch(alias_query, new_aliases_data)

        if edges_data:
            # Drop edges whose endpoints do not exist; this prevents a single bad edge
            # from failing the entire run.
            referenced_node_ids = sorted(
                {src for (_eid, src, *_rest) in edges_data}
                | {tgt for (_eid, _src, _pred, tgt, *_rest) in edges_data}
            )

            existing_rows = self.postgres.execute_query(
                """
                SELECT id
                FROM kg_nodes
                WHERE id = ANY(%s)
                """,
                (referenced_node_ids,),
            )
            existing_ids = {row[0] for row in existing_rows}

            filtered_edges = [
                e for e in edges_data if e[1] in existing_ids and e[3] in existing_ids
            ]
            stats["edges_skipped_missing_nodes"] = len(edges_data) - len(filtered_edges)
            stats["edges"] = len(filtered_edges)

            edge_query = """
                INSERT INTO kg_edges (
                    id, source_id, predicate, target_id, youtube_video_id,
                    earliest_timestamp_str, earliest_seconds, utterance_ids,
                    evidence, speaker_ids, confidence, extractor_model, kg_run_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """
            if filtered_edges:
                self.postgres.execute_batch(edge_query, filtered_edges)

        # Generate embeddings for newly created nodes.
        if new_nodes_data:
            self._embed_new_nodes([nid for (nid, *_rest) in new_nodes_data])

        return stats

    def _embed_new_nodes(self, node_ids: list[str]) -> None:
        if not node_ids:
            return

        rows = self.postgres.execute_query(
            """
            SELECT id, label
            FROM kg_nodes
            WHERE id = ANY(%s) AND embedding IS NULL
            """,
            (node_ids,),
        )

        to_embed = [(row[0], row[1]) for row in rows if row[1]]
        if not to_embed:
            return

        ids = [x[0] for x in to_embed]
        texts = [x[1] for x in to_embed]
        embeddings = self.embedding.generate_embeddings_batch(
            texts, task_type="RETRIEVAL_DOCUMENT"
        )

        update_rows = [
            (vector_literal(vec), node_id) for node_id, vec in zip(ids, embeddings)
        ]
        self.postgres.execute_batch(
            """
            UPDATE kg_nodes
            SET embedding = (%s)::vector, updated_at = NOW()
            WHERE id = %s
            """,
            update_rows,
        )
