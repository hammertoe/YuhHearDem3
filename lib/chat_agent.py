"""KG conversational agent with deterministic retrieval and LLM answering."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from google import genai
from google.genai.types import GenerateContentConfig
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from lib.db.postgres_client import PostgresClient
from lib.db.pgvector import vector_literal
from lib.embeddings.google_client import GoogleEmbeddingClient
from lib.id_generators import normalize_label

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


@dataclass
class PlannerOutput:
    """Parsed output from the planner step."""

    intent: str
    entities: list[str]
    predicates: list[str]
    node_types: list[str]
    followup_requires_focus: bool


@dataclass
class Citation:
    """A citation for an assistant response."""

    utterance_id: str
    youtube_video_id: str
    seconds_since_start: int
    timestamp_str: str
    speaker_id: str
    speaker_name: str
    text: str
    video_title: str | None
    video_date: str | None


@dataclass
class UsedEdge:
    """An edge used in answering."""

    id: str
    source_id: str
    predicate: str
    predicate_raw: str | None
    target_id: str
    confidence: float | None
    evidence: str | None
    utterance_ids: list[str]


@dataclass
class ChatResponse:
    """Full response from the chat agent."""

    assistant_message: dict[str, Any]
    citations: list[Citation]
    focus_nodes: list[dict[str, Any]]
    used_edges: list[UsedEdge]
    debug: dict[str, Any] | None


class KGChatAgent:
    """Conversational agent over canonical KG + transcripts."""

    DEFAULT_CONFIG = GenerateContentConfig(temperature=0.0)

    def __init__(
        self,
        postgres_client: PostgresClient,
        embedding_client: GoogleEmbeddingClient,
        model: str = DEFAULT_GEMINI_MODEL,
    ):
        self.postgres = postgres_client
        self.embedding = embedding_client

        api_key = self._get_api_key()
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _get_api_key(self) -> str:
        import os

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        return api_key

    def create_thread(self, title: str | None = None) -> str:
        """Create a new chat thread and return its ID."""
        thread_id = str(uuid.uuid4())
        self.postgres.execute_update(
            "INSERT INTO chat_threads (id, title) VALUES (%s, %s)",
            (thread_id, title),
        )
        return thread_id

    def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        """Get thread metadata and messages."""
        rows = self.postgres.execute_query(
            """
            SELECT ct.id, ct.title, ct.created_at, ct.updated_at
            FROM chat_threads ct
            WHERE ct.id = %s
            """,
            (thread_id,),
        )
        if not rows:
            return None

        row = rows[0]
        thread = {
            "id": row[0],
            "title": row[1],
            "created_at": str(row[2]),
            "updated_at": str(row[3]),
        }

        msg_rows = self.postgres.execute_query(
            """
            SELECT id, role, content, metadata, created_at
            FROM chat_messages
            WHERE thread_id = %s
            ORDER BY created_at ASC
            """,
            (thread_id,),
        )

        messages = [
            {
                "id": r[0],
                "role": r[1],
                "content": r[2],
                "metadata": r[3],
                "created_at": str(r[4]),
            }
            for r in msg_rows
        ]
        thread["messages"] = messages

        state_rows = self.postgres.execute_query(
            "SELECT state FROM chat_thread_state WHERE thread_id = %s",
            (thread_id,),
        )
        thread["state"] = json.loads(state_rows[0][0]) if state_rows else {}

        return thread

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _call_gemini(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model, contents=prompt, config=self.DEFAULT_CONFIG
        )
        if response.text is None:
            raise ValueError("Gemini returned empty response")
        return response.text

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        return json.loads(response)

    def _planner_prompt(self, user_question: str, thread_state: dict[str, Any]) -> str:
        focus_context = ""
        if thread_state.get("focus_node_ids"):
            focus_labels = thread_state.get("focus_node_labels", [])
            focus_context = f"\nCurrent focus concepts: {', '.join(focus_labels)}\n"

        prompt = f"""You are planning retrieval for a question about parliamentary debates.

USER QUESTION:
{user_question}
{focus_context}
CONVERSATION CONTEXT:
Use only the information above.

Decide:
1. intent: one of (find_status, find_proposer, find_agreements, find_disagreements, general_query)
2. entities: main entities mentioned (lowercase, comma-separated)
3. predicates: which KG predicates might be relevant (use only these: PROPOSES, ADDRESSES, RESPONSIBLE_FOR, IMPLEMENTED_BY, CAUSES, ASSOCIATED_WITH, AMENDS, GOVERNS, AGREES_WITH, DISAGREES_WITH, RESPONDS_TO, QUESTIONS)
4. node_types: which node types might be relevant (skos:Concept, foaf:Person, schema:Legislation, schema:Organization, schema:Place)
5. followup_requires_focus: true if the question refers to "that", "it", "the proposal" etc. and needs previous focus; false otherwise

Return JSON only:
{{
  "intent": "find_status",
  "entities": ["funding", "sport", "schools"],
  "predicates": ["PROPOSES", "RESPONSIBLE_FOR"],
  "node_types": ["skos:Concept"],
  "followup_requires_focus": false
}}"""
        return prompt

    def _run_planner(
        self, user_question: str, thread_state: dict[str, Any]
    ) -> PlannerOutput | None:
        try:
            prompt = self._planner_prompt(user_question, thread_state)
            raw = self._call_gemini(prompt)
            data = self._parse_json_response(raw)
            return PlannerOutput(
                intent=data.get("intent", "general_query"),
                entities=data.get("entities", []),
                predicates=data.get("predicates", []),
                node_types=data.get("node_types", []),
                followup_requires_focus=data.get("followup_requires_focus", False),
            )
        except Exception as e:
            print(f"Planner failed: {e}")
            return None

    def _retrieve_candidate_nodes(
        self,
        question: str,
        thread_state: dict[str, Any],
        top_k: int = 25,
    ) -> list[dict[str, Any]]:
        """Retrieve candidate nodes via vector search and alias matching."""
        candidates = []

        query_embedding = self.embedding.generate_query_embedding(question)

        vector_query = """
            SELECT id, type, label, aliases, embedding <=> %s AS distance
            FROM kg_nodes
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s ASC
            LIMIT %s
        """
        rows = self.postgres.execute_query(
            vector_query,
            (vector_literal(query_embedding), vector_literal(query_embedding), top_k),
        )
        for row in rows:
            candidates.append(
                {
                    "id": row[0],
                    "type": row[1],
                    "label": row[2],
                    "aliases": row[3],
                    "distance": row[4],
                    "source": "vector",
                }
            )

        alias_query = """
            SELECT kn.id, kn.type, kn.label, kn.aliases
            FROM kg_aliases ka
            JOIN kg_nodes kn ON ka.node_id = kn.id
            WHERE ka.alias_norm = %s
            LIMIT 5
        """
        rows = self.postgres.execute_query(alias_query, (normalize_label(question),))
        for row in rows:
            candidates.append(
                {
                    "id": row[0],
                    "type": row[1],
                    "label": row[2],
                    "aliases": row[3],
                    "distance": 0.0,
                    "source": "alias",
                }
            )

        unique = {}
        for c in candidates:
            if c["id"] not in unique:
                unique[c["id"]] = c
        return list(unique.values())

    def _retrieve_edges_for_nodes(
        self, node_ids: list[str], limit: int = 50
    ) -> list[dict[str, Any]]:
        if not node_ids:
            return []

        placeholders = ",".join(["%s"] * len(node_ids))
        query = f"""
            SELECT id, source_id, predicate, predicate_raw, target_id,
                   youtube_video_id, earliest_timestamp_str, earliest_seconds,
                   utterance_ids, evidence, speaker_ids, confidence
            FROM kg_edges
            WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
            ORDER BY confidence DESC NULLS LAST, earliest_seconds ASC
            LIMIT %s
        """
        params = tuple(node_ids + node_ids + [limit])
        rows = self.postgres.execute_query(query, params)

        return [
            {
                "id": row[0],
                "source_id": row[1],
                "predicate": row[2],
                "predicate_raw": row[3],
                "target_id": row[4],
                "youtube_video_id": row[5],
                "earliest_timestamp_str": row[6],
                "earliest_seconds": row[7],
                "utterance_ids": row[8] or [],
                "evidence": row[9],
                "speaker_ids": row[10] or [],
                "confidence": row[11],
            }
            for row in rows
        ]

    def _retrieve_sentences_for_utterances(
        self, utterance_ids: list[str]
    ) -> list[dict[str, Any]]:
        if not utterance_ids:
            return []

        placeholders = ",".join(["%s"] * len(utterance_ids))
        query = f"""
            SELECT s.id, s.text, s.seconds_since_start, s.timestamp_str,
                   s.youtube_video_id, s.video_date, s.video_title, s.speaker_id,
                   sp.full_name, sp.normalized_name
            FROM sentences s
            LEFT JOIN speakers sp ON s.speaker_id = sp.id
            WHERE s.id IN ({placeholders})
        """
        rows = self.postgres.execute_query(query, tuple(utterance_ids))

        return [
            {
                "id": row[0],
                "text": row[1],
                "seconds_since_start": row[2],
                "timestamp_str": row[3],
                "youtube_video_id": row[4],
                "video_date": str(row[5]) if row[5] else None,
                "video_title": row[6] or "",
                "speaker_id": row[7],
                "full_name": row[8],
                "normalized_name": row[9] or row[7],
            }
            for row in rows
        ]

    def _answerer_prompt(
        self,
        user_question: str,
        thread_state: dict[str, Any],
        planner_output: PlannerOutput | None,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        sentences: list[dict[str, Any]],
    ) -> str:
        focus_context = ""
        if thread_state.get("focus_node_labels"):
            focus_context = f"\nCurrent focus concepts: {', '.join(thread_state['focus_node_labels'])}\n"

        nodes_table = "\n".join(
            f"- {n['id']} ({n['type']}): {n['label']}" for n in nodes[:15]
        )
        edges_table = "\n".join(
            f"- {e['source_id']} {e['predicate']} {e['target_id']}"
            f" (confidence: {e['confidence']})"
            for e in edges[:20]
        )
        sentences_context = "\n".join(
            f"[{s['id']} @ {s['timestamp_str']} {s['speaker_id']}]: {s['text']}"
            for s in sentences[:10]
        )

        prompt = f"""You are answering a question about parliamentary debates using extracted knowledge graph facts and transcript evidence.

USER QUESTION:
{user_question}
{focus_context}
RETRIEVED CONTEXT:
Relevant concepts:
{nodes_table or "(none)"}

Relevant relationships:
{edges_table or "(none)"}

Transcript evidence:
{sentences_context or "(none)"}

INSTRUCTIONS:
1. Answer the question using ONLY the context above.
2. If the context doesn't contain enough information, say so clearly.
3. Cite evidence by utterance_id (e.g., "[Syxyah7QIaM:1234]").
4. Do NOT invent facts not in the context.
5. For follow-ups about "that", use the current focus concepts.
6. Return JSON only.

OUTPUT FORMAT:
{{
  "answer": "Your answer here with [utterance_id] citations.",
  "citations": ["Syxyah7QIaM:1234", "Syxyah7QIaM:5678"],
  "focus_node_ids": ["kg_abc123", "speaker_foo_1"]
}}

Answer the question. Return JSON only."""
        return prompt

    def _run_answerer(
        self,
        user_question: str,
        thread_state: dict[str, Any],
        planner_output: PlannerOutput | None,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        sentences: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        try:
            prompt = self._answerer_prompt(
                user_question,
                thread_state,
                planner_output,
                nodes,
                edges,
                sentences,
            )
            raw = self._call_gemini(prompt)
            data = self._parse_json_response(raw)
            return data
        except Exception as e:
            print(f"Answerer failed: {e}")
            return None

    def process_message(self, thread_id: str, user_content: str) -> ChatResponse:
        """Process a user message and return assistant response."""
        thread = self.get_thread(thread_id)
        if thread is None:
            raise ValueError(f"Thread {thread_id} not found")

        state = thread.get("state", {})

        user_message_id = str(uuid.uuid4())
        self.postgres.execute_update(
            "INSERT INTO chat_messages (id, thread_id, role, content, created_at) VALUES (%s, %s, %s, %s, %s)",
            (user_message_id, thread_id, "user", user_content, datetime.now()),
        )

        planner_output = self._run_planner(user_content, state)

        if planner_output and planner_output.followup_requires_focus:
            if state.get("focus_node_ids"):
                pass
            else:
                pass

        candidates = self._retrieve_candidate_nodes(user_content, state, top_k=25)

        if planner_output:
            candidates = [
                c
                for c in candidates
                if not planner_output.node_types
                or c["type"] in planner_output.node_types
            ]

        node_ids = [c["id"] for c in candidates]
        edges = self._retrieve_edges_for_nodes(node_ids, limit=50)

        utterance_ids = []
        for e in edges:
            utterance_ids.extend(e.get("utterance_ids", []))
        unique_utterance_ids = list(set(utterance_ids))[:30]
        sentences = self._retrieve_sentences_for_utterances(unique_utterance_ids)

        if len(candidates) < 5 or len(edges) < 3:
            from lib.advanced_search_features import AdvancedSearchFeatures

            adv = AdvancedSearchFeatures(
                postgres=self.postgres,
                memgraph=None,
                embedding_client=self.embedding,
            )
            fallback_results = adv.temporal_search(
                query=user_content,
                date_start=None,
                date_end=None,
                speaker_id=None,
                entity_type=None,
                limit=20,
            )
            for r in fallback_results:
                utterance_id = r.get("id")
                if utterance_id and utterance_id not in unique_utterance_ids:
                    unique_utterance_ids.append(utterance_id)
                    sentences.append(
                        {
                            "id": utterance_id,
                            "text": r["text"],
                            "seconds_since_start": r["seconds_since_start"],
                            "timestamp_str": r["timestamp_str"],
                            "youtube_video_id": r["video_id"],
                            "video_date": r.get("video_date", ""),
                            "video_title": r.get("video_title", ""),
                            "speaker_id": r["speaker_id"],
                            "full_name": r.get("speaker_name", ""),
                            "normalized_name": r.get("speaker_name", ""),
                        }
                    )

        answer_data = self._run_answerer(
            user_content,
            state,
            planner_output,
            candidates,
            edges,
            sentences,
        )

        if not answer_data:
            answer_data = {
                "answer": "Sorry, I couldn't generate an answer. Please try rephrasing your question.",
                "citations": [],
                "focus_node_ids": [],
            }

        assistant_message_id = str(uuid.uuid4())
        metadata = {
            "citations": answer_data.get("citations", []),
            "focus_node_ids": answer_data.get("focus_node_ids", []),
            "retrieval_stats": {
                "candidates": len(candidates),
                "edges": len(edges),
                "sentences": len(sentences),
            },
        }
        self.postgres.execute_update(
            """
            INSERT INTO chat_messages (id, thread_id, role, content, metadata, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                assistant_message_id,
                thread_id,
                "assistant",
                answer_data.get("answer", ""),
                json.dumps(metadata),
                datetime.now(),
            ),
        )

        self.postgres.execute_update(
            """
            INSERT INTO chat_thread_state (thread_id, state, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (thread_id) DO UPDATE
            SET state = EXCLUDED.state, updated_at = EXCLUDED.updated_at
            """,
            (
                thread_id,
                json.dumps(
                    {
                        "focus_node_ids": answer_data.get("focus_node_ids", []),
                        "focus_node_labels": [
                            c["label"]
                            for c in candidates
                            if c["id"] in answer_data.get("focus_node_ids", [])
                        ],
                        "last_question": user_content,
                    }
                ),
                datetime.now(),
            ),
        )

        assistant_message = {
            "id": assistant_message_id,
            "role": "assistant",
            "content": answer_data.get("answer", ""),
            "created_at": str(datetime.now()),
        }

        citation_objects = []
        for uid in answer_data.get("citations", []):
            sent = next((s for s in sentences if s["id"] == uid), None)
            if sent:
                citation_objects.append(
                    Citation(
                        utterance_id=sent["id"],
                        youtube_video_id=sent["youtube_video_id"],
                        seconds_since_start=sent["seconds_since_start"],
                        timestamp_str=sent["timestamp_str"],
                        speaker_id=sent["speaker_id"],
                        speaker_name=sent["normalized_name"] or sent["full_name"],
                        text=sent["text"],
                        video_title=sent.get("video_title"),
                        video_date=sent.get("video_date"),
                    )
                )

        used_edge_objects = [
            UsedEdge(
                id=e["id"],
                source_id=e["source_id"],
                predicate=e["predicate"],
                predicate_raw=e.get("predicate_raw"),
                target_id=e["target_id"],
                confidence=e.get("confidence"),
                evidence=e.get("evidence"),
                utterance_ids=e.get("utterance_ids", []),
            )
            for e in edges
            if any(
                uid in e.get("utterance_ids", [])
                for uid in answer_data.get("citations", [])
            )
        ]

        focus_nodes_objects = [
            {"id": c["id"], "label": c["label"], "type": c["type"]}
            for c in candidates
            if c["id"] in answer_data.get("focus_node_ids", [])
        ]

        debug = {
            "planner": planner_output.__dict__ if planner_output else None,
            "retrieval": {
                "candidates": len(candidates),
                "edges": len(edges),
                "sentences": len(sentences),
                "fallback_used": len(candidates) < 5 or len(edges) < 3,
            },
        }

        return ChatResponse(
            assistant_message=assistant_message,
            citations=citation_objects,
            focus_nodes=focus_nodes_objects,
            used_edges=used_edge_objects,
            debug=debug,
        )
