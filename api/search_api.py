"""Hybrid search API combining vector search and graph traversal."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from lib.db.postgres_client import PostgresClient
from lib.db.chat_schema import ensure_chat_schema
from lib.db.memgraph_client import MemgraphClient
from lib.chat_agent_v2 import KGChatAgentV2
from lib.kg_agent_loop import KGAgentLoop
from lib.kg_hybrid_graph_rag import kg_hybrid_graph_rag
from lib.advanced_search_features import AdvancedSearchFeatures
from lib.embeddings.google_client import GoogleEmbeddingClient
from lib.utils.config import config

app = FastAPI(title="Parliamentary Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_postgres() -> PostgresClient:
    assert postgres is not None
    return postgres


def _get_memgraph() -> MemgraphClient:
    assert memgraph is not None
    return memgraph


def _get_embedding_client() -> GoogleEmbeddingClient:
    assert embedding_client is not None
    return embedding_client


def _get_advanced_search() -> AdvancedSearchFeatures:
    assert advanced_search is not None
    return advanced_search


def _get_chat_agent() -> KGChatAgentV2:
    assert chat_agent is not None
    return chat_agent


@app.on_event("startup")
def _startup() -> None:
    global postgres, memgraph, embedding_client, advanced_search, chat_agent
    postgres = PostgresClient()
    try:
        # Try to ensure chat schema, but don't fail if it already exists with different structure
        ensure_chat_schema(postgres)
    except Exception:
        pass  # Schema may already exist with different structure
    try:
        memgraph = MemgraphClient()
    except Exception as e:
        print(f"Warning: Could not connect to Memgraph: {e}")
        memgraph = None
    embedding_client = GoogleEmbeddingClient()
    advanced_search = AdvancedSearchFeatures(
        postgres=postgres,
        memgraph=memgraph,
        embedding_client=embedding_client,
    )
    chat_agent = KGChatAgentV2(
        postgres_client=postgres,
        embedding_client=embedding_client,
        model=getattr(config, "gemini_model", "gemini-2.5-flash"),
        enable_thinking=getattr(config, "enable_thinking", False),
    )


class SearchRequest(BaseModel):
    query: str
    limit: int = 20
    alpha: float = 0.6


class SearchResult(BaseModel):
    id: str
    text: str
    timestamp_str: str
    seconds_since_start: int
    video_id: str
    video_title: str
    video_date: str
    speaker_id: str
    speaker_name: str
    paragraph_id: str
    score: float
    search_type: str
    provenance: str | None = None


class TemporalSearchRequest(BaseModel):
    query: str
    limit: int = 20
    alpha: float | None = 0.6
    start_date: str | None = None
    end_date: str | None = None
    speaker_id: str | None = None
    entity_type: str | None = None


class TrendResult(BaseModel):
    entity_id: str | None
    trends: list[dict[str, Any]]
    summary: dict[str, Any]
    moving_average: list[dict[str, Any]]


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    properties: dict[str, Any]


class GraphEdge(BaseModel):
    from_node: str = Field(serialization_alias="from")
    to_node: str = Field(serialization_alias="to")
    label: str
    properties: dict[str, Any]


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class Speaker(BaseModel):
    speaker_id: str
    normalized_name: str
    full_name: str
    title: str
    position: str
    role_in_video: str
    first_appearance: str
    total_appearances: int


class SpeakerStatsResponse(BaseModel):
    speaker_id: str
    normalized_name: str
    full_name: str
    title: str
    position: str
    role_in_video: str
    first_appearance: str
    total_appearances: int
    recent_contributions: list[dict[str, Any]]


class SpeakerVideoRole(BaseModel):
    role_label: str
    role_kind: str
    source: str
    source_id: str
    confidence: float | None = None
    evidence: str | None = None


class CreateThreadResponse(BaseModel):
    thread_id: str
    title: str | None
    created_at: str


class ThreadMessage(BaseModel):
    id: str
    role: str
    content: str
    metadata: dict[str, Any] | None
    created_at: str


class GetThreadResponse(BaseModel):
    id: str
    title: str | None
    created_at: str
    updated_at: str
    state: dict[str, Any]
    messages: list[ThreadMessage]


class ChatCitation(BaseModel):
    utterance_id: str
    youtube_video_id: str
    seconds_since_start: int
    timestamp_str: str
    speaker_id: str
    speaker_name: str
    text: str
    video_title: str | None
    video_date: str | None
    youtube_url: str | None = None


class ChatSource(BaseModel):
    utterance_id: str
    youtube_video_id: str
    youtube_url: str
    seconds_since_start: int
    timestamp_str: str
    speaker_id: str
    speaker_name: str
    speaker_title: str | None = None
    text: str
    video_title: str | None
    video_date: str | None


class ChatFocusNode(BaseModel):
    id: str
    label: str
    type: str


class ChatUsedEdge(BaseModel):
    id: str
    source_id: str
    predicate: str
    predicate_raw: str | None
    target_id: str
    confidence: float | None
    evidence: str | None
    utterance_ids: list[str]


class ChatMessageRequest(BaseModel):
    content: str


class ChatMessageResponse(BaseModel):
    thread_id: str
    assistant_message: ThreadMessage
    citations: list[ChatCitation] = []
    focus_nodes: list[ChatFocusNode] = []
    used_edges: list[ChatUsedEdge] = []
    sources: list[ChatSource] = []
    focus_node_ids: list[str] = []
    followup_questions: list[str] = []
    debug: dict[str, Any] | None = None


postgres: PostgresClient | None = None
memgraph: MemgraphClient | None = None
embedding_client: GoogleEmbeddingClient | None = None
advanced_search: AdvancedSearchFeatures | None = None
chat_agent: KGChatAgentV2 | None = None


def _get_postgres() -> PostgresClient:
    assert postgres is not None
    return postgres


def _get_memgraph() -> MemgraphClient:
    assert memgraph is not None
    return memgraph


def _get_embedding_client() -> GoogleEmbeddingClient:
    assert embedding_client is not None
    return embedding_client


def _get_advanced_search() -> AdvancedSearchFeatures:
    assert advanced_search is not None
    return advanced_search


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def vector_search_entities(query_embedding: list[float], limit: int) -> list[dict[str, Any]]:
    """Search entities by vector similarity."""
    results = _get_postgres().execute_query(
        """
        SELECT id, text, type,
               embedding <=> (%s)::vector as distance
        FROM entities
        ORDER BY embedding <=> (%s)::vector
        LIMIT %s
    """,
        (_vector_literal(query_embedding), _vector_literal(query_embedding), limit),
    )

    return [{"id": row[0], "text": row[1], "type": row[2], "distance": row[3]} for row in results]


def vector_search_paragraphs(query_embedding: list[float], limit: int) -> list[dict[str, Any]]:
    """Search paragraphs by vector similarity."""
    results = _get_postgres().execute_query(
        """
        SELECT id, text, speaker_id, start_seconds,
               embedding <=> (%s)::vector as distance
        FROM paragraphs
        ORDER BY embedding <=> (%s)::vector
        LIMIT %s
    """,
        (_vector_literal(query_embedding), _vector_literal(query_embedding), limit),
    )

    return [
        {
            "id": row[0],
            "text": row[1],
            "speaker_id": row[2],
            "start_seconds": row[3],
            "distance": row[4],
        }
        for row in results
    ]


def bm25_search_sentences(query: str, limit: int) -> list[dict[str, Any]]:
    """Search sentences using BM25 full-text search."""
    results = _get_postgres().execute_query(
        """
        SELECT
            s.id,
            s.text,
            s.seconds_since_start,
            s.timestamp_str,
            s.youtube_video_id,
            s.video_title,
            s.video_date,
            s.speaker_id,
            COALESCE(sp.full_name, sp.normalized_name, s.speaker_id) as speaker_name,
            s.paragraph_id,
            ts_rank(s.tsv, plainto_tsquery('english', %s)) as rank
        FROM sentences s
        LEFT JOIN speakers sp ON s.speaker_id = sp.id
        WHERE s.tsv @@ plainto_tsquery('english', %s)
        ORDER BY rank DESC
        LIMIT %s
    """,
        (query, query, limit),
    )

    return [
        {
            "id": row[0],
            "text": row[1],
            "seconds_since_start": row[2],
            "timestamp_str": row[3],
            "video_id": row[4],
            "video_title": row[5] or "",
            "video_date": str(row[6]) if row[6] is not None else "",
            "speaker_id": row[7],
            "speaker_name": row[8],
            "paragraph_id": row[9] or "",
            "score": float(row[10] or 0.0),
            "search_type": "bm25",
        }
        for row in results
    ]


def graph_expand_entities(
    entity_ids: list[str], max_depth: int = 2
) -> dict[str, list[dict[str, Any]]]:
    """Expand entities using graph traversal."""
    expanded = {}

    for entity_id in entity_ids:
        results = _get_memgraph().execute_query(
            """
            MATCH (e:Entity {id: $id})-[*1..2]-(related)
            RETURN DISTINCT e.id as entity_id,
                   related.id as related_id,
                   related.text as related_text
            LIMIT 50
        """,
            {"id": entity_id},
        )

        related_entities = [
            {"id": row["related_id"], "text": row["related_text"]} for row in results
        ]

        expanded[entity_id] = related_entities

    return expanded


def retrieve_sentences_for_entities(
    entity_ids: list[str], limit_per_entity: int = 10
) -> list[dict[str, Any]]:
    """Retrieve sentences that mention entities."""
    if not entity_ids:
        return []

    placeholders = ",".join(["%s"] * len(entity_ids))
    results = _get_postgres().execute_query(
        f"""
        SELECT s.id, s.text, s.seconds_since_start, s.timestamp_str,
               s.youtube_video_id, s.video_title, s.video_date,
               s.speaker_id, s.paragraph_id,
               COALESCE(sp.full_name, sp.normalized_name, s.speaker_id) as speaker_name,
               se.entity_id, se.entity_type, se.relationship_type
        FROM sentences s
        JOIN sentence_entities se ON s.id = se.sentence_id
        LEFT JOIN speakers sp ON s.speaker_id = sp.id
        WHERE se.entity_id IN ({placeholders})
        ORDER BY s.seconds_since_start
        LIMIT {limit_per_entity * len(entity_ids)}
    """,
        tuple(entity_ids),
    )

    return [
        {
            "id": row[0],
            "text": row[1],
            "seconds_since_start": row[2],
            "timestamp_str": row[3],
            "video_id": row[4],
            "video_title": row[5] or "",
            "video_date": str(row[6]) if row[6] is not None else "",
            "speaker_id": row[7],
            "paragraph_id": row[8],
            "speaker_name": row[9],
            "entity_id": row[10],
            "entity_type": row[11],
            "relationship_type": row[12],
        }
        for row in results
    ]


def retrieve_sentences_for_paragraphs(paragraph_ids: list[str]) -> list[dict[str, Any]]:
    """Retrieve sentences that belong to paragraphs."""
    if not paragraph_ids:
        return []

    placeholders = ",".join(["%s"] * len(paragraph_ids))
    results = _get_postgres().execute_query(
        f"""
        SELECT s.id, s.text, s.seconds_since_start, s.timestamp_str,
               s.youtube_video_id, s.video_title, s.video_date,
               s.speaker_id, s.paragraph_id, s.sentence_order,
               COALESCE(sp.full_name, sp.normalized_name, s.speaker_id) as speaker_name
        FROM sentences s
        JOIN speakers sp ON s.speaker_id = sp.id
        WHERE s.paragraph_id IN ({placeholders})
        ORDER BY s.paragraph_id, s.sentence_order
    """,
        tuple(paragraph_ids),
    )

    return [
        {
            "id": row[0],
            "text": row[1],
            "seconds_since_start": row[2],
            "timestamp_str": row[3],
            "video_id": row[4],
            "video_title": row[5] or "",
            "video_date": str(row[6]) if row[6] is not None else "",
            "speaker_id": row[7],
            "paragraph_id": row[8],
            "speaker_name": row[10],
        }
        for row in results
    ]


@app.post("/search", response_model=list[SearchResult])
async def search(request: SearchRequest):
    """Hybrid search combining entity + paragraph vector search."""
    try:
        try:
            query_embedding = _get_embedding_client().generate_query_embedding(request.query)
        except Exception as e:
            print(f"⚠️ Embeddings unavailable; falling back to BM25 only: {e}")
            return [SearchResult(**r) for r in bm25_search_sentences(request.query, request.limit)]

        phase1_entities = vector_search_entities(query_embedding, 10)
        phase1_paragraphs = vector_search_paragraphs(query_embedding, 10)

        entity_ids = [e["id"] for e in phase1_entities]
        paragraph_ids = [p["id"] for p in phase1_paragraphs]

        # TODO: Implement graph expansion results usage in re-ranking
        # phase2_expanded = graph_expand_entities(entity_ids)  # noqa: F841

        phase3_sentences_from_entities = retrieve_sentences_for_entities(entity_ids)
        phase3_sentences_from_paragraphs = retrieve_sentences_for_paragraphs(paragraph_ids)

        all_sentences = phase3_sentences_from_entities + phase3_sentences_from_paragraphs

        sentence_map = {s["id"]: s for s in all_sentences}
        unique_results = list(sentence_map.values())

        scored: list[dict[str, Any]] = []
        for result in unique_results:
            distances = [
                e["distance"] for e in phase1_entities if e["id"] == result.get("entity_id")
            ]
            vector_score = 1.0 - min(distances) if distances else 0.5
            result["score"] = float(vector_score)
            scored.append(result)

        scored.sort(key=lambda x: x["score"], reverse=True)

        return [
            SearchResult(
                id=r["id"],
                text=r["text"],
                timestamp_str=r["timestamp_str"],
                seconds_since_start=r["seconds_since_start"],
                video_id=r["video_id"],
                video_title=r.get("video_title", ""),
                video_date=r.get("video_date", ""),
                speaker_id=r["speaker_id"],
                speaker_name=r.get("speaker_name", r["speaker_id"]),
                paragraph_id=r.get("paragraph_id", ""),
                score=float(r["score"]),
                search_type="hybrid",
            )
            for r in scored[: request.limit]
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/temporal", response_model=list[SearchResult])
async def temporal_search(request: TemporalSearchRequest):
    """Temporal search with filters."""
    try:
        try:
            results = _get_advanced_search().temporal_search(
                request.query,
                request.start_date,
                request.end_date,
                request.speaker_id,
                request.entity_type,
                request.limit,
            )
            return [SearchResult(**r) for r in results]
        except Exception as e:
            # If embeddings are not available, fall back to BM25 filtered by dates/speaker.
            print(f"⚠️ Temporal embeddings unavailable; using BM25 fallback: {e}")

            where = ["s.tsv @@ plainto_tsquery('english', %s)"]
            params: list[Any] = [request.query]

            if request.start_date:
                where.append("s.video_date >= to_date(%s, 'YYYY-MM-DD')")
                params.append(request.start_date)
            if request.end_date:
                where.append("s.video_date <= to_date(%s, 'YYYY-MM-DD')")
                params.append(request.end_date)
            if request.speaker_id:
                where.append("s.speaker_id = %s")
                params.append(request.speaker_id)

            sql = f"""
                SELECT
                    s.id,
                    s.text,
                    s.timestamp_str,
                    s.seconds_since_start,
                    s.youtube_video_id,
                    s.video_title,
                    s.video_date,
                    s.speaker_id,
                    COALESCE(sp.full_name, sp.normalized_name, s.speaker_id) as speaker_name,
                    s.paragraph_id,
                    ts_rank(s.tsv, plainto_tsquery('english', %s)) as rank
                FROM sentences s
                LEFT JOIN speakers sp ON s.speaker_id = sp.id
                WHERE {" AND ".join(where)}
                ORDER BY rank DESC
                LIMIT %s
            """
            # rank query param must be first; reuse query as last before limit
            rank_query = request.query
            final_params = [rank_query, *params, request.limit]
            rows = _get_postgres().execute_query(sql, tuple(final_params))

            return [
                SearchResult(
                    id=row[0],
                    text=row[1],
                    timestamp_str=row[2],
                    seconds_since_start=row[3],
                    video_id=row[4],
                    video_title=row[5] or "",
                    video_date=str(row[6]) if row[6] is not None else "",
                    speaker_id=row[7],
                    speaker_name=row[8],
                    paragraph_id=row[9] or "",
                    score=float(row[10] or 0.0),
                    search_type="bm25_temporal",
                )
                for row in rows
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/trends")
async def get_trends(
    entity_id: str | None = None, days: int = 30, window_size: int = 7
) -> TrendResult:
    """Get trend analysis for entity."""
    try:
        result = _get_advanced_search().trend_analysis(
            entity_id,
            time_window_days=days,
            limit=100,
        )

        return TrendResult(
            entity_id=result["entity_id"],
            trends=result["trends"],
            summary=result["summary"],
            moving_average=result["moving_average"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph")
async def get_graph(entity_id: str, max_depth: int = 2) -> dict[str, Any]:
    """Get graph data for entity."""
    try:
        results = _get_advanced_search().multi_hop_query(
            entity_id,
            hops=max_depth,
            max_results=50,
        )

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        seen_nodes: set[str] = set()

        for r in results:
            start_id = r.get("start_entity")
            related_id = r.get("related_entity")

            if start_id and start_id not in seen_nodes:
                nodes.append(
                    GraphNode(
                        id=start_id,
                        label=start_id,
                        type=r.get("start_type", "Unknown"),
                        properties={},
                    )
                )
                seen_nodes.add(start_id)

            if related_id and related_id not in seen_nodes:
                nodes.append(
                    GraphNode(
                        id=related_id,
                        label=related_id,
                        type=r.get("related_type", "Unknown"),
                        properties={},
                    )
                )
                seen_nodes.add(related_id)

            if start_id and related_id:
                edges.append(
                    GraphEdge(
                        from_node=start_id,
                        to_node=related_id,
                        label=r.get("relationship_type", ""),
                        properties={},
                    )
                )

        return GraphResponse(nodes=nodes, edges=edges).model_dump(by_alias=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/speakers")
async def get_speakers() -> list[Speaker]:
    """Get list of all speakers."""
    try:
        results = _get_postgres().execute_query(
            """
            SELECT
                sp.id,
                sp.normalized_name,
                sp.full_name,
                sp.title,
                sp.position,
                'member' as role_in_video,
                MIN(s.timestamp_str) as first_appearance,
                COUNT(DISTINCT s.youtube_video_id) as total_appearances
            FROM speakers sp
            LEFT JOIN sentences s ON s.speaker_id = sp.id
            GROUP BY sp.id, sp.normalized_name, sp.full_name, sp.title, sp.position
            ORDER BY first_appearance NULLS LAST
            """
        )

        return [
            Speaker(
                speaker_id=row[0],
                normalized_name=row[1],
                full_name=row[2],
                title=row[3] or "",
                position=row[4] or "Unknown",
                role_in_video=row[5],
                first_appearance=str(row[6]) if row[6] is not None else "",
                total_appearances=row[7],
            )
            for row in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/speakers/{speaker_id}")
async def get_speaker_stats(speaker_id: str) -> SpeakerStatsResponse:
    """Get detailed stats for a speaker."""
    try:
        speaker_results = _get_postgres().execute_query(
            """
            SELECT id, normalized_name, full_name, title, position, 'member' as role_in_video,
                   MIN(s.timestamp_str) as first_appearance,
                   COUNT(DISTINCT s.youtube_video_id) as total_appearances
            FROM speakers sp
            LEFT JOIN sentences s ON s.speaker_id = sp.id
            WHERE sp.id = %s
            GROUP BY sp.id, sp.normalized_name, sp.full_name, sp.title, sp.position
            """,
            (speaker_id,),
        )

        if not speaker_results:
            raise HTTPException(status_code=404, detail="Speaker not found")

        speaker_row = speaker_results[0]

        contribution_results = _get_postgres().execute_query(
            """
            SELECT s.id, s.text, s.seconds_since_start, s.timestamp_str,
                   s.youtube_video_id, s.video_title, s.video_date, s.paragraph_id, s.speaker_id,
                   COALESCE(sp.full_name, sp.normalized_name, s.speaker_id) as speaker_name
            FROM sentences s
            LEFT JOIN speakers sp ON s.speaker_id = sp.id
            WHERE s.speaker_id = %s
            ORDER BY s.seconds_since_start DESC
            LIMIT 10
            """,
            (speaker_id,),
        )

        recent_contributions = [
            {
                "id": row[0],
                "text": row[1],
                "seconds_since_start": row[2],
                "timestamp_str": row[3],
                "video_id": row[4],
                "video_title": row[5],
                "video_date": str(row[6]) if row[6] is not None else "",
                "paragraph_id": row[7] or "",
                "speaker_id": row[8],
                "speaker_name": row[9],
                "score": 0.0,
                "search_type": "speaker",
            }
            for row in contribution_results
        ]

        return SpeakerStatsResponse(
            speaker_id=speaker_row[0],
            normalized_name=speaker_row[1],
            full_name=speaker_row[2],
            title=speaker_row[3] or "",
            position=speaker_row[4] or "Unknown",
            role_in_video=speaker_row[5],
            first_appearance=str(speaker_row[6]) if speaker_row[6] is not None else "",
            total_appearances=speaker_row[7],
            recent_contributions=recent_contributions,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/videos/{youtube_video_id}/speakers/{speaker_id}/roles",
    response_model=list[SpeakerVideoRole],
)
async def get_speaker_roles_for_video(
    youtube_video_id: str, speaker_id: str
) -> list[SpeakerVideoRole]:
    """Get session-scoped roles for a speaker in a given video."""

    try:
        rows = _get_postgres().execute_query(
            """
            SELECT role_label, role_kind, source, source_id, confidence, evidence
            FROM speaker_video_roles
            WHERE youtube_video_id = %s AND speaker_id = %s
            ORDER BY role_kind, role_label
            """,
            (youtube_video_id, speaker_id),
        )
        return [
            SpeakerVideoRole(
                role_label=r[0],
                role_kind=r[1],
                source=r[2],
                source_id=r[3],
                confidence=r[4],
                evidence=r[5],
            )
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/threads", response_model=CreateThreadResponse)
async def create_thread(title: str | None = None):
    """Create a new chat thread."""
    try:
        agent = _get_chat_agent()
        thread_id = agent.create_thread(title)
        from datetime import datetime

        return CreateThreadResponse(
            thread_id=thread_id,
            title=title,
            created_at=str(datetime.now()),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/threads/{thread_id}", response_model=GetThreadResponse)
async def get_thread(thread_id: str):
    """Get thread metadata and messages."""
    try:
        agent = _get_chat_agent()
        thread = agent.get_thread(thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")

        return GetThreadResponse(
            id=thread["id"],
            title=thread["title"],
            created_at=thread["created_at"],
            updated_at=thread["updated_at"],
            state=thread["state"],
            messages=[
                ThreadMessage(
                    id=m["id"],
                    role=m["role"],
                    content=m["content"],
                    metadata=m.get("metadata"),
                    created_at=m["created_at"],
                )
                for m in thread["messages"]
            ],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/threads/{thread_id}/messages", response_model=ChatMessageResponse)
async def send_message(thread_id: str, request: ChatMessageRequest):
    """Send a message to a thread and get assistant response."""
    try:
        agent = _get_chat_agent()
        response = await agent.process_message(thread_id, request.content)

        return ChatMessageResponse(
            thread_id=thread_id,
            assistant_message=ThreadMessage(**response["assistant_message"]),
            sources=[ChatSource(**s) for s in response.get("sources", [])],
            focus_node_ids=list(response.get("focus_node_ids", [])),
            followup_questions=list(response.get("followup_questions", [])),
            debug=response.get("debug"),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint for deployment monitoring."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api")
async def root():
    """Root endpoint."""
    return {
        "message": "Parliamentary Search API",
        "version": "1.0.0",
        "endpoints": {"search": "/search"},
    }


# Serve frontend static files
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
