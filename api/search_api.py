"""Hybrid search API combining vector search and graph traversal."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from lib.advanced_search_features import AdvancedSearchFeatures
from lib.chat_agent_v2 import KGChatAgentV2
from lib.db.chat_schema import ensure_chat_schema
from lib.db.postgres_client import PostgresClient
from lib.embeddings.google_client import GoogleEmbeddingClient
from lib.utils.config import config

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Parliamentary Search API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

cors_origins = config.api.cors_origins.split(",") if config.api.cors_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=config.api.cors_allow_credentials,
    allow_methods=(
        config.api.cors_allow_methods.split(",") if config.api.cors_allow_methods != "*" else ["*"]
    ),
    allow_headers=(
        config.api.cors_allow_headers.split(",") if config.api.cors_allow_headers != "*" else ["*"]
    ),
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests for audit trail."""
    start_time = datetime.now()

    logger.info(
        f"Request: {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}"
    )

    try:
        response = await call_next(request)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Response: {response.status_code} for {request.method} {request.url.path} - {duration:.3f}s"
        )
        return response
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(
            f"Request failed: {request.method} {request.url.path} - {duration:.3f}s - {e}",
            exc_info=True,
        )
        raise


def _get_postgres() -> PostgresClient:
    assert postgres is not None
    return postgres


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
    global postgres, embedding_client, advanced_search, chat_agent
    postgres = PostgresClient()
    try:
        ensure_chat_schema(postgres)
    except Exception as e:
        print(f"❌ Failed to ensure chat schema: {e}")
        raise
    embedding_client = GoogleEmbeddingClient()
    advanced_search = AdvancedSearchFeatures(
        postgres=postgres,
        embedding_client=embedding_client,
    )
    chat_agent = KGChatAgentV2(
        postgres_client=postgres,
        embedding_client=embedding_client,
        model=getattr(config, "gemini_model", "gemini-2.5-flash"),
        enable_thinking=getattr(config, "enable_thinking", False),
    )


class SearchRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=500)]
    limit: Annotated[int, Field(ge=1, le=100)] = 20
    alpha: Annotated[float, Field(ge=0.0, le=1.0)] = 0.6


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
    query: Annotated[str, Field(min_length=1, max_length=500)]
    limit: Annotated[int, Field(ge=1, le=100)] = 20
    alpha: Annotated[float | None, Field(ge=0.0, le=1.0)] = 0.6
    start_date: str | None = None
    end_date: str | None = None
    speaker_id: str | None = None
    entity_type: str | None = None


class TrendResult(BaseModel):
    entity_id: str | None
    trends: list[dict[str, Any]]
    summary: dict[str, Any]
    moving_average: list[dict[str, Any]]


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
    source_kind: str = "utterance"
    citation_id: str | None = None
    utterance_id: str = ""
    youtube_video_id: str = ""
    youtube_url: str = ""
    seconds_since_start: int = 0
    timestamp_str: str = ""
    speaker_id: str = ""
    speaker_name: str = ""
    speaker_title: str | None = None
    text: str = ""
    video_title: str | None = None
    video_date: str | None = None
    bill_id: str | None = None
    bill_number: str | None = None
    bill_title: str | None = None
    excerpt: str | None = None
    source_url: str | None = None
    chunk_index: int | None = None
    page_number: int | None = None
    matched_terms: list[str] | None = None


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
    content: Annotated[str, Field(min_length=1, max_length=5000)]


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
embedding_client: GoogleEmbeddingClient | None = None
advanced_search: AdvancedSearchFeatures | None = None
chat_agent: KGChatAgentV2 | None = None


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
@limiter.limit("30/minute")
async def search(request: Request, search_request: SearchRequest):
    """Hybrid search combining entity + paragraph vector search."""
    try:
        try:
            query_embedding = _get_embedding_client().generate_query_embedding(search_request.query)
        except Exception as e:
            print(f"⚠️ Embeddings unavailable; falling back to BM25 only: {e}")
            return [
                SearchResult(**r)
                for r in bm25_search_sentences(search_request.query, search_request.limit)
            ]

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
            for r in scored[: search_request.limit]
        ]

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/search/temporal", response_model=list[SearchResult])
@limiter.limit("30/minute")
async def temporal_search(request: Request, temporal_request: TemporalSearchRequest):
    """Temporal search with filters."""
    try:
        try:
            results = _get_advanced_search().temporal_search(
                temporal_request.query,
                temporal_request.start_date,
                temporal_request.end_date,
                temporal_request.speaker_id,
                temporal_request.entity_type,
                temporal_request.limit,
            )
            return [SearchResult(**r) for r in results]
        except Exception as e:
            # If embeddings are not available, fall back to BM25 filtered by dates/speaker.
            print(f"⚠️ Temporal embeddings unavailable; using BM25 fallback: {e}")

            where = ["s.tsv @@ plainto_tsquery('english', %s)"]
            params: list[Any] = [temporal_request.query]

            if temporal_request.start_date:
                where.append("s.video_date >= to_date(%s, 'YYYY-MM-DD')")
                params.append(temporal_request.start_date)
            if temporal_request.end_date:
                where.append("s.video_date <= to_date(%s, 'YYYY-MM-DD')")
                params.append(temporal_request.end_date)
            if temporal_request.speaker_id:
                where.append("s.speaker_id = %s")
                params.append(temporal_request.speaker_id)

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
            rank_query = temporal_request.query
            final_params = [rank_query, *params, temporal_request.limit]
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
        logger.error(f"Temporal search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


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
        logger.error(f"Trends retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


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
        logger.error(f"Get speakers failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


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
        logger.error(f"Get speaker stats failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


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
        logger.error(f"Create thread failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/chat/threads/{thread_id}/messages", response_model=ChatMessageResponse)
@limiter.limit("10/minute")
async def send_message(request: Request, thread_id: str, chat_request: ChatMessageRequest):
    """Send a message to a thread and get assistant response."""
    try:
        agent = _get_chat_agent()
        response = await agent.process_message(thread_id, chat_request.content)

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
        logger.error(f"Send message failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


async def stream_chat_response(thread_id: str, content: str):
    """Stream chat response with progress updates."""
    from lib.chat_agent_v2 import STAGE_MESSAGES

    progress_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    result_container: dict[str, Any] = {}
    error_container: dict[str, str] = {}
    done = False

    def progress_callback(stage: str, message: str | None) -> None:
        msg = message or STAGE_MESSAGES.get(stage, "")
        progress_queue.put_nowait((stage, msg))

    async def run_agent():
        nonlocal done
        try:
            agent = _get_chat_agent()
            response = await agent.process_message(
                thread_id, content, progress_callback=progress_callback
            )
            result_container["response"] = response
        except Exception as e:
            error_container["error"] = str(e)
        finally:
            done = True

    # Start agent task
    agent_task = asyncio.create_task(run_agent())

    # Drain progress while agent runs
    while not done:
        try:
            stage, msg = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
            event = json.dumps({"stage": stage, "message": msg})
            yield f"event: stage\ndata: {event}\n\n"
        except TimeoutError:
            pass

    await agent_task

    # Drain remaining progress
    while not progress_queue.empty():
        try:
            stage, msg = progress_queue.get_nowait()
            event = json.dumps({"stage": stage, "message": msg})
            yield f"event: stage\ndata: {event}\n\n"
        except Exception:
            break

    if "error" in error_container:
        error = json.dumps({"error": error_container["error"]})
        yield f"event: error\ndata: {error}\n\n"
    else:
        response = result_container["response"]
        final = {
            "thread_id": thread_id,
            "assistant_message": response["assistant_message"],
            "sources": response.get("sources", []),
            "focus_node_ids": list(response.get("focus_node_ids", [])),
            "followup_questions": list(response.get("followup_questions", [])),
            "debug": response.get("debug"),
        }
        yield f"event: final\ndata: {json.dumps(final)}\n\n"


@app.get("/chat/threads/{thread_id}/messages/stream")
@limiter.limit("10/minute")
async def stream_message(request: Request, thread_id: str, content: str):
    """Stream a message response with progress updates via SSE."""
    return StreamingResponse(
        stream_chat_response(thread_id, content),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    """Health check endpoint for deployment monitoring with dependency validation."""
    health_status = {"status": "ok", "timestamp": datetime.now().isoformat(), "checks": {}}

    try:
        _get_postgres().execute_query("SELECT 1")
        health_status["checks"]["database"] = "ok"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["checks"]["database"] = f"error: {e}"

    try:
        if embedding_client:
            _get_embedding_client().generate_query_embedding("test")
            health_status["checks"]["embeddings"] = "ok"
        else:
            health_status["checks"]["embeddings"] = "skipped"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["checks"]["embeddings"] = f"error: {e}"

    try:
        _get_chat_agent()
        health_status["checks"]["chat_agent"] = "ok"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["checks"]["chat_agent"] = f"error: {e}"

    return health_status


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
