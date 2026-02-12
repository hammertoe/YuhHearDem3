from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import unquote

from psycopg import errors as pg_errors

from lib.db.chat_schema import ensure_chat_schema
from lib.kg_agent_loop import KGAgentLoop
from lib.utils.config import config


_SRC_MARKDOWN_LINK_RE = re.compile(r"\]\s*\(([^)\]]+)[)\]]", re.IGNORECASE)
_SRC_TOKEN_RE = re.compile(r"(?:#src:|source:)([^,\s)\]]+)", re.IGNORECASE)


def _should_trace() -> bool:
    """Check if chat tracing is enabled."""
    return getattr(config, "chat_trace", False)


def _truncate_text(text: str, max_len: int = 200) -> str:
    """Truncate text to max_len with ellipsis."""
    if not text or len(text) <= max_len:
        return text or ""
    return text[: max_len - 3] + "..."


def _trace_print(section: str, message: str) -> None:
    """Print a trace message with consistent formatting."""
    if not _should_trace():
        return
    print(f"🔍 [CHAT_TRACE] {section}")
    print(f"  {message}")


def _trace_section_start(section: str) -> None:
    """Print a trace section header."""
    if not _should_trace():
        return
    print(f"\n{'=' * 60}")
    print(f"🔍 [CHAT_TRACE] {section}")
    print(f"{'=' * 60}")


def _trace_section_end() -> None:
    """Print a trace section footer."""
    if not _should_trace():
        return
    print(f"{'=' * 60}\n")


def _extract_answer_citation_ids(answer: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in _SRC_MARKDOWN_LINK_RE.finditer(answer or ""):
        href = unquote(str(match.group(1) or "").strip())

        token_matches = [m.group(1).strip() for m in _SRC_TOKEN_RE.finditer(href)]
        if not token_matches and re.match(r"^https?://[^#]+#src:", href, re.IGNORECASE):
            token_matches = [href.split("#src:", 1)[1].strip()]

        for citation_id in token_matches:
            if not citation_id or citation_id in seen:
                continue
            seen.add(citation_id)
            out.append(citation_id)
    return out


def _normalize_citation_id(raw_id: str) -> str:
    """Normalize citation IDs across link and storage formats."""
    raw = unquote(str(raw_id or "").strip())
    raw = re.sub(r"^https?://[^#]+#", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^#?src:", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^source:", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"[\]\),.;]+$", "", raw)
    return raw.strip()


def _citation_lookup_keys(citation_id: str) -> list[str]:
    """Generate equivalent lookup keys for a citation ID."""
    normalized = _normalize_citation_id(citation_id)
    if not normalized:
        return []

    keys = [normalized, normalized.lower()]
    if normalized.startswith("utt_"):
        bare = normalized[4:]
        if bare:
            keys.extend([bare, bare.lower()])
    else:
        keys.extend([f"utt_{normalized}", f"utt_{normalized.lower()}"])

    out: list[str] = []
    seen: set[str] = set()
    for k in keys:
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def _looks_like_utterance_id(citation_id: str) -> bool:
    """Heuristic for plausible utterance IDs used by sentence records."""
    normalized = _normalize_citation_id(citation_id)
    if not normalized:
        return False
    if normalized.startswith("utt_"):
        normalized = normalized[4:]
    return re.match(r"^[A-Za-z0-9_-]+:\d+$", normalized) is not None


def _merge_cite_utterance_ids(
    *, answer: str, cite_utterance_ids: list[str], retrieval: dict[str, Any] | None
) -> list[str]:
    answer_ids = _extract_answer_citation_ids(answer)
    combined = answer_ids + [str(x or "").strip() for x in cite_utterance_ids]

    known_ids = [
        str(c.get("utterance_id") or "").strip()
        for c in (retrieval or {}).get("citations", [])
        if isinstance(c, dict)
    ]
    known_ids = [k for k in known_ids if k]
    known_lookup: dict[str, str] = {}
    for known_id in known_ids:
        for key in _citation_lookup_keys(known_id):
            known_lookup[key] = known_id

    suffix_counts: dict[str, int] = {}
    for known_id in known_ids:
        match = re.search(r":(\d+)$", known_id)
        if not match:
            continue
        seconds = match.group(1)
        suffix_counts[seconds] = suffix_counts.get(seconds, 0) + 1

    out: list[str] = []
    seen: set[str] = set()
    for raw in combined:
        uid = _normalize_citation_id(str(raw or ""))
        if not uid:
            continue

        resolved_uid = uid
        if known_lookup:
            matched = None
            for key in _citation_lookup_keys(uid):
                if key in known_lookup:
                    matched = known_lookup[key]
                    break

            if matched is None:
                sec_match = re.match(r"^(?:utt_)?(\d+)$", uid, re.IGNORECASE)
                if sec_match:
                    seconds = sec_match.group(1)
                    if suffix_counts.get(seconds) == 1:
                        matched = next(
                            (
                                known_id
                                for known_id in known_ids
                                if known_id.endswith(f":{seconds}")
                            ),
                            None,
                        )

            if matched is None:
                if not _looks_like_utterance_id(uid):
                    continue
            else:
                resolved_uid = matched

        if resolved_uid in seen:
            continue
        seen.add(resolved_uid)
        out.append(resolved_uid)
    return out


@dataclass
class ChatSource:
    utterance_id: str
    youtube_video_id: str
    youtube_url: str
    seconds_since_start: int
    timestamp_str: str
    speaker_id: str
    speaker_name: str
    speaker_title: str | None
    text: str
    video_title: str | None
    video_date: str | None


class KGChatAgentV2:
    """Chat agent powered by Gemini tool-calling over the canonical KG.

    The agent loop uses a deterministic Graph-RAG tool to retrieve a compact
    subgraph + citations (with youtube timecoded URLs). The model then answers
    grounded in that evidence.
    """

    def __init__(
        self,
        *,
        postgres_client: Any,
        embedding_client: Any,
        model: str = "gemini-3-flash-preview",
        client: Any | None = None,
        enable_thinking: bool = False,
    ) -> None:
        self.postgres = postgres_client
        self.embedding = embedding_client
        self._chat_schema_ensured = False
        self.loop = KGAgentLoop(
            postgres=self.postgres,
            embedding_client=self.embedding,
            client=client,
            model=model,
            enable_thinking=enable_thinking,
        )

    def _ensure_chat_schema(self) -> None:
        if self._chat_schema_ensured:
            return
        ensure_chat_schema(self.postgres)
        self._chat_schema_ensured = True

    def create_thread(self, title: str | None = None) -> str:
        self._ensure_chat_schema()
        thread_id = str(uuid.uuid4())
        try:
            self.postgres.execute_update(
                "INSERT INTO chat_threads (id, title) VALUES (%s, %s)",
                (thread_id, title),
            )
        except pg_errors.UndefinedTable:
            # If the database schema changes mid-process (or another client is
            # using a different schema), retry once after re-ensuring.
            self._chat_schema_ensured = False
            self._ensure_chat_schema()
            self.postgres.execute_update(
                "INSERT INTO chat_threads (id, title) VALUES (%s, %s)",
                (thread_id, title),
            )
        return thread_id

    def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        self._ensure_chat_schema()
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
        thread: dict[str, Any] = {
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
        thread["messages"] = [
            {
                "id": r[0],
                "role": r[1],
                "content": r[2],
                "metadata": r[3],
                "created_at": str(r[4]),
            }
            for r in msg_rows
        ]

        state_rows = self.postgres.execute_query(
            "SELECT state FROM chat_thread_state WHERE thread_id = %s",
            (thread_id,),
        )
        if state_rows:
            try:
                thread["state"] = json.loads(state_rows[0][0])
            except Exception:
                thread["state"] = {}
        else:
            thread["state"] = {}

        return thread

    def _get_recent_history_for_llm(self, thread_id: str, limit: int = 16) -> list[dict[str, str]]:
        rows = self.postgres.execute_query(
            """
            SELECT role, content
            FROM chat_messages
            WHERE thread_id = %s AND role IN ('user', 'assistant')
            ORDER BY created_at ASC
            """,
            (thread_id,),
        )
        msgs = [{"role": r[0], "content": r[1]} for r in rows]
        return msgs[-limit:]

    def _sources_from_retrieval(
        self,
        retrieval: dict[str, Any] | None,
        cite_utterance_ids: list[str],
        max_sources: int = 24,
    ) -> list[ChatSource]:
        citations = (retrieval.get("citations") or []) if retrieval else []
        citation_by_id: dict[str, dict[str, Any]] = {}
        for c in citations:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("utterance_id") or "").strip()
            if not cid:
                continue
            for key in _citation_lookup_keys(cid):
                citation_by_id[key] = c
        fetched_ids: set[str] = set()
        out: list[ChatSource] = []

        for uid in cite_utterance_ids:
            normalized_uid = _normalize_citation_id(uid)
            if not normalized_uid:
                continue

            c = None
            for key in _citation_lookup_keys(normalized_uid):
                c = citation_by_id.get(key)
                if c is not None:
                    break

            if c:
                canonical_id = str(c.get("utterance_id") or normalized_uid)
                if canonical_id in fetched_ids:
                    continue
                out.append(
                    ChatSource(
                        utterance_id=c.get("utterance_id", ""),
                        youtube_video_id=c.get("youtube_video_id", ""),
                        youtube_url=c.get("youtube_url", ""),
                        seconds_since_start=int(c.get("seconds_since_start") or 0),
                        timestamp_str=c.get("timestamp_str", ""),
                        speaker_id=c.get("speaker_id", ""),
                        speaker_name=c.get("speaker_name", ""),
                        speaker_title=c.get("speaker_title"),
                        text=c.get("text", ""),
                        video_title=c.get("video_title"),
                        video_date=c.get("video_date"),
                    )
                )
                fetched_ids.add(canonical_id)
            else:
                if normalized_uid in fetched_ids:
                    continue
                source = self._fetch_source_by_id(normalized_uid)
                if source:
                    out.append(source)
                    fetched_ids.add(source.utterance_id)
            if len(out) >= max_sources:
                break
        return out

    def _fetch_source_by_id(self, utterance_id: str) -> ChatSource | None:
        """Fetch a source from the database by utterance ID."""
        try:
            rows = self.postgres.execute_query(
                """
                SELECT
                    s.id,
                    s.youtube_video_id,
                    s.seconds_since_start,
                    s.timestamp_str,
                    s.text,
                    s.speaker_id,
                    COALESCE(sp.full_name, sp.normalized_name, s.speaker_id) AS speaker_name,
                    COALESCE(vrole.role_label, sp.position, sp.title, '') AS speaker_title,
                    s.video_title,
                    s.video_date
                FROM sentences s
                LEFT JOIN speakers sp ON sp.id = s.speaker_id
                LEFT JOIN LATERAL (
                    SELECT svr.role_label
                    FROM speaker_video_roles svr
                    WHERE svr.youtube_video_id = s.youtube_video_id
                      AND svr.speaker_id = s.speaker_id
                    ORDER BY
                        CASE svr.role_kind
                            WHEN 'portfolio' THEN 1
                            WHEN 'office' THEN 2
                            WHEN 'session_role' THEN 3
                            ELSE 9
                        END,
                        svr.role_label
                    LIMIT 1
                ) AS vrole ON TRUE
                WHERE s.id = %s
                LIMIT 1
                """,
                (utterance_id,),
            )
            if not rows:
                return None
            row = rows[0]
            return ChatSource(
                utterance_id=row[0],
                youtube_video_id=row[1],
                youtube_url=f"https://youtube.com/watch?v={row[1]}&t={row[2]}",
                seconds_since_start=row[2],
                timestamp_str=row[3] or "",
                speaker_id=row[5] or "",
                speaker_name=row[6] or "",
                speaker_title=row[7] or None,
                text=row[4] or "",
                video_title=row[8] or "",
                video_date=str(row[9]) if row[9] else None,
            )
        except Exception:
            return None

    async def process_message(self, thread_id: str, user_content: str) -> dict[str, Any]:
        self._ensure_chat_schema()
        thread = self.get_thread(thread_id)
        if thread is None:
            raise ValueError(f"Thread {thread_id} not found")

        user_content = (user_content or "").strip()
        if not user_content:
            raise ValueError("Empty message")

        user_message_id = str(uuid.uuid4())
        _trace_section_start("PROCESS MESSAGE START")
        _trace_print("Thread ID", thread_id)
        _trace_print("User Message ID", user_message_id)
        _trace_print("User Query", f"{_truncate_text(user_content)} ({len(user_content)} chars)")
        _trace_section_end()

        self.postgres.execute_update(
            """
            INSERT INTO chat_messages (id, thread_id, role, content, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (user_message_id, thread_id, "user", user_content, datetime.now()),
        )

        history = self._get_recent_history_for_llm(thread_id)
        _trace_section_start("AGENT LOOP EXECUTION")
        _trace_print("History Size", f"{len(history)} messages")
        _trace_section_end()

        result = await self.loop.run(user_message=user_content, history=history)

        answer = (
            str(result.get("answer") or "").strip()
            or "I couldn't find enough evidence to answer that."
        )
        retrieval = result.get("retrieval")
        cite_utterance_ids = _merge_cite_utterance_ids(
            answer=answer,
            cite_utterance_ids=list(result.get("cite_utterance_ids") or []),
            retrieval=retrieval if isinstance(retrieval, dict) else None,
        )
        focus_node_ids = list(result.get("focus_node_ids") or [])
        followup_questions = [
            str(q).strip()
            for q in list(result.get("followup_questions") or [])
            if str(q or "").strip()
        ][:4]

        desired_sources = max(8, min(24, len(cite_utterance_ids) or 8))
        sources = self._sources_from_retrieval(
            retrieval,
            cite_utterance_ids,
            max_sources=desired_sources,
        )
        serialized_sources = [s.__dict__ for s in sources]

        assistant_message_id = str(uuid.uuid4())
        metadata = {
            "cite_utterance_ids": cite_utterance_ids,
            "focus_node_ids": focus_node_ids,
            "followup_questions": followup_questions,
            "sources": serialized_sources,
            "retrieval_debug": (retrieval or {}).get("debug")
            if isinstance(retrieval, dict)
            else None,
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
                answer,
                json.dumps(metadata),
                datetime.now(),
            ),
        )

        # Update thread state for follow-ups
        node_label_by_id = (
            {n.get("id"): n.get("label") for n in (retrieval or {}).get("nodes", [])}
            if isinstance(retrieval, dict)
            else {}
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
                        "focus_node_ids": focus_node_ids,
                        "focus_node_labels": [
                            node_label_by_id.get(nid, nid) for nid in focus_node_ids
                        ],
                        "last_user_message": user_content,
                    }
                ),
                datetime.now(),
            ),
        )

        _trace_section_start("RESPONSE SUMMARY")
        _trace_print("Assistant Message ID", assistant_message_id)
        _trace_print("Answer Length", f"{len(answer)} chars")
        _trace_print("Citation IDs", f"{len(cite_utterance_ids)} items")
        _trace_print("Focus Node IDs", f"{len(focus_node_ids)} items")
        _trace_print("Sources Count", f"{len(sources)} items")
        _trace_section_end()

        return {
            "thread_id": thread_id,
            "assistant_message": {
                "id": assistant_message_id,
                "role": "assistant",
                "content": answer,
                "created_at": str(datetime.now()),
                "metadata": metadata,
            },
            "sources": serialized_sources,
            "focus_node_ids": focus_node_ids,
            "followup_questions": followup_questions,
            "debug": {
                "tool_iterations": None,
                "retrieval": (retrieval or {}).get("debug")
                if isinstance(retrieval, dict)
                else None,
            },
        }
