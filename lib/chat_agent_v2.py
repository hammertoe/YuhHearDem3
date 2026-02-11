from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psycopg import errors as pg_errors

from lib.db.chat_schema import ensure_chat_schema
from lib.kg_agent_loop import KGAgentLoop


@dataclass
class ChatSource:
    utterance_id: str
    youtube_video_id: str
    youtube_url: str
    seconds_since_start: int
    timestamp_str: str
    speaker_id: str
    speaker_name: str
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
    ) -> None:
        self.postgres = postgres_client
        self.embedding = embedding_client
        self._chat_schema_ensured = False
        self.loop = KGAgentLoop(
            postgres=self.postgres,
            embedding_client=self.embedding,
            client=client,
            model=model,
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

    def _get_recent_history_for_llm(
        self, thread_id: str, limit: int = 16
    ) -> list[dict[str, str]]:
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
        max_sources: int = 8,
    ) -> list[ChatSource]:
        if not retrieval:
            return []
        citations = retrieval.get("citations") or []
        citation_by_id = {c.get("utterance_id"): c for c in citations}

        out: list[ChatSource] = []
        for uid in cite_utterance_ids:
            c = citation_by_id.get(uid)
            if not c:
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
                    text=c.get("text", ""),
                    video_title=c.get("video_title"),
                    video_date=c.get("video_date"),
                )
            )
            if len(out) >= max_sources:
                break
        return out

    async def process_message(
        self, thread_id: str, user_content: str
    ) -> dict[str, Any]:
        self._ensure_chat_schema()
        thread = self.get_thread(thread_id)
        if thread is None:
            raise ValueError(f"Thread {thread_id} not found")

        user_content = (user_content or "").strip()
        if not user_content:
            raise ValueError("Empty message")

        user_message_id = str(uuid.uuid4())
        self.postgres.execute_update(
            """
            INSERT INTO chat_messages (id, thread_id, role, content, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (user_message_id, thread_id, "user", user_content, datetime.now()),
        )

        history = self._get_recent_history_for_llm(thread_id)
        result = await self.loop.run(user_message=user_content, history=history)

        answer = (
            str(result.get("answer") or "").strip()
            or "I couldn't find enough evidence to answer that."
        )
        cite_utterance_ids = list(result.get("cite_utterance_ids") or [])
        focus_node_ids = list(result.get("focus_node_ids") or [])
        retrieval = result.get("retrieval")

        assistant_message_id = str(uuid.uuid4())
        metadata = {
            "cite_utterance_ids": cite_utterance_ids,
            "focus_node_ids": focus_node_ids,
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

        sources = self._sources_from_retrieval(retrieval, cite_utterance_ids)

        return {
            "thread_id": thread_id,
            "assistant_message": {
                "id": assistant_message_id,
                "role": "assistant",
                "content": answer,
                "created_at": str(datetime.now()),
                "metadata": metadata,
            },
            "sources": [s.__dict__ for s in sources],
            "focus_node_ids": focus_node_ids,
            "debug": {
                "tool_iterations": None,
                "retrieval": (retrieval or {}).get("debug")
                if isinstance(retrieval, dict)
                else None,
            },
        }
