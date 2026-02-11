"""Unit tests for chat schema bootstrapping."""

from __future__ import annotations

from psycopg import errors as pg_errors

from lib.chat_agent_v2 import KGChatAgentV2
from lib.db.chat_schema import CHAT_SCHEMA_STATEMENTS, ensure_chat_schema


class _FakePostgres:
    def __init__(self) -> None:
        self.updates: list[tuple[str, tuple | None]] = []
        self._chat_threads_created = False

    def execute_update(self, query: str, params: tuple | None = None) -> int:
        q = " ".join((query or "").split())
        self.updates.append((q, params))

        if "CREATE TABLE IF NOT EXISTS chat_threads" in q:
            self._chat_threads_created = True

        if q.startswith("INSERT INTO chat_threads") and not self._chat_threads_created:
            raise pg_errors.UndefinedTable('relation "chat_threads" does not exist')

        return 1


def test_ensure_chat_schema_executes_all_statements() -> None:
    pg = _FakePostgres()
    ensure_chat_schema(pg)

    assert len(pg.updates) == len(CHAT_SCHEMA_STATEMENTS)
    assert any("CREATE TABLE IF NOT EXISTS chat_threads" in q for q, _ in pg.updates)
    assert any("CREATE TABLE IF NOT EXISTS chat_messages" in q for q, _ in pg.updates)
    assert any(
        "CREATE TABLE IF NOT EXISTS chat_thread_state" in q for q, _ in pg.updates
    )


def test_create_thread_bootstraps_schema_on_missing_tables() -> None:
    pg = _FakePostgres()

    # Pass a dummy client to avoid GOOGLE_API_KEY checks inside KGAgentLoop.
    agent = KGChatAgentV2(
        postgres_client=pg,
        embedding_client=object(),
        client=object(),
    )
    thread_id = agent.create_thread("Test")

    assert thread_id
    assert len(thread_id) == 36

    inserts = [q for q, _ in pg.updates if q.startswith("INSERT INTO chat_threads")]
    assert len(inserts) == 1
    assert any("CREATE TABLE IF NOT EXISTS chat_threads" in q for q, _ in pg.updates)
