"""Chat schema helpers.

The chat agents persist threads/messages/state in Postgres. In dev setups it's
easy to have an older database that predates chat tables; this module provides
an idempotent schema bootstrap.
"""

from __future__ import annotations

from typing import Any


CHAT_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS chat_threads (
        id TEXT PRIMARY KEY,
        title TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id TEXT PRIMARY KEY,
        thread_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
        content TEXT NOT NULL,
        metadata JSONB,
        created_at TIMESTAMP DEFAULT NOW(),
        CONSTRAINT fk_chat_messages_thread
            FOREIGN KEY (thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_thread_state (
        thread_id TEXT PRIMARY KEY,
        state JSONB NOT NULL,
        updated_at TIMESTAMP DEFAULT NOW(),
        CONSTRAINT fk_chat_thread_state_thread
            FOREIGN KEY (thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_id ON chat_messages(thread_id)",
    "CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_chat_thread_state_thread_id ON chat_thread_state(thread_id)",
)


def ensure_chat_schema(postgres: Any) -> None:
    """Ensure chat tables/indexes exist.

    This is safe to call multiple times.
    """

    for stmt in CHAT_SCHEMA_STATEMENTS:
        postgres.execute_update(stmt)
