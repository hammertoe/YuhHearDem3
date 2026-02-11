"""Migrate chat schema from old chat_sessions to new chat_threads model."""

import sys

sys.path.insert(0, ".")

from lib.db.postgres_client import PostgresClient


def migrate_chat_schema(pg: PostgresClient) -> None:
    """Add thread_id column to chat_messages and clean up old schema."""

    # Add thread_id column if it doesn't exist
    try:
        pg.execute_update("""
            ALTER TABLE chat_messages ADD COLUMN thread_id TEXT
        """)
        print("✓ Added thread_id column to chat_messages")
    except Exception as e:
        if "already exists" in str(e):
            print("✓ thread_id column already exists")
        else:
            raise

    # Drop old foreign key constraint if it exists
    try:
        pg.execute_update("""
            ALTER TABLE chat_messages
            DROP CONSTRAINT IF EXISTS chat_messages_chat_session_id_fkey
        """)
        print("✓ Dropped old chat_session_id foreign key constraint")
    except Exception as e:
        print(f"⚠ Could not drop FK constraint: {e}")

    # Drop old indexes if they exist
    try:
        pg.execute_update("DROP INDEX IF EXISTS idx_chat_messages_session_id")
        print("✓ Dropped old idx_chat_messages_session_id index")
    except Exception:
        pass

    # Set default thread_id for existing rows (legacy messages get a placeholder)
    try:
        pg.execute_update("""
            UPDATE chat_messages SET thread_id = 'legacy_messages'
            WHERE thread_id IS NULL
        """)
        print("✓ Set default thread_id for legacy messages")
    except Exception as e:
        print(f"⚠ Could not set default thread_id: {e}")

    # Make thread_id NOT NULL (it should be for new messages)
    try:
        pg.execute_update("""
            ALTER TABLE chat_messages
            ALTER COLUMN thread_id SET NOT NULL
        """)
        print("✓ Set thread_id as NOT NULL")
    except Exception as e:
        if "already not null" in str(e):
            print("✓ thread_id already NOT NULL")
        else:
            raise

    # Add index on thread_id for performance
    try:
        pg.execute_update("""
            CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_id ON chat_messages(thread_id)
        """)
        print("✓ Created index on thread_id")
    except Exception as e:
        print(f"⚠ Could not create index: {e}")

    print("\n✅ Migration complete! The chat_messages table now supports thread_id.")


if __name__ == "__main__":
    pg = PostgresClient()
    migrate_chat_schema(pg)
