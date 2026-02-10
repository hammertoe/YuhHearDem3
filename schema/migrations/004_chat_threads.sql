-- ============================================================================
-- Chat Threads & Messages (for KG conversational agent)
-- ============================================================================

CREATE TABLE IF NOT EXISTS chat_threads (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT fk_chat_messages_thread
        FOREIGN KEY (thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chat_thread_state (
    thread_id TEXT PRIMARY KEY,
    state JSONB NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT fk_chat_thread_state_thread
        FOREIGN KEY (thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_id ON chat_messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_thread_state_thread_id ON chat_thread_state(thread_id);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_id ON chat_threads(thread_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_thread_state_thread_id ON chat_thread_state(thread_id);
