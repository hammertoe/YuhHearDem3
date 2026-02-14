-- Add bill_excerpts table for storing chunked bill text with embeddings.

-- Create bill_excerpts table
CREATE TABLE IF NOT EXISTS bill_excerpts (
    id TEXT PRIMARY KEY,
    bill_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    embedding vector(768),
    tsv tsvector,
    source_url TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT uq_bill_excerpts_bill_chunk UNIQUE (bill_id, chunk_index),
    CONSTRAINT fk_bill_excerpts_bill
        FOREIGN KEY (bill_id) REFERENCES bills(id) ON DELETE CASCADE
);

-- Indexes for bill_excerpts
CREATE INDEX IF NOT EXISTS idx_bill_excerpts_bill_id ON bill_excerpts(bill_id);
CREATE INDEX IF NOT EXISTS idx_bill_excerpts_embedding ON bill_excerpts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Trigger function for tsv
CREATE OR REPLACE FUNCTION bill_excerpts_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('english', NEW.text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Attach trigger
DROP TRIGGER IF EXISTS bill_excerpts_tsv ON bill_excerpts;
CREATE TRIGGER bill_excerpts_tsv
    BEFORE INSERT OR UPDATE ON bill_excerpts
    FOR EACH ROW
    EXECUTE FUNCTION bill_excerpts_tsv_trigger();
