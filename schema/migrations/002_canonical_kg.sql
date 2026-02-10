-- Canonical Knowledge Graph tables for LLM-first KG extraction
-- Migration: 002_canonical_kg.sql

-- ============================================================================
-- KG NODES: Canonical nodes with embeddings
-- ============================================================================

CREATE TABLE IF NOT EXISTS kg_nodes (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    type TEXT NOT NULL,
    aliases TEXT[] DEFAULT '{}',
    embedding vector(768),
    tsv tsvector,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for kg_nodes
CREATE INDEX IF NOT EXISTS idx_kg_nodes_embedding ON kg_nodes
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_kg_nodes_type_id ON kg_nodes(type, id);

CREATE INDEX IF NOT EXISTS idx_kg_nodes_tsv ON kg_nodes
    USING gin (tsv);

CREATE INDEX IF NOT EXISTS idx_kg_nodes_label ON kg_nodes(label);

-- ============================================================================
-- KG ALIASES: Alias index for fast linking
-- ============================================================================

CREATE TABLE IF NOT EXISTS kg_aliases (
    alias_norm TEXT PRIMARY KEY,
    alias_raw TEXT NOT NULL,
    node_id TEXT NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence FLOAT
);

-- Indexes for kg_aliases
CREATE INDEX IF NOT EXISTS idx_kg_aliases_node_id ON kg_aliases(node_id);

CREATE INDEX IF NOT EXISTS idx_kg_aliases_type ON kg_aliases(type);

CREATE INDEX IF NOT EXISTS idx_kg_aliases_source ON kg_aliases(source);

-- ============================================================================
-- KG EDGES: Canonical edges with provenance
-- ============================================================================

CREATE TABLE IF NOT EXISTS kg_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    predicate_raw TEXT,
    target_id TEXT NOT NULL,
    youtube_video_id TEXT NOT NULL,
    earliest_timestamp_str TEXT,
    earliest_seconds INTEGER,
    utterance_ids TEXT[],
    evidence TEXT,
    speaker_ids TEXT[],
    confidence FLOAT,
    extractor_model TEXT,
    kg_run_id TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for kg_edges
CREATE INDEX IF NOT EXISTS idx_kg_edges_video_time ON kg_edges(youtube_video_id, earliest_seconds);

CREATE INDEX IF NOT EXISTS idx_kg_edges_triple ON kg_edges(source_id, predicate, target_id);

CREATE INDEX IF NOT EXISTS idx_kg_edges_source_id ON kg_edges(source_id);

CREATE INDEX IF NOT EXISTS idx_kg_edges_target_id ON kg_edges(target_id);

CREATE INDEX IF NOT EXISTS idx_kg_edges_run_id ON kg_edges(kg_run_id);

-- ============================================================================
-- TRIGGERS: Auto-update tsv columns
-- ============================================================================

CREATE OR REPLACE FUNCTION kg_nodes_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('english', coalesce(NEW.label, '') || ' ' || coalesce(array_to_string(NEW.aliases, ' '), ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER kg_nodes_tsv_update BEFORE INSERT OR UPDATE ON kg_nodes
    FOR EACH ROW EXECUTE FUNCTION kg_nodes_tsv_trigger();

-- ============================================================================
-- FOREIGN KEY CONSTRAINTS: Add foreign keys to kg_edges
-- ============================================================================

ALTER TABLE kg_edges
    ADD CONSTRAINT fk_kg_edges_source
    FOREIGN KEY (source_id) REFERENCES kg_nodes(id) ON DELETE CASCADE;

ALTER TABLE kg_edges
    ADD CONSTRAINT fk_kg_edges_target
    FOREIGN KEY (target_id) REFERENCES kg_nodes(id) ON DELETE CASCADE;

-- ============================================================================
-- VIEWS: Common queries
-- ============================================================================

CREATE OR REPLACE VIEW kg_edges_with_details AS
SELECT
    e.id AS edge_id,
    e.predicate,
    e.predicate_raw,
    e.youtube_video_id,
    e.earliest_timestamp_str,
    e.earliest_seconds,
    e.utterance_ids,
    e.evidence,
    e.speaker_ids,
    e.confidence,
    e.extractor_model,
    e.kg_run_id,
    e.created_at,
    source_node.id AS source_id,
    source_node.label AS source_label,
    source_node.type AS source_type,
    target_node.id AS target_id,
    target_node.label AS target_label,
    target_node.type AS target_type
FROM kg_edges e
JOIN kg_nodes source_node ON e.source_id = source_node.id
JOIN kg_nodes target_node ON e.target_id = target_node.id;
