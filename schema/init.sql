-- ============================================================================
-- Three-Tier Schema for Parliamentary Search System
-- ============================================================================
-- TIER 1: Paragraphs (Have embeddings - Search targets)
-- TIER 2: Entities (Have embeddings - Search targets)
-- TIER 3: Sentences (NO embeddings - Context/provenance only)
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- ============================================================================
-- TIER 1: PARAGRAPHS
-- ============================================================================

CREATE TABLE IF NOT EXISTS paragraphs (
    id TEXT PRIMARY KEY,
    youtube_video_id TEXT NOT NULL,
    start_seconds INTEGER NOT NULL,
    end_seconds INTEGER NOT NULL,
    text TEXT NOT NULL,
    speaker_id TEXT NOT NULL,
    voice_id INTEGER,
    start_timestamp TEXT,
    end_timestamp TEXT,
    embedding vector(768),
    tsv tsvector,
    video_date DATE,
    video_title TEXT,
    sentence_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- TIER 2: ENTITIES
-- ============================================================================

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    type TEXT NOT NULL,
    embedding vector(768),
    tsv tsvector,
    normalized_name TEXT,
    first_seen_date DATE,
    last_seen_date DATE,
    mention_count INTEGER DEFAULT 0,
    title TEXT,
    position TEXT,
    party TEXT,
    bill_number TEXT,
    bill_status TEXT,
    category TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- TIER 3: SENTENCES (NO embeddings)
-- ============================================================================

CREATE TABLE IF NOT EXISTS sentences (
    id TEXT PRIMARY KEY,
    youtube_video_id TEXT NOT NULL,
    seconds_since_start INTEGER NOT NULL,
    timestamp_str TEXT,
    text TEXT NOT NULL,
    speaker_id TEXT NOT NULL,
    voice_id INTEGER,
    paragraph_id TEXT NOT NULL,
    sentence_order INTEGER NOT NULL,
    tsv tsvector,
    video_date DATE,
    video_title TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT fk_paragraph
        FOREIGN KEY (paragraph_id) REFERENCES paragraphs(id) ON DELETE CASCADE
);

-- ============================================================================
-- RELATIONSHIPS: Link entities to paragraphs and sentences
-- ============================================================================

CREATE TABLE IF NOT EXISTS paragraph_entities (
    paragraph_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    relationship_type TEXT,
    confidence FLOAT,
    PRIMARY KEY (paragraph_id, entity_id),
    CONSTRAINT fk_pe_paragraph
        FOREIGN KEY (paragraph_id) REFERENCES paragraphs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sentence_entities (
    sentence_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    relationship_type TEXT,
    context_snippet TEXT,
    PRIMARY KEY (sentence_id, entity_id),
    CONSTRAINT fk_se_sentence
        FOREIGN KEY (sentence_id) REFERENCES sentences(id) ON DELETE CASCADE
);

-- ============================================================================
-- FIRST-CLASS ENTITIES: Speakers and Bills
-- ============================================================================

CREATE TABLE IF NOT EXISTS speakers (
    id TEXT PRIMARY KEY,
    normalized_name TEXT NOT NULL UNIQUE,
    full_name TEXT,
    title TEXT,
    position TEXT,
    constituency TEXT,
    party TEXT,
    first_appearance_date DATE,
    last_appearance_date DATE,
    total_appearances INTEGER DEFAULT 0,
    total_paragraphs INTEGER DEFAULT 0,
    bio TEXT,
    profile_url TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS bills (
    id TEXT PRIMARY KEY,
    bill_number TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    bill_type TEXT,
    status TEXT,
    introduced_date DATE,
    passed_date DATE,
    source_url TEXT,
    source_text TEXT,
    category TEXT,
    keywords TEXT[],
    entity_id TEXT UNIQUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT fk_bill_entity
        FOREIGN KEY (entity_id) REFERENCES entities(id)
);

-- ============================================================================
-- VIDEO METADATA
-- ============================================================================

CREATE TABLE IF NOT EXISTS videos (
    youtube_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    upload_date DATE,
    duration_seconds INTEGER,
    processed BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMP,
    num_paragraphs INTEGER,
    num_sentences INTEGER,
    num_speakers INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- ORDER PAPERS
-- ============================================================================

CREATE TABLE IF NOT EXISTS order_papers (
    id TEXT PRIMARY KEY,
    sitting_date DATE NOT NULL,
    order_paper_number TEXT NOT NULL,
    session TEXT,
    sitting_number TEXT,
    raw_text TEXT NOT NULL,
    parsed_json JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT uq_order_papers UNIQUE (sitting_date, order_paper_number)
);

CREATE TABLE IF NOT EXISTS order_paper_items (
    id TEXT PRIMARY KEY,
    order_paper_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    item_type TEXT NOT NULL,
    title TEXT NOT NULL,
    mover TEXT,
    action TEXT,
    status_text TEXT,
    linked_bill_id TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT fk_op_items_order_paper
        FOREIGN KEY (order_paper_id) REFERENCES order_papers(id) ON DELETE CASCADE,
    CONSTRAINT fk_op_items_bill
        FOREIGN KEY (linked_bill_id) REFERENCES bills(id)
);

-- ============================================================================
-- SESSION-SCOPED SPEAKER ROLES (multiple roles per video)
-- ============================================================================

CREATE TABLE IF NOT EXISTS speaker_video_roles (
    youtube_video_id TEXT NOT NULL,
    speaker_id TEXT,
    speaker_name_raw TEXT NOT NULL,
    speaker_name_norm TEXT NOT NULL,
    role_label TEXT NOT NULL,
    role_label_norm TEXT NOT NULL,
    role_kind TEXT NOT NULL,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL DEFAULT '',
    confidence FLOAT,
    evidence TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT uq_speaker_video_role UNIQUE (
        youtube_video_id,
        speaker_name_norm,
        role_kind,
        role_label_norm,
        source,
        source_id
    )
);

-- ============================================================================
-- VIDEO ↔ ORDER PAPER MATCH DECISIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS video_order_paper_matches (
    youtube_video_id TEXT PRIMARY KEY,
    order_paper_id TEXT,
    score FLOAT NOT NULL DEFAULT 0,
    confidence TEXT NOT NULL,
    status TEXT NOT NULL,
    reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    candidate_scores JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT fk_vopm_order_paper
        FOREIGN KEY (order_paper_id) REFERENCES order_papers(id) ON DELETE SET NULL,
    CONSTRAINT chk_vopm_confidence
        CHECK (confidence IN ('high', 'medium', 'low')),
    CONSTRAINT chk_vopm_status
        CHECK (status IN ('auto_matched', 'needs_review', 'manual_override'))
);

-- ============================================================================
-- CANONICAL KNOWLEDGE GRAPH (LLM-FIRST)
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

CREATE TABLE IF NOT EXISTS kg_aliases (
    alias_norm TEXT PRIMARY KEY,
    alias_raw TEXT NOT NULL,
    node_id TEXT NOT NULL,
    type TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence FLOAT,
    CONSTRAINT fk_kg_aliases_node
        FOREIGN KEY (node_id) REFERENCES kg_nodes(id) ON DELETE CASCADE
);

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
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT fk_kg_edges_source
        FOREIGN KEY (source_id) REFERENCES kg_nodes(id) ON DELETE CASCADE,
    CONSTRAINT fk_kg_edges_target
        FOREIGN KEY (target_id) REFERENCES kg_nodes(id) ON DELETE CASCADE
);

-- ============================================================================
-- INDEXES: Paragraphs (vector search)
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_paragraphs_embedding ON paragraphs
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_paragraphs_tsv ON paragraphs
    USING gin (tsv);

CREATE INDEX IF NOT EXISTS idx_paragraphs_speaker ON paragraphs(speaker_id);

CREATE INDEX IF NOT EXISTS idx_paragraphs_video ON paragraphs(youtube_video_id);

CREATE INDEX IF NOT EXISTS idx_paragraphs_start_time ON paragraphs(start_seconds);

-- ============================================================================
-- INDEXES: Entities (vector search)
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_entities_embedding ON entities
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_entities_tsv ON entities
    USING gin (tsv);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(normalized_name);

-- ============================================================================
-- INDEXES: Sentences (BM25 only)
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_sentences_tsv ON sentences
    USING gin (tsv);

CREATE INDEX IF NOT EXISTS idx_sentences_paragraph ON sentences(paragraph_id);

CREATE INDEX IF NOT EXISTS idx_sentences_speaker ON sentences(speaker_id);

CREATE INDEX IF NOT EXISTS idx_sentences_timestamp ON sentences(seconds_since_start);

CREATE INDEX IF NOT EXISTS idx_sentences_video ON sentences(youtube_video_id);

-- ============================================================================
-- INDEXES: Relationships
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_pe_entity ON paragraph_entities(entity_id);

CREATE INDEX IF NOT EXISTS idx_pe_type ON paragraph_entities(entity_type);

CREATE INDEX IF NOT EXISTS idx_pe_relationship ON paragraph_entities(relationship_type);

CREATE INDEX IF NOT EXISTS idx_pe_paragraph ON paragraph_entities(paragraph_id);

CREATE INDEX IF NOT EXISTS idx_se_entity ON sentence_entities(entity_id);

CREATE INDEX IF NOT EXISTS idx_se_type ON sentence_entities(entity_type);

CREATE INDEX IF NOT EXISTS idx_se_relationship ON sentence_entities(relationship_type);

CREATE INDEX IF NOT EXISTS idx_se_sentence ON sentence_entities(sentence_id);

-- ============================================================================
-- INDEXES: Speakers and Bills
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_speakers_name ON speakers(normalized_name);

CREATE INDEX IF NOT EXISTS idx_speakers_party ON speakers(party);

CREATE INDEX IF NOT EXISTS idx_speakers_appearance ON speakers(first_appearance_date);

CREATE INDEX IF NOT EXISTS idx_bills_number ON bills(bill_number);

CREATE INDEX IF NOT EXISTS idx_bills_status ON bills(status);

CREATE INDEX IF NOT EXISTS idx_bills_category ON bills(category);

CREATE INDEX IF NOT EXISTS idx_bills_introduced ON bills(introduced_date);

-- ============================================================================
-- INDEXES: Videos
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_videos_processed ON videos(processed);

CREATE INDEX IF NOT EXISTS idx_videos_upload_date ON videos(upload_date);

-- ============================================================================
-- INDEXES: Order Papers
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_order_papers_date ON order_papers(sitting_date);
CREATE INDEX IF NOT EXISTS idx_order_papers_number ON order_papers(order_paper_number);
CREATE INDEX IF NOT EXISTS idx_order_paper_items_order_paper ON order_paper_items(order_paper_id);
CREATE INDEX IF NOT EXISTS idx_order_paper_items_type ON order_paper_items(item_type);

-- ============================================================================
-- INDEXES: Session-Scoped Speaker Roles
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_speaker_video_roles_video ON speaker_video_roles(youtube_video_id);
CREATE INDEX IF NOT EXISTS idx_speaker_video_roles_speaker_id ON speaker_video_roles(speaker_id);
CREATE INDEX IF NOT EXISTS idx_speaker_video_roles_speaker_name ON speaker_video_roles(speaker_name_norm);
CREATE INDEX IF NOT EXISTS idx_vopm_order_paper_id ON video_order_paper_matches(order_paper_id);
CREATE INDEX IF NOT EXISTS idx_vopm_status ON video_order_paper_matches(status);

-- ============================================================================
-- INDEXES: Canonical KG
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_kg_nodes_embedding ON kg_nodes
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_kg_nodes_type_id ON kg_nodes(type, id);

CREATE INDEX IF NOT EXISTS idx_kg_nodes_tsv ON kg_nodes
    USING gin (tsv);

CREATE INDEX IF NOT EXISTS idx_kg_nodes_label ON kg_nodes(label);

CREATE INDEX IF NOT EXISTS idx_kg_aliases_node_id ON kg_aliases(node_id);
CREATE INDEX IF NOT EXISTS idx_kg_aliases_type ON kg_aliases(type);
CREATE INDEX IF NOT EXISTS idx_kg_aliases_source ON kg_aliases(source);

CREATE INDEX IF NOT EXISTS idx_kg_edges_video_time ON kg_edges(youtube_video_id, earliest_seconds);
CREATE INDEX IF NOT EXISTS idx_kg_edges_triple ON kg_edges(source_id, predicate, target_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_source_id ON kg_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_target_id ON kg_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_run_id ON kg_edges(kg_run_id);

-- ============================================================================
-- TRIGGERS: Auto-update tsv columns
-- ============================================================================

CREATE OR REPLACE FUNCTION paragraphs_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('english', coalesce(NEW.text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER paragraphs_tsv_update BEFORE INSERT OR UPDATE ON paragraphs
    FOR EACH ROW EXECUTE FUNCTION paragraphs_tsv_trigger();

CREATE OR REPLACE FUNCTION entities_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('english', coalesce(NEW.text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER entities_tsv_update BEFORE INSERT OR UPDATE ON entities
    FOR EACH ROW EXECUTE FUNCTION entities_tsv_trigger();

CREATE OR REPLACE FUNCTION sentences_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('english', coalesce(NEW.text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER sentences_tsv_update BEFORE INSERT OR UPDATE ON sentences
    FOR EACH ROW EXECUTE FUNCTION sentences_tsv_trigger();

CREATE OR REPLACE FUNCTION kg_nodes_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector(
        'english',
        coalesce(NEW.label, '') || ' ' || coalesce(array_to_string(NEW.aliases, ' '), '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER kg_nodes_tsv_update BEFORE INSERT OR UPDATE ON kg_nodes
    FOR EACH ROW EXECUTE FUNCTION kg_nodes_tsv_trigger();

-- ============================================================================
-- CHAT THREADS & MESSAGES (for KG conversational agent)
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

CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_id ON chat_messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_thread_state_thread_id ON chat_thread_state(thread_id);

CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_id ON chat_messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_thread_state_thread_id ON chat_thread_state(thread_id);

-- ============================================================================
-- VIEWS: Common queries
-- ============================================================================

CREATE OR REPLACE VIEW paragraph_speakers AS
SELECT
    p.id AS paragraph_id,
    p.text,
    p.start_seconds,
    p.end_seconds,
    p.start_timestamp,
    p.end_timestamp,
    p.video_date,
    p.video_title,
    s.id AS speaker_id,
    s.normalized_name AS speaker_name,
    s.title AS speaker_title,
    s.position AS speaker_position,
    s.party AS speaker_party
FROM paragraphs p
JOIN speakers s ON p.speaker_id = s.id;

CREATE OR REPLACE VIEW sentence_details AS
SELECT
    s.id AS sentence_id,
    s.text,
    s.seconds_since_start,
    s.timestamp_str,
    s.sentence_order,
    s.video_date,
    s.video_title,
    p.id AS paragraph_id,
    p.start_seconds AS paragraph_start,
    p.end_seconds AS paragraph_end,
    sp.id AS speaker_id,
    sp.normalized_name AS speaker_name,
    sp.title AS speaker_title,
    sp.position AS speaker_position
FROM sentences s
JOIN paragraphs p ON s.paragraph_id = p.id
JOIN speakers sp ON s.speaker_id = sp.id;

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
