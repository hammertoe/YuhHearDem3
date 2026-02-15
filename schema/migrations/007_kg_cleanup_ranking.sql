-- KG Cleanup Ranking Fields
-- Migration: 007_kg_cleanup_ranking.sql
-- Adds ranking and support fields for KG cleanup and deduplication

-- ============================================================================
-- KG NODES: Add ranking and merge tracking fields
-- ============================================================================

ALTER TABLE kg_nodes
    ADD COLUMN IF NOT EXISTS pagerank_score DOUBLE PRECISION NULL;

ALTER TABLE kg_nodes
    ADD COLUMN IF NOT EXISTS merge_cluster_id TEXT NULL;

ALTER TABLE kg_nodes
    ADD COLUMN IF NOT EXISTS merged_from_count INTEGER NOT NULL DEFAULT 1;

-- Indexes for new kg_nodes columns
CREATE INDEX IF NOT EXISTS idx_kg_nodes_pagerank_score ON kg_nodes(pagerank_score);

CREATE INDEX IF NOT EXISTS idx_kg_nodes_merge_cluster_id ON kg_nodes(merge_cluster_id);

-- ============================================================================
-- KG EDGES: Add support and ranking fields
-- ============================================================================

ALTER TABLE kg_edges
    ADD COLUMN IF NOT EXISTS support_count INTEGER NOT NULL DEFAULT 1;

ALTER TABLE kg_edges
    ADD COLUMN IF NOT EXISTS edge_weight DOUBLE PRECISION NULL;

ALTER TABLE kg_edges
    ADD COLUMN IF NOT EXISTS edge_rank_score DOUBLE PRECISION NULL;

-- Indexes for new kg_edges columns
CREATE INDEX IF NOT EXISTS idx_kg_edges_support_count ON kg_edges(support_count);

CREATE INDEX IF NOT EXISTS idx_kg_edges_edge_rank_score ON kg_edges(edge_rank_score);

-- ============================================================================
-- VIEW UPDATE: Refresh kg_edges_with_details to include new columns
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
    e.support_count,
    e.edge_weight,
    e.edge_rank_score,
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
