-- Add predicate_raw to canonical KG edges and refresh view.

ALTER TABLE kg_edges
    ADD COLUMN IF NOT EXISTS predicate_raw TEXT;

-- If an older version of the view exists without predicate_raw, CREATE OR REPLACE
-- can fail because it is not allowed to change the view's column set.
DROP VIEW IF EXISTS kg_edges_with_details;

CREATE VIEW kg_edges_with_details AS
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
