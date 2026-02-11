-- Persist automatic order-paper matching decisions per video.

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

CREATE INDEX IF NOT EXISTS idx_vopm_order_paper_id ON video_order_paper_matches(order_paper_id);
CREATE INDEX IF NOT EXISTS idx_vopm_status ON video_order_paper_matches(status);
