-- Session-scoped speaker roles (multiple roles per video)

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

CREATE INDEX IF NOT EXISTS idx_speaker_video_roles_video ON speaker_video_roles(youtube_video_id);
CREATE INDEX IF NOT EXISTS idx_speaker_video_roles_speaker_id ON speaker_video_roles(speaker_id);
CREATE INDEX IF NOT EXISTS idx_speaker_video_roles_speaker_name ON speaker_video_roles(speaker_name_norm);
