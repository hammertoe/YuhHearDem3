"""Migrate kg_edges to source-aware provenance fields."""

from __future__ import annotations

from lib.db.postgres_client import PostgresClient


def main() -> None:
    with PostgresClient() as pg:
        print("Applying kg_edges provenance migration...")

        pg.execute_update(
            """
            ALTER TABLE kg_edges
            ADD COLUMN IF NOT EXISTS source_kind TEXT;
            """
        )
        pg.execute_update(
            """
            ALTER TABLE kg_edges
            ADD COLUMN IF NOT EXISTS source_ref_id TEXT;
            """
        )
        pg.execute_update(
            """
            ALTER TABLE kg_edges
            ADD COLUMN IF NOT EXISTS evidence_ids TEXT[];
            """
        )

        pg.execute_update(
            """
            UPDATE kg_edges
            SET source_kind = 'transcript'
            WHERE source_kind IS NULL;
            """
        )
        pg.execute_update(
            """
            UPDATE kg_edges
            SET source_ref_id = youtube_video_id
            WHERE source_ref_id IS NULL;
            """
        )
        pg.execute_update(
            """
            UPDATE kg_edges
            SET evidence_ids = COALESCE(utterance_ids, '{}')
            WHERE evidence_ids IS NULL;
            """
        )

        pg.execute_update(
            """
            ALTER TABLE kg_edges
            ALTER COLUMN source_kind SET NOT NULL;
            """
        )
        pg.execute_update(
            """
            ALTER TABLE kg_edges
            ALTER COLUMN source_ref_id SET NOT NULL;
            """
        )
        pg.execute_update(
            """
            ALTER TABLE kg_edges
            ALTER COLUMN evidence_ids SET NOT NULL;
            """
        )
        pg.execute_update(
            """
            ALTER TABLE kg_edges
            ALTER COLUMN evidence_ids SET DEFAULT '{}';
            """
        )

        pg.execute_update(
            """
            ALTER TABLE kg_edges
            DROP CONSTRAINT IF EXISTS kg_edges_source_kind_check;
            """
        )
        pg.execute_update(
            """
            ALTER TABLE kg_edges
            ADD CONSTRAINT kg_edges_source_kind_check
            CHECK (source_kind IN ('transcript', 'bill'));
            """
        )

        pg.execute_update(
            """
            ALTER TABLE kg_edges
            ALTER COLUMN youtube_video_id DROP NOT NULL;
            """
        )

        pg.execute_update(
            """
            CREATE INDEX IF NOT EXISTS idx_kg_edges_source_kind_ref
            ON kg_edges(source_kind, source_ref_id);
            """
        )
        pg.execute_update(
            """
            CREATE INDEX IF NOT EXISTS idx_kg_edges_evidence_ids
            ON kg_edges USING gin (evidence_ids);
            """
        )

        print("Done.")


if __name__ == "__main__":
    main()
