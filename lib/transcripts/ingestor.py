from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import spacy

from lib.db.postgres_client import PostgresClient
from lib.embeddings.google_client import GoogleEmbeddingClient
from lib.id_generators import generate_entity_id
from lib.processors.paragraph_splitter import (
    group_transcripts_into_paragraphs,
    split_paragraph_into_sentences,
)


def _format_vector(embedding: list[float]) -> str:
    # pgvector accepts text format like '[0.1,0.2,...]'.
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


@dataclass(frozen=True)
class IngestStats:
    paragraphs: int
    sentences: int
    speakers: int
    entities: int
    sentence_entities: int
    paragraph_entities: int


class TranscriptIngestor:
    def __init__(
        self,
        postgres: PostgresClient | None = None,
        embedding_client: GoogleEmbeddingClient | None = None,
        nlp: Any | None = None,
    ):
        self.postgres = postgres or PostgresClient()
        self.embedding_client = embedding_client or GoogleEmbeddingClient()
        self.nlp = nlp or spacy.load("en_core_web_md")

    def ingest_transcript_json(
        self,
        transcript_data: dict[str, Any],
        youtube_video_id: str,
        *,
        embed_paragraphs: bool = True,
        embed_entities: bool = True,
    ) -> IngestStats:
        video_metadata = transcript_data.get("video_metadata") or {}
        title = video_metadata.get("title", "")
        upload_date_raw = (video_metadata.get("upload_date", "") or "").strip()
        upload_date = upload_date_raw[:8] if len(upload_date_raw) >= 8 else ""

        self._upsert_video(
            youtube_video_id=youtube_video_id,
            title=title,
            upload_date=upload_date,
            num_sentences=len(transcript_data.get("transcripts", []) or []),
            num_speakers=len(transcript_data.get("speakers", []) or []),
        )

        speakers = transcript_data.get("speakers", []) or []
        for s in speakers:
            self._upsert_speaker(s)

        transcripts = transcript_data.get("transcripts", []) or []
        paragraphs = group_transcripts_into_paragraphs(youtube_video_id, transcripts)
        paragraph_texts = [p.get_text() for p in paragraphs]

        paragraph_embeddings: list[list[float]] = []
        if embed_paragraphs and paragraph_texts:
            paragraph_embeddings = self.embedding_client.generate_embeddings_batch(
                paragraph_texts
            )

        sentence_entities_count = 0
        paragraph_entities_count = 0
        entity_ids_seen: set[str] = set()
        entity_texts_by_id: dict[str, tuple[str, str]] = {}

        for idx, paragraph in enumerate(paragraphs):
            emb = paragraph_embeddings[idx] if idx < len(paragraph_embeddings) else None
            self._insert_paragraph(
                paragraph, title=title, upload_date=upload_date, embedding=emb
            )

            sentences = split_paragraph_into_sentences(
                paragraph,
                youtube_video_id=youtube_video_id,
                video_date=upload_date,
                video_title=title,
            )
            paragraph_entity_ids: set[str] = set()

            for sentence in sentences:
                self._insert_sentence(sentence)
                extracted = self._extract_entities_from_text(sentence.get("text", ""))
                for ent_text, ent_type in extracted:
                    ent_id = generate_entity_id(ent_text, ent_type)
                    if ent_id not in entity_ids_seen:
                        self._upsert_entity(ent_id, ent_text, ent_type)
                        entity_ids_seen.add(ent_id)
                        entity_texts_by_id[ent_id] = (ent_text, ent_type)
                    self._insert_sentence_entity(
                        sentence_id=sentence["id"],
                        entity_id=ent_id,
                        entity_type=ent_type,
                    )
                    sentence_entities_count += 1
                    paragraph_entity_ids.add(ent_id)

            for ent_id in paragraph_entity_ids:
                self._insert_paragraph_entity(
                    paragraph_id=paragraph.id,
                    entity_id=ent_id,
                )
                paragraph_entities_count += 1

        if embed_entities and entity_texts_by_id:
            self._embed_entities(entity_texts_by_id)

        return IngestStats(
            paragraphs=len(paragraphs),
            sentences=len(transcripts),
            speakers=len(speakers),
            entities=len(entity_ids_seen),
            sentence_entities=sentence_entities_count,
            paragraph_entities=paragraph_entities_count,
        )

    def _embed_entities(self, entity_texts_by_id: dict[str, tuple[str, str]]) -> None:
        # Embed unique entity texts and store in entities.embedding.
        entity_ids = list(entity_texts_by_id.keys())
        texts = [entity_texts_by_id[eid][0] for eid in entity_ids]
        embeddings = self.embedding_client.generate_embeddings_batch(
            texts, task_type="RETRIEVAL_DOCUMENT"
        )

        for eid, emb in zip(entity_ids, embeddings):
            vector = _format_vector(emb)
            self.postgres.execute_update(
                """
                UPDATE entities
                SET embedding = (%s)::vector, updated_at = NOW()
                WHERE id = %s
                """,
                (vector, eid),
            )

    def _upsert_video(
        self,
        youtube_video_id: str,
        title: str,
        upload_date: str,
        num_sentences: int,
        num_speakers: int,
    ) -> None:
        self.postgres.execute_update(
            """
            INSERT INTO videos (
                youtube_id, title, description, upload_date, duration_seconds,
                processed, processed_at, num_sentences, num_speakers
            )
            VALUES (%s, %s, %s, to_date(NULLIF(%s, ''), 'YYYYMMDD'), %s, %s, NOW(), %s, %s)
            ON CONFLICT (youtube_id) DO UPDATE SET
                title = EXCLUDED.title,
                processed = EXCLUDED.processed,
                processed_at = NOW(),
                num_sentences = EXCLUDED.num_sentences,
                num_speakers = EXCLUDED.num_speakers
            """,
            (
                youtube_video_id,
                title,
                "",
                upload_date,
                None,
                True,
                num_sentences,
                num_speakers,
            ),
        )

    def _upsert_speaker(self, speaker: dict[str, Any]) -> None:
        speaker_id = speaker.get("speaker_id", "")
        if not speaker_id:
            return
        self.postgres.execute_update(
            """
            INSERT INTO speakers (
                id, normalized_name, full_name, title, position,
                party, first_appearance_date, last_appearance_date
            )
            VALUES (%s, %s, %s, %s, %s, %s, NULL, NULL)
            ON CONFLICT (id) DO UPDATE SET
                normalized_name = EXCLUDED.normalized_name,
                full_name = EXCLUDED.full_name,
                position = EXCLUDED.position,
                updated_at = NOW()
            """,
            (
                speaker_id,
                speaker_id,
                speaker.get("name", "") or speaker_id,
                "",
                speaker.get("position", "") or "Unknown",
                speaker.get("party", ""),
            ),
        )

    def _insert_paragraph(
        self,
        paragraph: Any,
        title: str,
        upload_date: str,
        embedding: list[float] | None,
    ) -> None:
        vector = _format_vector(embedding) if embedding else None
        self.postgres.execute_update(
            """
            INSERT INTO paragraphs (
                id, youtube_video_id, start_seconds, end_seconds,
                text, speaker_id, voice_id, start_timestamp,
                end_timestamp, embedding, video_date, video_title,
                sentence_count
            )
            VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, (%s)::vector,
                to_date(NULLIF(%s, ''), 'YYYYMMDD'), %s,
                %s
            )
            ON CONFLICT (id) DO NOTHING
            """,
            (
                paragraph.id,
                paragraph.youtube_video_id,
                paragraph.start_seconds,
                paragraph.end_seconds,
                paragraph.get_text(),
                paragraph.speaker_id,
                paragraph.voice_id,
                paragraph.start_timestamp,
                paragraph.end_timestamp,
                vector,
                upload_date,
                title,
                len(paragraph.sentences),
            ),
        )

    def _insert_sentence(self, sentence: dict[str, Any]) -> None:
        self.postgres.execute_update(
            """
            INSERT INTO sentences (
                id, youtube_video_id, seconds_since_start, timestamp_str,
                text, speaker_id, voice_id, paragraph_id,
                sentence_order, video_date, video_title
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, to_date(NULLIF(%s, ''), 'YYYYMMDD'), %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (
                sentence["id"],
                sentence["youtube_video_id"],
                sentence["seconds_since_start"],
                sentence.get("timestamp_str"),
                sentence.get("text", ""),
                sentence.get("speaker_id", ""),
                sentence.get("voice_id"),
                sentence.get("paragraph_id", ""),
                sentence.get("sentence_order"),
                sentence.get("video_date") or "",
                sentence.get("video_title") or "",
            ),
        )

    def _extract_entities_from_text(self, text: str) -> list[tuple[str, str]]:
        doc = self.nlp(text)
        extracted: list[tuple[str, str]] = []
        for ent in getattr(doc, "ents", []):
            ent_text = str(getattr(ent, "text", "")).strip()
            ent_type = str(getattr(ent, "label_", "")).strip() or "ENTITY"
            if ent_text:
                extracted.append((ent_text, ent_type))
        return extracted

    def _upsert_entity(self, entity_id: str, text: str, entity_type: str) -> None:
        self.postgres.execute_update(
            """
            INSERT INTO entities (id, text, type)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                text = EXCLUDED.text,
                type = EXCLUDED.type,
                updated_at = NOW()
            """,
            (entity_id, text, entity_type),
        )

    def _insert_sentence_entity(
        self, sentence_id: str, entity_id: str, entity_type: str
    ) -> None:
        self.postgres.execute_update(
            """
            INSERT INTO sentence_entities (sentence_id, entity_id, entity_type, relationship_type)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (sentence_id, entity_id) DO NOTHING
            """,
            (sentence_id, entity_id, entity_type, "MENTIONS"),
        )

    def _insert_paragraph_entity(self, paragraph_id: str, entity_id: str) -> None:
        self.postgres.execute_update(
            """
            INSERT INTO paragraph_entities (paragraph_id, entity_id, entity_type, relationship_type)
            VALUES (%s, %s,
                (SELECT type FROM entities WHERE id = %s),
                %s
            )
            ON CONFLICT (paragraph_id, entity_id) DO NOTHING
            """,
            (paragraph_id, entity_id, entity_id, "MENTIONS"),
        )
