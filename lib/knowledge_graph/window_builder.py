"""Window builder for concept and discourse extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lib.db.postgres_client import PostgresClient
from lib.db.pgvector import vector_literal
from lib.embeddings.google_client import GoogleEmbeddingClient

DEFAULT_WINDOW_SIZE = 10
DEFAULT_STRIDE = 6
DEFAULT_CONTEXT_SIZE = 3
MIN_UTTERANCE_LENGTH = 15


@dataclass
class Utterance:
    """Represents a single utterance from the sentences table."""

    id: str
    timestamp_str: str | None
    seconds_since_start: int
    speaker_id: str
    text: str

    @classmethod
    def from_row(cls, row: tuple) -> Utterance:
        """Create Utterance from database row."""
        return cls(
            id=row[0],
            timestamp_str=row[1],
            seconds_since_start=row[2],
            speaker_id=row[3],
            text=row[4],
        )


@dataclass
class Window:
    """Base class for extraction windows."""

    utterances: list[Utterance] = field(default_factory=list)
    window_type: str = "concept"
    window_index: int = 0

    @property
    def text(self) -> str:
        """Get structured text containing utterance metadata.

        The extractor requires stable provenance. We embed utterance_id, timestamp,
        and speaker_id in the text so the LLM can cite them.
        """
        lines: list[str] = []
        for u in self.utterances:
            ts = u.timestamp_str or ""
            lines.append(
                f"[utterance_id={u.id} t={ts} speaker_id={u.speaker_id}] {u.text}"
            )
        return "\n".join(lines)

    @property
    def utterance_ids(self) -> list[str]:
        """Get list of utterance IDs."""
        return [u.id for u in self.utterances]

    @property
    def speaker_ids(self) -> list[str]:
        """Get unique speaker IDs preserving first-seen order."""
        seen: set[str] = set()
        ordered: list[str] = []
        for u in self.utterances:
            if u.speaker_id not in seen:
                ordered.append(u.speaker_id)
                seen.add(u.speaker_id)
        return ordered

    @property
    def earliest_timestamp(self) -> str | None:
        """Get earliest timestamp in window."""
        if not self.utterances:
            return None
        earliest = min(self.utterances, key=lambda u: u.seconds_since_start)
        return earliest.timestamp_str

    @property
    def earliest_seconds(self) -> int | None:
        """Get earliest seconds in window."""
        if not self.utterances:
            return None
        return min(u.seconds_since_start for u in self.utterances)


@dataclass
class ConceptWindow(Window):
    """Window for concept extraction with fixed size and overlap."""

    window_size: int = 10
    stride: int = 6
    window_index: int = 0

    def __post_init__(self):
        self.window_type = "concept"


class WindowBuilder:
    """Build concept windows from transcript utterances."""

    DEFAULT_WINDOW_SIZE = 10
    DEFAULT_STRIDE = 6

    MIN_UTTERANCE_LENGTH = 15

    NODE_TYPES = [
        "foaf:Person",
        "schema:Legislation",
        "schema:Organization",
        "schema:Place",
        "skos:Concept",
    ]

    def __init__(
        self,
        postgres_client: PostgresClient,
        embedding_client: GoogleEmbeddingClient | None = None,
    ):
        self.postgres = postgres_client
        self.embedding = embedding_client

    def fetch_utterances(self, youtube_video_id: str) -> list[Utterance]:
        """Fetch all utterances for a video from the sentences table."""
        query = """
            SELECT
                id,
                timestamp_str,
                seconds_since_start,
                speaker_id,
                text
            FROM sentences
            WHERE youtube_video_id = %s
            ORDER BY seconds_since_start ASC
        """
        rows = self.postgres.execute_query(query, (youtube_video_id,))
        return [Utterance.from_row(row) for row in rows]

    def build_concept_windows(
        self,
        utterances: list[Utterance],
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = DEFAULT_STRIDE,
        filter_short: bool = True,
    ) -> list[ConceptWindow]:
        """Build concept windows with fixed size and overlap.

        Args:
            utterances: List of all utterances for the video
            window_size: Number of utterances per window
            stride: Number of utterances to advance between windows
            filter_short: Filter out ultra-short utterances
        """
        if filter_short:
            utterances = [
                u for u in utterances if len(u.text) >= self.MIN_UTTERANCE_LENGTH
            ]

        windows = []
        start_idx = 0
        window_index = 0

        while start_idx + window_size <= len(utterances):
            window_utterances = utterances[start_idx : start_idx + window_size]

            if window_utterances:
                window = ConceptWindow(
                    utterances=window_utterances,
                    window_size=window_size,
                    stride=stride,
                    window_index=window_index,
                )
                windows.append(window)
                window_index += 1

            start_idx += stride

        return windows

    def build_all_windows(
        self,
        youtube_video_id: str,
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = DEFAULT_STRIDE,
        context_size: int = 3,
        filter_short: bool = True,
    ) -> list[ConceptWindow]:
        """Build concept windows for a video."""
        utterances = self.fetch_utterances(youtube_video_id)

        concept_windows = self.build_concept_windows(
            utterances, window_size, stride, filter_short
        )

        return concept_windows

    def get_candidate_nodes(
        self,
        window_text: str,
        speaker_ids: list[str],
        youtube_video_id: str,
        top_k: int = 25,
    ) -> list[dict[str, Any]]:
        """Retrieve candidate canonical nodes for context.

        Args:
            window_text: Text of the window
            speaker_ids: Speaker IDs in the window
            youtube_video_id: Video ID for context
            top_k: Number of vector results to retrieve
        """
        candidates = []

        if speaker_ids:
            for speaker_id in speaker_ids:
                speaker_node_id = f"speaker_{speaker_id}"
                query = """
                    SELECT id, type, label, aliases
                    FROM kg_nodes
                    WHERE id = %s
                """
                rows = self.postgres.execute_query(query, (speaker_node_id,))
                for row in rows:
                    candidates.append(
                        {
                            "id": row[0],
                            "type": row[1],
                            "label": row[2],
                            "aliases": row[3],
                        }
                    )

        vector_query = """
            SELECT id, type, label, aliases, embedding <=> (%s)::vector AS distance
            FROM kg_nodes
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> (%s)::vector
            LIMIT %s
        """

        embedding_client = self.embedding or GoogleEmbeddingClient()
        query_embedding = embedding_client.generate_query_embedding(window_text)
        rows = self.postgres.execute_query(
            vector_query,
            (vector_literal(query_embedding), vector_literal(query_embedding), top_k),
        )
        for row in rows:
            candidates.append(
                {
                    "id": row[0],
                    "type": row[1],
                    "label": row[2],
                    "aliases": row[3],
                    "distance": row[4],
                }
            )

        unique_candidates = []
        seen_ids = set()
        for c in candidates:
            if c["id"] not in seen_ids:
                unique_candidates.append(c)
                seen_ids.add(c["id"])

        # Always keep speaker nodes; then fill up to top_k with vector hits.
        speaker_node_ids = {f"speaker_{sid}" for sid in speaker_ids}
        fixed = [c for c in unique_candidates if c["id"] in speaker_node_ids]
        rest = [c for c in unique_candidates if c["id"] not in speaker_node_ids]
        return (fixed + rest)[: max(top_k, len(fixed))]

    def format_known_nodes(self, candidates: list[dict[str, Any]]) -> str:
        """Format candidate nodes as a table string for the LLM prompt."""
        lines = ["| ID | Type | Label | Aliases |"]
        lines.append("|---|---|---|---|")

        for c in candidates:
            aliases_str = ", ".join(c["aliases"][:3]) if c["aliases"] else ""
            lines.append(f"| {c['id']} | {c['type']} | {c['label']} | {aliases_str} |")

        return "\n".join(lines)
