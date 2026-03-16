"""Build sliding extraction windows from bill excerpts."""

from __future__ import annotations

from dataclasses import dataclass

from lib.db.postgres_client import PostgresClient
from lib.knowledge_graph.window_builder import ConceptWindow, Utterance


@dataclass
class BillExcerpt:
    """Represents a single bill excerpt chunk."""

    bill_id: str
    chunk_index: int
    text: str
    source_url: str | None
    page_number: int | None

    @property
    def evidence_id(self) -> str:
        return f"bill:{self.bill_id}:{self.chunk_index}"


class BillWindowBuilder:
    """Build overlapping bill windows from bill_excerpts rows."""

    DEFAULT_WINDOW_SIZE = 4
    DEFAULT_STRIDE = 2

    def __init__(self, postgres_client: PostgresClient):
        self.postgres = postgres_client

    def fetch_bill_ids(self) -> list[str]:
        """Return bill IDs that have excerpt chunks."""
        rows = self.postgres.execute_query(
            """
            SELECT DISTINCT bill_id
            FROM bill_excerpts
            ORDER BY bill_id
            """
        )
        return [str(row[0]) for row in rows if row and row[0]]

    def fetch_bill_excerpts(self, bill_id: str) -> list[BillExcerpt]:
        """Fetch ordered excerpt chunks for a bill."""
        rows = self.postgres.execute_query(
            """
            SELECT bill_id, chunk_index, text, source_url, page_number
            FROM bill_excerpts
            WHERE bill_id = %s
            ORDER BY chunk_index ASC
            """,
            (bill_id,),
        )
        return [
            BillExcerpt(
                bill_id=str(row[0]),
                chunk_index=int(row[1]),
                text=str(row[2] or ""),
                source_url=str(row[3]) if row[3] else None,
                page_number=int(row[4]) if row[4] is not None else None,
            )
            for row in rows
            if row and row[2]
        ]

    def build_bill_windows(
        self,
        *,
        bill_id: str,
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = DEFAULT_STRIDE,
    ) -> list[ConceptWindow]:
        """Build overlapping ConceptWindow objects from bill excerpt chunks."""
        excerpts = self.fetch_bill_excerpts(bill_id)
        if not excerpts:
            return []

        if len(excerpts) < window_size:
            utterances = [
                Utterance(
                    id=ex.evidence_id,
                    timestamp_str=(f"p.{ex.page_number}" if ex.page_number else None),
                    seconds_since_start=ex.chunk_index,
                    speaker_id="bill",
                    text=ex.text,
                )
                for ex in excerpts
            ]
            return [
                ConceptWindow(
                    utterances=utterances,
                    window_size=len(excerpts),
                    stride=stride,
                    window_index=0,
                )
            ]

        windows: list[ConceptWindow] = []
        start_idx = 0
        window_index = 0

        while start_idx + window_size <= len(excerpts):
            window_excerpts = excerpts[start_idx : start_idx + window_size]
            utterances = [
                Utterance(
                    id=ex.evidence_id,
                    timestamp_str=(f"p.{ex.page_number}" if ex.page_number else None),
                    seconds_since_start=ex.chunk_index,
                    speaker_id="bill",
                    text=ex.text,
                )
                for ex in window_excerpts
            ]

            windows.append(
                ConceptWindow(
                    utterances=utterances,
                    window_size=window_size,
                    stride=stride,
                    window_index=window_index,
                )
            )
            window_index += 1
            start_idx += stride

        return windows
