from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class _Speaker:
    name: str
    title: str | None = None
    role: str | None = None


@dataclass
class _AgendaItem:
    topic_title: str
    primary_speaker: str | None = None
    description: str | None = None


@dataclass
class _ParsedOrderPaper:
    session_title: str
    session_date: date
    sitting_number: str | None
    speakers: list[_Speaker]
    agenda_items: list[_AgendaItem]


def test_ingest_order_paper_pdf_uses_execute_update_for_inserts(
    tmp_path, monkeypatch
) -> None:
    from scripts import ingest_order_paper_pdf as mod

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%\n")

    parsed_paper = _ParsedOrderPaper(
        session_title="THE HONOURABLE THE HOUSE OF ASSEMBLY, FIRST SESSION OF 2022-2027",
        session_date=date(2026, 1, 13),
        sitting_number="ONE HUNDRED AND TWENTY-SIXTH SITTING",
        speakers=[_Speaker(name="A. Speaker")],
        agenda_items=[
            _AgendaItem(topic_title="Item 1", primary_speaker="Mover 1"),
            _AgendaItem(topic_title="Item 2", primary_speaker="Mover 2"),
        ],
    )

    class _FakeGeminiClient:
        pass

    class _FakeOrderPaperParser:
        def __init__(self, gemini_client: _FakeGeminiClient) -> None:
            self._gemini_client = gemini_client

        def parse(self, pdf_path: str):
            return parsed_paper

    class _FakePostgres:
        def __init__(self) -> None:
            self.update_calls: list[tuple[str, tuple | None]] = []

        def execute_update(self, query: str, params: tuple | None = None) -> int:
            self.update_calls.append((query, params))
            return 1

        def execute_query(self, query: str, params: tuple | None = None):
            raise AssertionError("execute_query should not be used for INSERT/UPSERT")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    fake_postgres = _FakePostgres()

    monkeypatch.setattr(mod, "GeminiClient", _FakeGeminiClient)
    monkeypatch.setattr(mod, "OrderPaperParser", _FakeOrderPaperParser)
    monkeypatch.setattr(mod, "PostgresClient", lambda: fake_postgres)

    order_paper_id = mod.ingest_order_paper(str(pdf_path), chamber="house")

    assert order_paper_id.startswith("op_h_20260113_")
    assert len(fake_postgres.update_calls) == 1 + len(parsed_paper.agenda_items)
