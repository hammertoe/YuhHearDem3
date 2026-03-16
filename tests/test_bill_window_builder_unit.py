from __future__ import annotations

from typing import Any


class _FakePostgres:
    def execute_query(
        self, _sql: str, _params: tuple[Any, ...] | None = None
    ) -> list[tuple[Any, ...]]:
        return [
            ("bill_a", 0, "Interpretation and definitions.", "https://example/bill-a.pdf", 1),
            ("bill_a", 1, "Licensing framework for providers.", "https://example/bill-a.pdf", 2),
            ("bill_a", 2, "Enforcement powers and penalties.", "https://example/bill-a.pdf", 3),
            ("bill_a", 3, "Appeals and compliance timelines.", "https://example/bill-a.pdf", 4),
            (
                "bill_a",
                4,
                "Commencement and transitional provisions.",
                "https://example/bill-a.pdf",
                5,
            ),
        ]


def test_bill_window_builder_should_create_overlapping_windows() -> None:
    from lib.knowledge_graph.bill_window_builder import BillWindowBuilder

    builder = BillWindowBuilder(postgres_client=_FakePostgres())
    windows = builder.build_bill_windows(
        bill_id="bill_a",
        window_size=3,
        stride=2,
    )

    assert len(windows) == 2
    assert windows[0].utterance_ids == ["bill:bill_a:0", "bill:bill_a:1", "bill:bill_a:2"]
    assert windows[1].utterance_ids == ["bill:bill_a:2", "bill:bill_a:3", "bill:bill_a:4"]
