from __future__ import annotations

from datetime import date

from lib.order_papers.video_matcher import (
    MatchConfidence,
    MatchStatus,
    match_order_paper_for_video,
    match_order_paper_for_video_metadata,
)


class _FakePostgres:
    def __init__(
        self,
        *,
        video_row: tuple | None,
        video_speakers: list[tuple],
        order_papers: list[tuple],
    ) -> None:
        self.video_row = video_row
        self.video_speakers = video_speakers
        self.order_papers = order_papers
        self.updates: list[tuple[str, tuple]] = []

    def execute_query(self, query: str, params: tuple | None = None):
        if "FROM videos" in query:
            if self.video_row is None:
                return []
            return [self.video_row]
        if "FROM sentences s" in query:
            return self.video_speakers
        if "FROM order_papers" in query:
            return self.order_papers
        raise AssertionError(f"Unexpected query: {query}")

    def execute_update(self, query: str, params: tuple | None = None) -> int:
        self.updates.append((query, params or tuple()))
        return 1


def test_match_order_paper_for_video_auto_matches_high_confidence() -> None:
    postgres = _FakePostgres(
        video_row=(
            "The Honourable The House - Tuesday 13th January, 2026 - Part 1",
            date(2026, 1, 13),
        ),
        video_speakers=[("Hon Jane Doe",), ("Mark Brown",)],
        order_papers=[
            (
                "op_h_20260113_one",
                date(2026, 1, 13),
                "House of Assembly Session",
                {
                    "speakers": [
                        {"name": "Hon Jane Doe"},
                        {"name": "Mark Brown"},
                    ]
                },
            )
        ],
    )

    decision = match_order_paper_for_video(
        postgres, youtube_video_id="vid_1", persist=True
    )

    assert decision.status == MatchStatus.AUTO_MATCHED
    assert decision.confidence == MatchConfidence.HIGH
    assert decision.order_paper_id == "op_h_20260113_one"
    assert postgres.updates


def test_match_order_paper_for_video_marks_needs_review_when_low_confidence() -> None:
    postgres = _FakePostgres(
        video_row=("Committee Feed", date(2026, 1, 25)),
        video_speakers=[("Unknown Speaker",)],
        order_papers=[
            (
                "op_h_20260110_one",
                date(2026, 1, 10),
                "House of Assembly Session",
                {"speakers": [{"name": "Different Person"}]},
            )
        ],
    )

    decision = match_order_paper_for_video(
        postgres, youtube_video_id="vid_2", persist=False
    )

    assert decision.status == MatchStatus.NEEDS_REVIEW
    assert decision.confidence == MatchConfidence.LOW
    assert decision.order_paper_id == "op_h_20260110_one"
    assert not postgres.updates


def test_match_order_paper_for_video_metadata_without_videos_row() -> None:
    postgres = _FakePostgres(
        video_row=None,
        video_speakers=[],
        order_papers=[
            (
                "op_h_20260113_one",
                date(2026, 1, 13),
                "House of Assembly Session",
                {"speakers": []},
            )
        ],
    )

    decision = match_order_paper_for_video_metadata(
        postgres,
        youtube_video_id="brand_new_video",
        video_title="The Honourable The House - Tuesday 13th January, 2026 - Part 1",
        upload_date=None,
        persist=False,
    )

    assert decision.status == MatchStatus.AUTO_MATCHED
    assert decision.confidence == MatchConfidence.HIGH
    assert decision.order_paper_id == "op_h_20260113_one"
