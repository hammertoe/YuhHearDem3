from __future__ import annotations

from datetime import date

from lib.order_papers.pdf_parser import OrderPaperParser


def test_parse_response_extracts_chamber() -> None:
    parser = OrderPaperParser(gemini_client=None)  # type: ignore[arg-type]

    parsed = parser._parse_response(
        {
            "session_title": "The Honourable The Senate, First Session of 2022-2027",
            "session_date": "2026-01-13",
            "sitting_number": "Sixty-Seventh Sitting",
            "chamber": "senate",
            "speakers": [{"name": "J. Doe", "title": "Hon.", "role": "Senator"}],
            "agenda_items": [{"topic_title": "Bill Item"}],
        }
    )

    assert parsed.chamber == "senate"
    assert parsed.session_date == date(2026, 1, 13)


def test_parse_response_defaults_unknown_chamber_to_house() -> None:
    parser = OrderPaperParser(gemini_client=None)  # type: ignore[arg-type]

    parsed = parser._parse_response(
        {
            "session_title": "The Honourable The House of Assembly",
            "session_date": "2026-01-13",
            "sitting_number": "One Hundredth Sitting",
            "chamber": "unknown",
            "speakers": [],
            "agenda_items": [],
        }
    )

    assert parsed.chamber == "house"
