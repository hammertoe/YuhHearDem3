from __future__ import annotations

from lib.order_papers.parser import parse_order_paper_text


def test_parse_order_paper_extracts_date_and_number() -> None:
    text = """
### Barbados House of Assembly - Session Context Information

**1. General Session Details:**
*   **Session:** First Session of 2022–2027
*   **Sitting Number:** One Hundred and Twenty-Fifth Sitting
*   **Date:** Tuesday, 6th January, 2026
*   **Order Paper Number:** No. 125

**3. Bills on the Order Paper:**

*   **Barbados Citizenship Bill, 2025**
    *   Mover: Hon. W. A. Abrahams
    *   Action: Second Reading
    *   Status: Notice given 8th Aug, 2025.
""".strip()

    parsed = parse_order_paper_text(text)

    assert parsed.sitting_date == "2026-01-06"
    assert parsed.order_paper_number == "125"
    assert parsed.session.startswith("First Session")
    assert len(parsed.items) == 1
    assert parsed.items[0].item_type == "BILL"
    assert "Barbados Citizenship Bill" in parsed.items[0].title
