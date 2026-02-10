from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class OrderPaperItem:
    item_type: str
    title: str
    mover: str | None = None
    action: str | None = None
    status_text: str | None = None
    bill_number: str | None = None


@dataclass(frozen=True)
class OrderPaperParsed:
    session: str
    sitting_number: str
    sitting_date: str  # YYYY-MM-DD
    order_paper_number: str
    items: list[OrderPaperItem]


_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _parse_date_line(value: str) -> str:
    """Parse e.g. 'Tuesday, 6th January, 2026' to YYYY-MM-DD."""
    # Extract day, month name, year.
    m = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s*,\s*(\d{4})\b",
        value,
    )
    if not m:
        raise ValueError(f"Could not parse sitting date: {value!r}")

    day = int(m.group(1))
    month_name = m.group(2).lower()
    year = int(m.group(3))
    if month_name not in _MONTHS:
        raise ValueError(f"Unknown month name in date: {value!r}")

    month = _MONTHS[month_name]
    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_order_paper_text(text: str) -> OrderPaperParsed:
    """Parse an order paper-like text file into structured metadata and items."""
    session = ""
    sitting_number = ""
    sitting_date = ""
    order_paper_number = ""

    # Normalize whitespace for easier regex use.
    lines = [ln.rstrip() for ln in text.splitlines()]
    joined = "\n".join(lines)

    if m := re.search(r"\*\*Session:\*\*\s*(.+)", joined):
        session = m.group(1).strip()
    if m := re.search(r"\*\*Sitting Number:\*\*\s*(.+)", joined):
        sitting_number = m.group(1).strip()
    if m := re.search(r"\*\*Date:\*\*\s*(.+)", joined):
        sitting_date = _parse_date_line(m.group(1).strip())
    if m := re.search(r"\*\*Order Paper Number:\*\*\s*No\.\s*(\d+)", joined):
        order_paper_number = m.group(1).strip()

    if not sitting_date:
        raise ValueError("Order paper missing required sitting date")
    if not order_paper_number:
        raise ValueError("Order paper missing required order paper number")

    # Extract Bills section.
    items: list[OrderPaperItem] = []
    bills_section = ""
    section_match = re.search(
        r"\*\*3\.\s*Bills on the Order Paper:\*\*(.*?)(?:\n\*\*4\.|\Z)",
        joined,
        flags=re.DOTALL,
    )
    if section_match:
        bills_section = section_match.group(1)

    # Bill headings are markdown bold list items: *   **Title**
    bill_heading_re = re.compile(r"^\*\s+\*\*(.+?)\*\*\s*$", flags=re.MULTILINE)
    headings = list(bill_heading_re.finditer(bills_section))
    for idx, h in enumerate(headings):
        title = h.group(1).strip()
        start = h.end()
        end = (
            headings[idx + 1].start() if idx + 1 < len(headings) else len(bills_section)
        )
        block = bills_section[start:end]

        def field(name: str) -> str | None:
            m = re.search(rf"\b{name}:\s*(.+)", block)
            return m.group(1).strip() if m else None

        items.append(
            OrderPaperItem(
                item_type="BILL",
                title=title,
                mover=field("Mover"),
                action=field("Action"),
                status_text=field("Status"),
            )
        )

    return OrderPaperParsed(
        session=session,
        sitting_number=sitting_number,
        sitting_date=sitting_date,
        order_paper_number=order_paper_number,
        items=items,
    )
