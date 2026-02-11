from __future__ import annotations


def test_extract_speaker_roles_from_pdf_style_parsed_json() -> None:
    from lib.order_papers.role_extract import extract_speaker_roles

    parsed_json = {
        "session_title": "HOUSE SITTING",
        "speakers": [
            {
                "name": "Hon Jane Doe",
                "title": "MP",
                "role": "Member for St James North",
            },
            {"name": "Hon Jane Doe", "title": "MP", "role": "Minister for Education"},
            {"name": "", "role": "Minister"},
            {"name": "Hon Someone", "role": ""},
        ],
        "agenda_items": [],
    }

    assert extract_speaker_roles(parsed_json) == [
        ("Hon Jane Doe", "Member for St James North"),
        ("Hon Jane Doe", "Minister for Education"),
    ]


def test_extract_speaker_roles_returns_empty_for_nonmatching_schema() -> None:
    from lib.order_papers.role_extract import extract_speaker_roles

    parsed_json = {
        "session": "...",
        "items": [],
    }

    assert extract_speaker_roles(parsed_json) == []
