from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "label,expected",
    [
        ("Member for St James North", "constituency"),
        ("Minister for Education", "executive"),
        ("Deputy Speaker", "procedural"),
        ("Leader of the Opposition", "parliamentary"),
        ("Senator", "parliamentary"),
    ],
)
def test_infer_role_kind(label: str, expected: str) -> None:
    from lib.roles import infer_role_kind

    assert infer_role_kind(label) == expected


def test_normalize_role_label_collapses_whitespace_and_lowercases() -> None:
    from lib.roles import normalize_role_label

    assert (
        normalize_role_label("  Minister   for  Education \n")
        == "minister for education"
    )


def test_split_role_labels_splits_on_commas() -> None:
    from lib.roles import split_role_labels

    assert split_role_labels("Leader of the Opposition, Chairman of Committee") == [
        "Leader of the Opposition",
        "Chairman of Committee",
    ]
