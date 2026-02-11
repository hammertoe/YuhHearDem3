"""Helpers for modeling speaker roles/positions over time."""

from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_person_name(name: str) -> str:
    """Normalize a person name for fuzzy-ish matching (lowercase, collapse whitespace)."""

    normalized = (name or "").strip().lower()
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized


def normalize_role_label(label: str) -> str:
    """Normalize a role/position label (lowercase, collapse whitespace)."""

    normalized = (label or "").strip().lower()
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized


def infer_role_kind(role_label: str) -> str:
    """Infer a coarse role kind from the label."""

    label = normalize_role_label(role_label)
    if not label:
        return "other"

    if "committee" in label or "commission" in label:
        return "committee"

    if label.startswith("member for ") or label.startswith("member of "):
        return "constituency"

    if "minister" in label or label.startswith("attorney general"):
        return "executive"

    if "speaker" in label or "chair" in label:
        return "procedural"

    if "leader of the opposition" in label or label.startswith("leader of"):
        return "parliamentary"

    if "senator" in label or label.startswith("member"):
        return "parliamentary"

    return "other"


def split_role_labels(role_label: str) -> list[str]:
    """Split a composite role string into individual role labels."""

    raw = (role_label or "").strip()
    if not raw:
        return []

    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]
