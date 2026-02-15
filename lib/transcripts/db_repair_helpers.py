"""Helpers for transcript DB repair."""

from __future__ import annotations

import re


def build_temp_table_ddl(table_name: str, columns_sql: str) -> list[str]:
    """Build DROP/CREATE statements for a temp table."""
    name = str(table_name).strip()
    cols = str(columns_sql).strip()
    return [
        f"DROP TABLE IF EXISTS {name}",
        f"CREATE TEMP TABLE {name} ({cols})",
    ]


def build_temp_id(raw_id: str) -> str:
    """Build a temporary ID safe for DB updates."""
    base = re.sub(r"[^a-zA-Z0-9_]+", "_", str(raw_id or "").strip())
    return f"tmp_{base}"


def filter_sentence_mapping_by_existing(
    mapping,
    existing_ids: set[str],
):
    """Filter sentence id mapping to ids present in DB."""
    filtered_old_to_new = {
        old_id: new_id
        for old_id, new_id in mapping.old_to_new.items()
        if old_id in existing_ids and old_id != new_id
    }
    filtered_new_seconds = {
        old_id: mapping.new_seconds_by_old_id[old_id] for old_id in filtered_old_to_new
    }
    filtered_new_timestamps = {
        old_id: mapping.new_timestamp_by_old_id[old_id] for old_id in filtered_old_to_new
    }
    return type(mapping)(
        old_to_new=filtered_old_to_new,
        new_seconds_by_old_id=filtered_new_seconds,
        new_timestamp_by_old_id=filtered_new_timestamps,
        cleaned_transcripts=mapping.cleaned_transcripts,
        reasons=mapping.reasons,
    )


def filter_paragraph_updates_by_existing(
    updates,
    existing_ids: set[str],
) -> list[dict[str, object]]:
    """Filter paragraph updates to ids present in DB."""
    return [row for row in updates if row.get("old_id") in existing_ids]


def should_use_existing_order(
    existing_ids: set[str],
    mapped_old_ids: set[str],
) -> bool:
    """Decide whether to use existing DB order for mapping."""
    if not existing_ids:
        return False
    overlap = len(existing_ids.intersection(mapped_old_ids))
    return overlap < len(existing_ids)
