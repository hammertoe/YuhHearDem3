"""DB repair helper tests."""

from lib.transcripts.db_repair_helpers import (
    build_temp_id,
    build_temp_table_ddl,
    filter_paragraph_updates_by_existing,
    filter_sentence_mapping_by_existing,
    should_use_existing_order,
)
from lib.transcripts.timestamp_fix import SentenceIdMapping


def test_build_temp_table_ddl_includes_drop_and_create() -> None:
    """Ensures temp table DDL includes drop and create statements."""
    ddl = build_temp_table_ddl("sentence_id_map", "old_id text")
    assert ddl == [
        "DROP TABLE IF EXISTS sentence_id_map",
        "CREATE TEMP TABLE sentence_id_map (old_id text)",
    ]


def test_build_temp_id_prefixes_and_sanitizes() -> None:
    """Builds a temp id with a stable prefix and safe chars."""
    temp_id = build_temp_id("abc:123_2")
    assert temp_id.startswith("tmp_")
    assert ":" not in temp_id


def test_filter_sentence_mapping_by_existing_ids() -> None:
    """Filters sentence mappings to ids present in DB."""
    mapping = SentenceIdMapping(
        old_to_new={"vid:1": "vid:1", "vid:2": "vid:20"},
        new_seconds_by_old_id={"vid:1": 1, "vid:2": 20},
        new_timestamp_by_old_id={"vid:1": "00:00:01", "vid:2": "00:00:20"},
        cleaned_transcripts=[],
        reasons=[],
    )
    filtered = filter_sentence_mapping_by_existing(mapping, {"vid:2"})
    assert list(filtered.old_to_new.keys()) == ["vid:2"]


def test_filter_paragraph_updates_by_existing_ids() -> None:
    """Filters paragraph updates to ids present in DB."""
    updates: list[dict[str, object]] = [
        {"old_id": "vid:1", "new_id": "vid:1"},
        {"old_id": "vid:2", "new_id": "vid:20"},
    ]
    filtered = filter_paragraph_updates_by_existing(updates, {"vid:2"})
    assert filtered == [{"old_id": "vid:2", "new_id": "vid:20"}]


def test_should_use_existing_order_when_overlap_missing() -> None:
    """Uses existing ordering when mapped ids don't cover DB ids."""
    assert should_use_existing_order({"a", "b"}, {"a"}) is True
    assert should_use_existing_order({"a", "b"}, {"a", "b"}) is False
