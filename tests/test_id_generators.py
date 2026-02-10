"""ID generator tests."""

import pytest
from lib.id_generators import (
    generate_paragraph_id,
    generate_sentence_id,
    generate_speaker_id,
    generate_bill_id,
    generate_entity_id,
    parse_timestamp_to_seconds,
    format_seconds_to_timestamp,
)


def test_generate_paragraph_id():
    """Test paragraph ID generation."""
    paragraph_id = generate_paragraph_id("Syxyah7QIaM", 1234)
    assert paragraph_id == "Syxyah7QIaM:1234"
    print("✅ Paragraph ID generation works")


def test_generate_sentence_id():
    """Test sentence ID generation."""
    sentence_id = generate_sentence_id("Syxyah7QIaM", 5678)
    assert sentence_id == "Syxyah7QIaM:5678"
    print("✅ Sentence ID generation works")


def test_generate_speaker_id():
    """Test speaker ID generation."""
    speaker_id = generate_speaker_id("Hon Santia Bradshaw", set())
    assert speaker_id == "s_hon_santia_bradshaw_1"
    print(f"✅ Speaker ID generation works: {speaker_id}")


def test_generate_speaker_id_duplicates():
    """Test speaker ID deduplication."""
    existing = {"s_hon_santia_bradshaw_1"}
    speaker_id = generate_speaker_id("Hon Santia Bradshaw", existing)
    assert speaker_id == "s_hon_santia_bradshaw_2"
    print("✅ Speaker ID deduplication works")


def test_generate_bill_id():
    """Test bill ID generation."""
    bill_id = generate_bill_id("HR 1234", set())
    assert bill_id == "L_HR_1234_1"
    print(f"✅ Bill ID generation works: {bill_id}")


def test_generate_entity_id():
    """Test entity ID generation."""
    entity_id = generate_entity_id("Road Traffic Act", "BILL")
    assert entity_id.startswith("ent_")
    assert len(entity_id) == 16
    print(f"✅ Entity ID generation works: {entity_id}")


def test_parse_timestamp_to_seconds():
    """Test timestamp parsing."""
    seconds = parse_timestamp_to_seconds("00:36:34")
    assert seconds == 2194

    seconds = parse_timestamp_to_seconds("01:30:15")
    assert seconds == 5415
    print("✅ Timestamp parsing works")


def test_format_seconds_to_timestamp():
    """Test timestamp formatting."""
    timestamp = format_seconds_to_timestamp(2194)
    assert timestamp == "00:36:34"

    timestamp = format_seconds_to_timestamp(5415)
    assert timestamp == "01:30:15"
    print("✅ Timestamp formatting works")


def test_timestamp_roundtrip():
    """Test timestamp conversion roundtrip."""
    original = "00:45:12"
    seconds = parse_timestamp_to_seconds(original)
    formatted = format_seconds_to_timestamp(seconds)
    assert original == formatted
    print("✅ Timestamp roundtrip works")


def test_generate_speaker_id_special_chars():
    """Test speaker ID with special characters."""
    speaker_id = generate_speaker_id("Santia J. O'Bradshaw", set())
    assert "santia_j_o_bradshaw" in speaker_id
    assert "'" not in speaker_id
    assert "." not in speaker_id
    print("✅ Speaker ID special character handling works")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
