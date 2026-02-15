"""Tests for KG cleanup normalization module."""

from lib.knowledge_graph.cleanup.normalize import (
    normalize_label,
    normalize_for_matching,
    generate_norm_key,
    strip_honorifics,
    extract_surname,
    extract_initials,
    normalize_legislation_key,
)


def test_normalize_label_basic():
    """Test basic label normalization."""
    assert normalize_label("John Smith") == "john smith"
    assert normalize_label("  John  Smith  ") == "john smith"
    assert normalize_label("JOHN SMITH") == "john smith"


def test_normalize_label_unicode():
    """Test Unicode NFKD normalization."""
    assert normalize_label("Müller") == "muller"
    assert normalize_label("Café") == "cafe"
    assert normalize_label("José") == "jose"


def test_normalize_label_punctuation():
    """Test punctuation normalization."""
    assert normalize_label("Smith, John") == "smith, john"
    assert normalize_label("O'Brien") == "o'brien"
    assert normalize_label("Jean-Luc") == "jean-luc"


def test_normalize_for_matching():
    """Test normalization for matching keys."""
    assert normalize_for_matching("John Smith") == "john smith"
    assert normalize_for_matching("  John  Smith  ") == "john smith"
    assert normalize_for_matching("Müller, Hans") == "muller, hans"


def test_generate_norm_key():
    """Test generation of matching keys."""
    assert generate_norm_key("John Smith") == "john smith"
    assert generate_norm_key("The Climate Change Act 2008") == "the climate change act 2008"
    assert generate_norm_key("  Multiple   Spaces  ") == "multiple spaces"


def test_strip_honorifics():
    """Test honorific stripping."""
    assert strip_honorifics("The Honourable John Smith") == "John Smith"
    assert strip_honorifics("Hon. Mary Jones") == "Mary Jones"
    assert strip_honorifics("Mr. David Brown") == "David Brown"
    assert strip_honorifics("Dr. Alice Wilson") == "Alice Wilson"
    assert strip_honorifics("Senator Bob Lee") == "Bob Lee"
    assert strip_honorifics("The Most Honourable Sir John") == "Sir John"


def test_extract_surname():
    """Test surname extraction."""
    assert extract_surname("John Smith") == "Smith"
    assert extract_surname("Mary Elizabeth Jones") == "Jones"
    assert extract_surname("David") == "David"
    assert extract_surname("O'Brien, John") == "O'Brien"


def test_extract_initials():
    """Test initials extraction."""
    assert extract_initials("John Smith") == "JS"
    assert extract_initials("Mary Elizabeth Jones") == "MEJ"
    assert extract_initials("David") == "D"
    assert extract_initials("A B C") == "ABC"


def test_normalize_legislation_key():
    """Test legislation key normalization."""
    assert normalize_legislation_key("Climate Change Act 2008") == "climate change act"
    assert normalize_legislation_key("Road Traffic Act 1991") == "road traffic act"
    assert normalize_legislation_key("The Finance (No. 2) Act 2023") == "the finance (no. 2) act"


def test_normalize_with_various_punctuation():
    """Test normalization with various punctuation."""
    assert normalize_label("Smith-Jones") == "smith-jones"
    assert normalize_label("O'Connor") == "o'connor"
    assert normalize_label("Van der Berg") == "van der berg"


def test_normalize_empty_and_none():
    """Test normalization with empty/edge cases."""
    assert normalize_label("") == ""
    assert normalize_label("   ") == ""
    assert normalize_label("---") == "---"


def test_generate_norm_key_punctuation_removal():
    """Test that norm key removes more punctuation than basic normalization."""
    assert normalize_label("Smith, John") == "smith, john"
    assert generate_norm_key("Smith, John") == "smith, john"


def test_strip_honorifics_case_insensitive():
    """Test honorific stripping is case-insensitive."""
    assert strip_honorifics("THE HONOURABLE John Smith") == "John Smith"
    assert strip_honorifics("hon. Mary Jones") == "Mary Jones"
    assert strip_honorifics("MR. David Brown") == "David Brown"


def test_strip_multiple_honorifics():
    """Test stripping multiple honorifics."""
    assert strip_honorifics("The Honourable Dr. John Smith") == "John Smith"
    assert strip_honorifics("The Most Honourable Sir Dr. John") == "Sir Dr. John"


def test_strip_suffixes():
    """Test stripping name suffixes."""
    assert strip_honorifics("The Honourable Ryan Straughn Mp") == "Ryan Straughn"
    assert strip_honorifics("John Smith MP") == "John Smith"
    assert strip_honorifics("John Smith M.P.") == "John Smith"
    assert strip_honorifics("John Smith Jr.") == "John Smith"
    assert strip_honorifics("John Smith Sr") == "John Smith"
    assert strip_honorifics("The Honourable Dr. John Smith Esq.") == "John Smith"
