"""KG cleanup normalization functions."""

import re
import unicodedata


def normalize_label(label: str) -> str:
    """Normalize label: lowercase, trim, collapse whitespace."""
    if not label:
        return ""
    normalized = unicodedata.normalize("NFKD", label)
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = normalized.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def normalize_for_matching(label: str) -> str:
    """Normalize label for matching: Unicode fold, lowercase, collapse whitespace."""
    if not label:
        return ""
    normalized = unicodedata.normalize("NFKD", label)
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = normalized.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def generate_norm_key(label: str) -> str:
    """Generate matching key using normalization for matching."""
    if not label:
        return ""
    normalized = normalize_for_matching(label)
    return normalized


def strip_honorifics(name: str) -> str:
    """Strip honorifics and suffixes from a name."""
    if not name:
        return ""
    name = name.strip()

    honorifics = [
        r"^the\s+(most\s+)?honourable\s+",
        r"^hon\.?\s+",
        r"^mr\.?\s+",
        r"^ms\.?\s+",
        r"^mrs\.?\s+",
        r"^dr\.?\s+",
        r"^senator\s+",
    ]

    for honorific in honorifics:
        name = re.sub(honorific, "", name, flags=re.IGNORECASE).strip()

    suffixes = [
        r"\s+M\.P\.$",
        r"\s+MP$",
        r"\s+Mp$",
        r"\s+Jr\.?$",
        r"\s+Sr\.?$",
        r"\s+Esq\.?$",
    ]

    for suffix in suffixes:
        name = re.sub(suffix, "", name, flags=re.IGNORECASE).strip()

    return name


def extract_surname(name: str) -> str:
    """Extract surname from a full name."""
    if not name:
        return ""
    name = name.strip()

    if "," in name:
        parts = name.split(",")
        return parts[0].strip()

    parts = name.split()
    return parts[-1] if parts else ""


def extract_initials(name: str) -> str:
    """Extract initials from a name."""
    if not name:
        return ""
    name = name.strip()

    parts = name.split()
    initials: list[str] = []

    for part in parts:
        part_stripped = part.strip()
        if not part_stripped:
            continue

        if "." in part_stripped:
            initials.append(part_stripped.replace(".", "").upper())
        elif len(part_stripped) == 1 and part_stripped.isalpha() and part_stripped.isupper():
            initials.append(part_stripped.upper())
        elif len(part_stripped) == 2 and part_stripped.isalpha() and part_stripped[1].isupper():
            initials.append(part_stripped[0].upper())

    return "".join(initials)


def normalize_legislation_key(title: str) -> str:
    """Normalize legislation title by removing year."""
    if not title:
        return ""
    normalized = title.strip()
    normalized = re.sub(r"\b\d{4}\b", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalize_label(normalized)
