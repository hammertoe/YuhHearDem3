"""ID generators for three-tier storage."""

from __future__ import annotations

import hashlib
import re
from datetime import timedelta


def generate_paragraph_id(youtube_video_id: str, start_seconds: int) -> str:
    """Generate unique paragraph ID: {youtube_id}:{start_seconds}"""
    return f"{youtube_video_id}:{start_seconds}"


def generate_sentence_id(youtube_video_id: str, seconds_since_start: int) -> str:
    """Generate unique sentence ID: {youtube_id}:{seconds_since_start}"""
    return f"{youtube_video_id}:{seconds_since_start}"


def generate_speaker_id(name: str, existing_ids: set[str] | None = None) -> str:
    """Generate speaker ID: s_{normalized_name}_{number}"""
    existing_ids = existing_ids or set()

    normalized = name.lower().strip()
    # Preserve word boundaries first, then drop punctuation.
    # Apostrophes often represent word boundaries in names (e.g. O'Bradshaw).
    normalized = normalized.replace("'", "_")
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"[^a-z0-9_]", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")

    base_id = f"s_{normalized}"

    counter = 1
    while f"{base_id}_{counter}" in existing_ids:
        counter += 1

    return f"{base_id}_{counter}"


def generate_bill_id(bill_number: str, existing_ids: set[str] | None = None) -> str:
    """Generate bill ID: L_{bill_number}_{number}"""
    existing_ids = existing_ids or set()

    normalized = re.sub(r"\s+", "_", bill_number.upper().strip())
    normalized = re.sub(r"[^A-Z0-9_]", "", normalized)
    base_id = f"L_{normalized}"
    counter = 1
    while f"{base_id}_{counter}" in existing_ids:
        counter += 1

    return f"{base_id}_{counter}"


def generate_order_paper_id(
    chamber_code: str, session_date, order_paper_number: str
) -> str:
    """Generate order paper ID: op_{chamber_code}_{YYYYMMDD}_{order_paper_number}"""
    date_str = session_date.strftime("%Y%m%d")
    normalized_number = re.sub(r"[^A-Za-z0-9_]", "", order_paper_number)
    normalized_number = re.sub(r"\s+", "_", normalized_number).strip("_")
    return f"op_{chamber_code}_{date_str}_{normalized_number}"


def generate_entity_id(text: str, entity_type: str) -> str:
    """Generate entity ID: ent_{hash} using MD5 of type:text"""
    normalized = text.strip().lower()
    unique_str = f"{entity_type}:{normalized}"
    hash_obj = hashlib.md5(unique_str.encode())
    hash_hex = hash_obj.hexdigest()[:12]
    return f"ent_{hash_hex}"


def parse_timestamp_to_seconds(timestamp: str) -> int:
    """Parse HH:MM:SS timestamp to seconds since start."""
    parts = timestamp.split(":")
    if len(parts) == 3:
        hours, mins, secs = map(int, parts)
        return hours * 3600 + mins * 60 + secs
    elif len(parts) == 2:
        mins, secs = map(int, parts)
        return mins * 60 + secs
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")


def format_seconds_to_timestamp(seconds: int) -> str:
    """Format seconds since start to HH:MM:SS."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_timedelta_to_str(td: timedelta) -> str:
    """Format timedelta to HH:MM:SS."""
    total_seconds = int(td.total_seconds())
    return format_seconds_to_timestamp(total_seconds)


def normalize_label(label: str) -> str:
    """Normalize label for KG nodes: lowercase, trim, collapse whitespace."""
    normalized = label.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def generate_kg_node_id(node_type: str, label: str) -> str:
    """Generate KG node ID: kg_<md5(type:normalized_label)>[:12]."""
    normalized = normalize_label(label)
    unique_str = f"{node_type}:{normalized}"
    hash_obj = hashlib.md5(unique_str.encode())
    hash_hex = hash_obj.hexdigest()[:12]
    return f"kg_{hash_hex}"


def generate_kg_edge_id(
    source_id: str,
    predicate: str,
    target_id: str,
    youtube_video_id: str,
    earliest_seconds: int,
    evidence: str,
) -> str:
    """Generate KG edge ID: kge_<md5(source_id|predicate|target_id|video_id|seconds|evidence_hash)>[:12]."""
    unique_str = f"{source_id}|{predicate}|{target_id}|{youtube_video_id}|{earliest_seconds}|{evidence}"
    hash_obj = hashlib.md5(unique_str.encode())
    hash_hex = hash_obj.hexdigest()[:12]
    return f"kge_{hash_hex}"
