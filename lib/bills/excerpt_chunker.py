"""Bill excerpt chunking for retrievable bill text."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class BillChunk:
    bill_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int


DEFAULT_CHUNK_SIZE = 900
DEFAULT_OVERLAP = 150
MIN_CHUNK_SIZE = 100


def generate_chunk_id(bill_id: str, chunk_index: int) -> str:
    """Generate a stable ID for a bill chunk."""
    unique = f"bex_{bill_id}_{chunk_index}".encode()
    return "bex_" + hashlib.md5(unique).hexdigest()[:12]


def chunk_text(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[BillChunk]:
    """Split bill text into overlapping chunks.

    Args:
        text: Full bill text to chunk
        chunk_size: Target size for each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of BillChunk objects with text and character offsets
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [BillChunk(bill_id="", chunk_index=0, text=text, char_start=0, char_end=len(text))]

    chunks: list[BillChunk] = []
    char_start = 0
    chunk_index = 0

    while char_start < len(text):
        char_end = char_start + chunk_size

        if char_end >= len(text):
            char_end = len(text)
            chunk_text = text[char_start:]
        else:
            paragraph_break = text.rfind("\n\n", char_start, char_end)
            sentence_break = text.rfind(". ", char_start, char_end)

            best_break = -1
            for break_pos in [paragraph_break, sentence_break]:
                if break_pos > char_start + MIN_CHUNK_SIZE:
                    if best_break == -1 or break_pos > best_break:
                        best_break = break_pos

            if best_break > char_start + MIN_CHUNK_SIZE:
                char_end = best_break + 1
                chunk_text = text[char_start:char_end]
            else:
                chunk_text = text[char_start:char_end].strip()
                if chunk_text and not chunk_text.endswith((".", ")", "]", '"', "'")):
                    chunk_text += "."

        chunk_text = chunk_text.strip()
        if chunk_text and len(chunk_text) >= MIN_CHUNK_SIZE:
            chunks.append(
                BillChunk(
                    bill_id="",
                    chunk_index=chunk_index,
                    text=chunk_text,
                    char_start=char_start,
                    char_end=char_start + len(chunk_text),
                )
            )
            chunk_index += 1

        char_start = char_end - overlap
        if char_start <= (chunks[-1].char_start if chunks else 0):
            char_start = (chunks[-1].char_end if chunks else 0) + 1

        if char_start >= len(text):
            break

    return chunks


def chunk_bill_text(
    bill_id: str,
    source_text: str,
    description: str | None = None,
    title: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[BillChunk]:
    """Chunk bill source text, falling back to description or title if no source text.

    Args:
        bill_id: The bill ID for chunk IDs
        source_text: Full text of the bill (preferred)
        description: Bill description (fallback if no source_text)
        title: Bill title (fallback if no source_text or description)
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks

    Returns:
        List of BillChunk objects with bill_id set
    """
    text_to_chunk = (source_text or "").strip()

    if not text_to_chunk and description:
        text_to_chunk = (description or "").strip()

    if not text_to_chunk and title:
        text_to_chunk = (title or "").strip()

    if not text_to_chunk:
        return []

    raw_chunks = chunk_text(text_to_chunk, chunk_size, overlap)

    return [
        BillChunk(
            bill_id=bill_id,
            chunk_index=c.chunk_index,
            text=c.text,
            char_start=c.char_start,
            char_end=c.char_end,
        )
        for c in raw_chunks
    ]
