from __future__ import annotations


def vector_literal(vec: list[float]) -> str:
    """Convert a float list to a pgvector literal string.

    pgvector accepts a text literal like: '[0.1,0.2, ...]'.
    """

    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"
