"""Helpers for Gemini finish-reason handling."""

from __future__ import annotations

from typing import Any


RETRYABLE_FINISH_REASONS = {"RECITATION"}


class RetryableFinishReasonError(RuntimeError):
    """Raised when model output ended with a retryable finish reason."""


def normalize_finish_reason_name(finish_reason: Any) -> str:
    """Normalize finish reason to an uppercase string."""
    if finish_reason is None:
        return ""

    if hasattr(finish_reason, "name"):
        return str(getattr(finish_reason, "name")).upper()

    if hasattr(finish_reason, "value"):
        return str(getattr(finish_reason, "value")).upper()

    value = str(finish_reason).strip()
    if value.startswith("FinishReason."):
        value = value.split(".", 1)[1]
    return value.upper()


def is_retryable_finish_reason(finish_reason: Any) -> bool:
    """Return True when the finish reason should trigger a retry."""
    return normalize_finish_reason_name(finish_reason) in RETRYABLE_FINISH_REASONS


def raise_if_retryable_finish_reason(response: Any) -> None:
    """Raise when response has retryable finish reason on first candidate."""
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return

    finish_reason = getattr(candidates[0], "finish_reason", None)
    if not is_retryable_finish_reason(finish_reason):
        return

    finish_reason_name = normalize_finish_reason_name(finish_reason)
    raise RetryableFinishReasonError(
        f"Retryable finish_reason encountered: {finish_reason_name}"
    )
