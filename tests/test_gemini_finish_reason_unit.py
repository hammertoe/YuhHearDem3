from __future__ import annotations

import pytest

from lib.gemini_finish_reason import (
    RetryableFinishReasonError,
    is_retryable_finish_reason,
    normalize_finish_reason_name,
    raise_if_retryable_finish_reason,
)


class _Candidate:
    def __init__(self, finish_reason):
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, finish_reason):
        self.candidates = [_Candidate(finish_reason)]


def test_normalize_finish_reason_handles_enum_style_string() -> None:
    assert normalize_finish_reason_name("FinishReason.RECITATION") == "RECITATION"


def test_is_retryable_finish_reason_for_recitation() -> None:
    assert is_retryable_finish_reason("RECITATION") is True


def test_raise_if_retryable_finish_reason_raises_for_recitation() -> None:
    response = _Response("RECITATION")

    with pytest.raises(RetryableFinishReasonError):
        raise_if_retryable_finish_reason(response)


def test_raise_if_retryable_finish_reason_noop_for_stop() -> None:
    response = _Response("STOP")

    raise_if_retryable_finish_reason(response)
