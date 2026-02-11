"""Unit tests for trace helper functions."""

from lib.kg_agent_loop import (
    _format_contents_summary,
    _format_tool_result_summary,
    _truncate_text,
)


def test_truncate_text_should_shorten_long_text() -> None:
    text = "This is a very long text that should be truncated with an ellipsis"
    result = _truncate_text(text, max_len=30)
    assert result == "This is a very long text th..."
    assert len(result) <= 30


def test_truncate_text_should_not_modify_short_text() -> None:
    text = "Short"
    result = _truncate_text(text, max_len=30)
    assert result == "Short"


def test_truncate_text_should_handle_empty_text() -> None:
    assert _truncate_text("", 30) == ""
    assert _truncate_text(None, 30) == ""


def test_format_contents_summary_should_handle_empty_list() -> None:
    from google.genai import types

    content = types.Content(role="user", parts=None)
    result = _format_contents_summary([content])
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["parts"] == []


def test_format_tool_result_summary_should_format_structured_result() -> None:
    result = {
        "query": "water management funding",
        "hops": 1,
        "seeds": [
            {"id": "kg_water", "label": "water"},
            {"id": "kg_funding", "label": "funding"},
        ],
        "nodes": [
            {"id": "kg_water", "label": "Water"},
            {"id": "kg_funding", "label": "Funding"},
            {"id": "kg_infrastructure", "label": "Infrastructure"},
        ],
        "edges": [{"id": "edge1"}, {"id": "edge2"}],
        "citations": [
            {"utterance_id": "utt_123"},
            {"utterance_id": "utt_456"},
        ],
    }
    formatted = _format_tool_result_summary(result)
    assert "query='water management funding...'" in formatted
    assert "hops=1" in formatted
    assert "seeds_count=2" in formatted
    assert "nodes_count=3" in formatted
    assert "edges_count=2" in formatted
    assert "citations_count=2" in formatted
    assert "nodes_preview=['Water', 'Funding', 'Infrastructure']" in formatted
    assert "citations_preview=['utt_123', 'utt_456']" in formatted
