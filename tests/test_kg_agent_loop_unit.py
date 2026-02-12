from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any


@dataclass
class _FakeFunctionCall:
    name: str
    args: dict[str, Any]


class _FakeResponse:
    def __init__(
        self,
        *,
        text: str | None,
        function_calls: list[_FakeFunctionCall] | None,
        content: Any | None = None,
    ):
        self.text = text
        self.function_calls = function_calls

        # Mirror the shape of real google-genai responses used by KGAgentLoop.
        if content is not None:

            class _Cand:
                def __init__(self, c: Any) -> None:
                    self.content = c

            self.candidates = [_Cand(content)]


class _FakeAioModels:
    def __init__(self, responses: list[_FakeResponse]):
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, *, model: str, contents: Any, config: Any):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if not self._responses:
            raise RuntimeError("No more fake responses")
        return self._responses.pop(0)


class _FakeGeminiClient:
    def __init__(self, responses: list[_FakeResponse]):
        class _Aio:
            def __init__(self, models: _FakeAioModels):
                self.models = models

        self.aio = _Aio(_FakeAioModels(responses))


class _FakePostgres:
    def execute_query(self, _sql: str, _params: Any = None):
        return []

    def execute_update(self, _sql: str, _params: Any = None):
        return None


class _FakeEmbedding:
    def generate_query_embedding(self, _query: str) -> list[float]:
        return [0.0] * 768


def test_agent_loop_runs_tool_then_answers():
    from lib.kg_agent_loop import KGAgentLoop

    tool_call = _FakeFunctionCall(
        name="kg_hybrid_graph_rag",
        args={
            "query": "water",
            "hops": 1,
            "seed_k": 3,
            "max_edges": 10,
            "max_citations": 5,
        },
    )

    responses = [
        _FakeResponse(text=None, function_calls=[tool_call]),
        _FakeResponse(
            text=json.dumps(
                {
                    "answer": "Here is what I found.",
                    "cite_utterance_ids": [],
                    "focus_node_ids": [],
                }
            ),
            function_calls=None,
        ),
    ]
    client = _FakeGeminiClient(responses)

    loop = KGAgentLoop(
        postgres=_FakePostgres(),
        embedding_client=_FakeEmbedding(),
        client=client,
        model="gemini-3-flash-preview",
        max_tool_iterations=3,
    )

    result = asyncio.run(loop.run(user_message="Tell me about water", history=[]))
    assert result["answer"] == "Here is what I found."
    assert "retrieval" in result


def test_agent_loop_preserves_model_tool_call_content_when_continuing() -> None:
    """The tool-call message should be reused, not reconstructed.

    Newer Gemini tool calling requires additional fields (e.g. thought signatures)
    on functionCall parts. The safest approach is to append the model-provided
    content object back into the conversation when continuing.
    """

    from lib.kg_agent_loop import KGAgentLoop

    tool_call = _FakeFunctionCall(
        name="kg_hybrid_graph_rag",
        args={"query": "water"},
    )

    # Use a sentinel object so this test fails if the loop reconstructs a new
    # equivalent content object instead of appending the exact model-provided
    # one (which may contain required fields like thought signatures).
    model_tool_call_content = object()

    responses = [
        _FakeResponse(
            text=None,
            function_calls=[tool_call],
            content=model_tool_call_content,
        ),
        _FakeResponse(
            text=json.dumps(
                {
                    "answer": "ok",
                    "cite_utterance_ids": [],
                    "focus_node_ids": [],
                }
            ),
            function_calls=None,
        ),
    ]

    client = _FakeGeminiClient(responses)
    loop = KGAgentLoop(
        postgres=_FakePostgres(),
        embedding_client=_FakeEmbedding(),
        client=client,
    )

    asyncio.run(loop.run(user_message="Tell me about water", history=[]))

    # Second model call should include the model-provided tool-call content.
    second_contents = client.aio.models.calls[1]["contents"]
    assert model_tool_call_content in second_contents


def test_agent_loop_uses_json_schema_without_tools_for_final_answer() -> None:
    from lib.kg_agent_loop import KGAgentLoop

    tool_call = _FakeFunctionCall(
        name="kg_hybrid_graph_rag",
        args={"query": "water"},
    )

    responses = [
        _FakeResponse(text=None, function_calls=[tool_call]),
        _FakeResponse(
            text=json.dumps(
                {
                    "answer": "ok",
                    "cite_utterance_ids": [],
                    "focus_node_ids": [],
                    "followup_questions": [],
                }
            ),
            function_calls=None,
        ),
    ]
    client = _FakeGeminiClient(responses)
    loop = KGAgentLoop(
        postgres=_FakePostgres(),
        embedding_client=_FakeEmbedding(),
        client=client,
    )

    asyncio.run(loop.run(user_message="Tell me about water", history=[]))

    second_config = client.aio.models.calls[1]["config"]
    assert getattr(second_config, "response_mime_type", None) == "application/json"
    assert getattr(second_config, "response_schema", None) is not None
    assert not getattr(second_config, "tools", None)


def test_agent_loop_strips_wuhloss_leading_interjection() -> None:
    """Avoid starting answers with filler like 'Wuhloss,'"""

    from lib.kg_agent_loop import KGAgentLoop

    responses = [
        _FakeResponse(
            text=json.dumps(
                {
                    "answer": "Wuhloss, here's what I found.",
                    "cite_utterance_ids": [],
                    "focus_node_ids": [],
                }
            ),
            function_calls=None,
        )
    ]
    client = _FakeGeminiClient(responses)
    loop = KGAgentLoop(
        postgres=_FakePostgres(),
        embedding_client=_FakeEmbedding(),
        client=client,
    )

    result = asyncio.run(loop.run(user_message="Tell me about water", history=[]))
    assert result["answer"].startswith("Wuhloss") is False


def test_agent_loop_strips_key_connections_section() -> None:
    from lib.kg_agent_loop import KGAgentLoop

    responses = [
        _FakeResponse(
            text=json.dumps(
                {
                    "answer": "Intro\n\nKey connections\n- a -> b\n- c -> d\n\nOutro",
                    "cite_utterance_ids": [],
                    "focus_node_ids": [],
                }
            ),
            function_calls=None,
        )
    ]
    client = _FakeGeminiClient(responses)
    loop = KGAgentLoop(
        postgres=_FakePostgres(),
        embedding_client=_FakeEmbedding(),
        client=client,
    )

    result = asyncio.run(loop.run(user_message="Tell me about water", history=[]))
    assert "Key connections" not in result["answer"]


def test_agent_loop_does_not_append_sources_line_when_missing_inline_links() -> None:
    from lib.kg_agent_loop import KGAgentLoop

    responses = [
        _FakeResponse(
            text=json.dumps(
                {
                    "answer": "Water policy was debated.",
                    "cite_utterance_ids": ["utt_1", "utt_2"],
                    "focus_node_ids": [],
                }
            ),
            function_calls=None,
        )
    ]
    client = _FakeGeminiClient(responses)
    loop = KGAgentLoop(
        postgres=_FakePostgres(),
        embedding_client=_FakeEmbedding(),
        client=client,
    )

    result = asyncio.run(loop.run(user_message="Tell me about water", history=[]))
    assert result["answer"] == "Water policy was debated."


def test_agent_loop_keeps_existing_inline_source_links() -> None:
    from lib.kg_agent_loop import KGAgentLoop

    answer = "Water policy was debated [1](source:utt_1)."
    responses = [
        _FakeResponse(
            text=json.dumps(
                {
                    "answer": answer,
                    "cite_utterance_ids": ["utt_1"],
                    "focus_node_ids": [],
                }
            ),
            function_calls=None,
        )
    ]
    client = _FakeGeminiClient(responses)
    loop = KGAgentLoop(
        postgres=_FakePostgres(),
        embedding_client=_FakeEmbedding(),
        client=client,
    )

    result = asyncio.run(loop.run(user_message="Tell me about water", history=[]))
    assert result["answer"] == answer


def test_agent_loop_limits_followup_questions_to_four() -> None:
    from lib.kg_agent_loop import KGAgentLoop

    responses = [
        _FakeResponse(
            text=json.dumps(
                {
                    "answer": "Water policy was debated.",
                    "cite_utterance_ids": [],
                    "focus_node_ids": [],
                    "followup_questions": [
                        "Q1",
                        "Q2",
                        "Q3",
                        "Q4",
                        "Q5",
                    ],
                }
            ),
            function_calls=None,
        )
    ]
    client = _FakeGeminiClient(responses)
    loop = KGAgentLoop(
        postgres=_FakePostgres(),
        embedding_client=_FakeEmbedding(),
        client=client,
    )

    result = asyncio.run(loop.run(user_message="Tell me about water", history=[]))
    assert result["followup_questions"] == ["Q1", "Q2", "Q3", "Q4"]


def test_agent_loop_strips_embedded_followup_questions_from_answer() -> None:
    from lib.kg_agent_loop import KGAgentLoop

    answer = (
        "Water policy was debated [1](#src:utt_1).\n\n"
        "Here are some follow-up questions you might have:\n\n"
        "What specific measures were proposed?\n"
        "Who opposed the proposal?\n"
        "What timeline was discussed?"
    )
    responses = [
        _FakeResponse(
            text=json.dumps(
                {
                    "answer": answer,
                    "cite_utterance_ids": ["utt_1"],
                    "focus_node_ids": [],
                    "followup_questions": [
                        "What specific measures were proposed?",
                        "Who opposed the proposal?",
                        "What timeline was discussed?",
                    ],
                }
            ),
            function_calls=None,
        )
    ]
    client = _FakeGeminiClient(responses)
    loop = KGAgentLoop(
        postgres=_FakePostgres(),
        embedding_client=_FakeEmbedding(),
        client=client,
    )

    result = asyncio.run(loop.run(user_message="Tell me about water", history=[]))
    assert "follow-up questions" not in result["answer"].lower()
    assert result["answer"] == "Water policy was debated [1](#src:utt_1)."


def test_extract_embedded_followups_when_model_puts_them_in_answer_text() -> None:
    from lib.kg_agent_loop import _extract_embedded_followup_questions

    answer = (
        "Main answer paragraph.\n\n"
        "Here are some follow-up questions you might have:\n"
        "- What changed in 2023?\n"
        "- Who led the debate?\n"
        "- Which bill was referenced?"
    )

    cleaned, questions = _extract_embedded_followup_questions(answer)

    assert cleaned == "Main answer paragraph."
    assert questions == [
        "What changed in 2023?",
        "Who led the debate?",
        "Which bill was referenced?",
    ]


def test_infer_citation_ids_from_bracket_numbers_uses_retrieval_order() -> None:
    from lib.kg_agent_loop import _infer_citation_ids_from_bracket_numbers

    retrieval = {
        "citations": [
            {"utterance_id": "utt_111"},
            {"utterance_id": "utt_222"},
            {"utterance_id": "utt_333"},
        ]
    }

    inferred = _infer_citation_ids_from_bracket_numbers(
        "Point one [1]. Point three [3]. Repeat [1].",
        retrieval,
    )

    assert inferred == ["utt_111", "utt_333"]


def test_infer_citation_ids_from_bracket_numbers_ignores_markdown_links() -> None:
    from lib.kg_agent_loop import _infer_citation_ids_from_bracket_numbers

    retrieval = {
        "citations": [
            {"utterance_id": "utt_111"},
            {"utterance_id": "utt_222"},
        ]
    }

    inferred = _infer_citation_ids_from_bracket_numbers(
        "Inline [1](#src:utt_999) and [2](#src:utt_888)",
        retrieval,
    )

    assert inferred == []


def test_filter_to_known_citation_ids_matches_utt_prefix_variants() -> None:
    from lib.kg_agent_loop import _filter_to_known_citation_ids

    retrieval = {
        "citations": [
            {"utterance_id": "TNXMUaNl5wg:687"},
            {"utterance_id": "AqlXNpcikR4:2161"},
        ]
    }

    filtered = _filter_to_known_citation_ids(
        ["utt_TNXMUaNl5wg:687", "AqlXNpcikR4:2161", "missing"],
        retrieval,
    )

    assert filtered == ["TNXMUaNl5wg:687", "AqlXNpcikR4:2161"]


def test_infer_citation_ids_from_src_links_parses_grouped_or_malformed_text() -> None:
    from lib.kg_agent_loop import _infer_citation_ids_from_src_links

    retrieval = {
        "citations": [
            {"utterance_id": "TNXMUaNl5wg:687"},
            {"utterance_id": "AqlXNpcikR4:2161"},
            {"utterance_id": "AqlXNpcikR4:126"},
        ]
    }

    answer = (
        "Quote [cite] (#src:utt_TNXMUaNl5wg:687, #src:utt_AqlXNpcikR4:2161), "
        "and another source: #src:utt_AqlXNpcikR4:126"
    )

    inferred = _infer_citation_ids_from_src_links(answer, retrieval)

    assert inferred == ["TNXMUaNl5wg:687", "AqlXNpcikR4:2161", "AqlXNpcikR4:126"]
