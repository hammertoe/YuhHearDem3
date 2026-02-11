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
