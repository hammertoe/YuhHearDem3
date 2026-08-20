"""Unit tests for CerebrasClient.

Mocks the cerebras-cloud-sdk so we can verify message construction, tool-call
parsing, and error handling without hitting the real API.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


class _FakeToolCall:
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.function = MagicMock()
        self.function.name = name
        self.function.arguments = arguments


class _FakeMessage:
    def __init__(self, content: str | None, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeChoice:
    def __init__(self, message: _FakeMessage, finish_reason: str = "stop"):
        self.message = message
        self.finish_reason = finish_reason


class _FakeCompletion:
    def __init__(self, choices: list[_FakeChoice]):
        self.choices = choices


class _FakeCompletions:
    def __init__(self):
        self.last_kwargs: dict = {}
        self.next_response: _FakeCompletion | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        assert self.next_response is not None
        return self.next_response


class _FakeCerebrasSDK:
    last_init_kwargs: dict = {}

    def __init__(self, **kwargs):
        self.chat = MagicMock()
        self.chat.completions = _FakeCompletions()
        _FakeCerebrasSDK.last_init_kwargs = kwargs


@pytest.fixture
def mock_cerebras_sdk(monkeypatch: pytest.MonkeyPatch) -> _FakeCerebrasSDK:
    fake_module = types.ModuleType("cerebras.cloud.sdk")
    fake_module.Cerebras = _FakeCerebrasSDK
    sys.modules["cerebras"] = types.ModuleType("cerebras")
    sys.modules["cerebras.cloud"] = types.ModuleType("cerebras.cloud")
    sys.modules["cerebras.cloud.sdk"] = fake_module
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
    return _FakeCerebrasSDK


def test_init_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    from lib.llm.cerebras_client import CerebrasClient

    with pytest.raises(ValueError):
        CerebrasClient(api_key="")


def test_init_uses_explicit_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    from lib.llm.cerebras_client import CerebrasClient

    sys.modules["cerebras"] = types.ModuleType("cerebras")
    sys.modules["cerebras.cloud"] = types.ModuleType("cerebras.cloud")
    sys.modules["cerebras.cloud.sdk"] = types.ModuleType("cerebras.cloud.sdk")
    sys.modules["cerebras.cloud.sdk"].Cerebras = _FakeCerebrasSDK
    client = CerebrasClient(api_key="explicit-key")
    assert client.api_key == "explicit-key"


def test_chat_passes_messages_and_temperature(mock_cerebras_sdk) -> None:
    from lib.llm.cerebras_client import CerebrasClient

    client = CerebrasClient()
    completions = client._client.chat.completions  # type: ignore[attr-defined]
    completions.next_response = _FakeCompletion([_FakeChoice(_FakeMessage("hi"))])

    response = client.chat(
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.5,
        max_tokens=128,
    )
    assert response.text == "hi"
    assert response.tool_calls == []
    assert completions.last_kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert completions.last_kwargs["temperature"] == 0.5
    assert completions.last_kwargs["max_completion_tokens"] == 128


def test_chat_parses_tool_calls(mock_cerebras_sdk) -> None:
    from lib.llm.cerebras_client import CerebrasClient

    client = CerebrasClient()
    completions = client._client.chat.completions  # type: ignore[attr-defined]
    completions.next_response = _FakeCompletion(
        [
            _FakeChoice(
                _FakeMessage(
                    None,
                    tool_calls=[
                        _FakeToolCall(
                            "call_1",
                            "search",
                            '{"q": "barbados"}',
                        )
                    ],
                )
            )
        ]
    )

    response = client.chat(
        messages=[{"role": "user", "content": "go"}],
        tools=[{"type": "function", "function": {"name": "search"}}],
        tool_choice="auto",
    )
    assert response.tool_calls[0].name == "search"
    assert response.tool_calls[0].arguments == {"q": "barbados"}
    assert response.tool_calls[0].id == "call_1"


def test_chat_handles_malformed_tool_arguments(mock_cerebras_sdk) -> None:
    from lib.llm.cerebras_client import CerebrasClient

    client = CerebrasClient()
    completions = client._client.chat.completions  # type: ignore[attr-defined]
    completions.next_response = _FakeCompletion(
        [
            _FakeChoice(
                _FakeMessage(
                    None,
                    tool_calls=[_FakeToolCall("c1", "bad_tool", "{not-json")],
                )
            )
        ]
    )

    response = client.chat(messages=[{"role": "user", "content": "x"}])
    assert response.tool_calls[0].arguments == {}
