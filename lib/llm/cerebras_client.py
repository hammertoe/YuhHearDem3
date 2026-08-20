"""Cerebras LLM client wrapper.

Provides an OpenAI-compatible chat interface around the `cerebras-cloud-sdk`,
normalising responses so callers (e.g. the KG agent loop) don't need to depend
on Cerebras/OpenAI shapes directly. Supports function/tool calling, structured
JSON outputs, and the same surface used by the previous Gemini client.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMToolCall:
    name: str
    arguments: dict[str, Any]
    id: str | None = None


@dataclass
class LLMResponse:
    """Normalised completion returned by `CerebrasClient`."""

    text: str | None = None
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    reasoning: str | None = None
    raw: Any = None


class CerebrasClient:
    """Wrapper around the Cerebras Cloud SDK chat.completions API."""

    DEFAULT_MODEL = "gemma-4-31b"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        from cerebras.cloud.sdk import Cerebras  # type: ignore[import-untyped]

        self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        if not self.api_key:
            raise ValueError("CEREBRAS_API_KEY must be provided or set in environment")
        self.model = model or os.getenv("LLM_MODEL", self.DEFAULT_MODEL)
        self._client = Cerebras(api_key=self.api_key)

    @staticmethod
    def _normalise_tool_calls(message: Any) -> list[LLMToolCall]:
        raw_calls = getattr(message, "tool_calls", None) or []
        out: list[LLMToolCall] = []
        for call in raw_calls:
            function = getattr(call, "function", None)
            if function is None:
                continue
            args_raw = getattr(function, "arguments", "") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to parse tool call arguments: %r", args_raw)
                args = {}
            out.append(
                LLMToolCall(
                    name=getattr(function, "name", "") or "",
                    arguments=args,
                    id=getattr(call, "id", None),
                )
            )
        return out

    @staticmethod
    def _normalise_response(response: Any) -> LLMResponse:
        choice = response.choices[0] if getattr(response, "choices", None) else None
        message = getattr(choice, "message", None) if choice is not None else None
        text = getattr(message, "content", None) if message is not None else None
        reasoning = getattr(message, "reasoning", None) if message is not None else None
        finish = getattr(choice, "finish_reason", None) if choice is not None else None
        return LLMResponse(
            text=text,
            tool_calls=CerebrasClient._normalise_tool_calls(message) if message is not None else [],
            finish_reason=finish,
            reasoning=reasoning,
            raw=response,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Synchronous chat completion."""
        kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if response_format:
            kwargs["response_format"] = response_format

        response = self._client.chat.completions.create(**kwargs)
        return self._normalise_response(response)

    async def achat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Async wrapper — runs the sync SDK call in a thread."""
        return await asyncio.to_thread(
            self.chat,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
