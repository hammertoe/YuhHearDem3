from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from lib.kg_hybrid_graph_rag import kg_hybrid_graph_rag


AGENT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "cite_utterance_ids": {"type": "array", "items": {"type": "string"}},
        "focus_node_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "cite_utterance_ids", "focus_node_ids"],
}


def _parse_json_best_effort(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    raw = text.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        return None


_LEADING_INTERJECTION_RE = re.compile(r"^(wuh\s*-?\s*loss)[\s,!.:-]+", re.IGNORECASE)
_KEY_CONNECTIONS_BLOCK_RE = re.compile(
    r"(?ims)^\s*#{0,6}\s*key connections\s*$\n(?:^\s*[-*]\s.*\n)+\n?",
)


def _promote_section_headings(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []

    def is_blank(s: str) -> bool:
        return not s.strip()

    for i, line in enumerate(lines):
        stripped = line.strip()
        prev_blank = i == 0 or is_blank(lines[i - 1])
        next_nonblank = i + 1 < len(lines) and not is_blank(lines[i + 1])

        if (
            prev_blank
            and next_nonblank
            and stripped
            and not stripped.startswith("#")
            and not stripped.startswith(">")
            and not stripped.startswith("-")
            and not stripped.endswith(":")
            and len(stripped) <= 64
            and re.fullmatch(r"[A-Z][A-Za-z0-9 \"'\-()]+", stripped) is not None
        ):
            # Heuristic: if it's mostly Title Case words, treat as a section heading.
            words = [w for w in re.split(r"\s+", stripped) if w]
            small = {
                "and",
                "or",
                "the",
                "a",
                "an",
                "of",
                "to",
                "for",
                "in",
                "on",
                "with",
            }
            titleish = 0
            for w in words:
                lw = w.lower().strip("\"'()")
                if lw in small:
                    titleish += 1
                elif w[:1].isupper():
                    titleish += 1
            if len(words) >= 2 and (titleish / max(1, len(words))) >= 0.75:
                out.append(f"### {stripped}")
                continue

        out.append(line)

    return "\n".join(out)


def _clean_answer_text(text: str | None) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    raw = _LEADING_INTERJECTION_RE.sub("", raw).lstrip()
    raw = _KEY_CONNECTIONS_BLOCK_RE.sub("", raw).strip()
    raw = _promote_section_headings(raw)
    return raw


def _filter_to_known_citation_ids(
    cite_utterance_ids: list[str],
    retrieval: dict[str, Any] | None,
) -> list[str]:
    if not cite_utterance_ids:
        return []
    if not isinstance(retrieval, dict):
        return [
            str(x or "").strip() for x in cite_utterance_ids if str(x or "").strip()
        ]

    known = {
        str(c.get("utterance_id") or "").strip()
        for c in (retrieval.get("citations") or [])
        if isinstance(c, dict)
    }
    known.discard("")
    if not known:
        return [
            str(x or "").strip() for x in cite_utterance_ids if str(x or "").strip()
        ]

    out: list[str] = []
    seen: set[str] = set()
    for raw in cite_utterance_ids:
        uid = str(raw or "").strip()
        if not uid or uid in seen:
            continue
        if uid in known:
            seen.add(uid)
            out.append(uid)
    return out


@dataclass
class _ToolCall:
    name: str
    args: dict[str, Any]


class KGAgentLoop:
    def __init__(
        self,
        *,
        postgres: Any,
        embedding_client: Any,
        client: Any | None = None,
        model: str = "gemini-3-flash-preview",
        max_tool_iterations: int = 4,
    ) -> None:
        self.postgres = postgres
        self.embedding_client = embedding_client
        self.model = model
        self.max_tool_iterations = max_tool_iterations

        if client is not None:
            self.client = client
        else:
            import os

            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set")
            self.client = genai.Client(api_key=api_key)

    def _system_prompt(self) -> str:
        return (
            "You are YuhHearDem, a friendly AI guide to Barbados Parliament debates. "
            "Keep a lightly Caribbean (Bajan) tone - warm and plainspoken, not cheesy.\n\n"
            "You MUST ground everything in retrieved evidence from the knowledge graph and transcript utterances.\n\n"
            "Rules:\n"
            "- Before answering, call the tool `kg_hybrid_graph_rag` to retrieve a compact subgraph + citations.\n"
            "- Use ONLY the tool results as your source of truth. Do not invent facts.\n"
            "- Interpret the tool results: use `edges` + `nodes` to explain relationships (who/what connects to what, agreements/disagreements, proposals, responsibilities) in plain language.\n"
            "- Prefer quoting MPs directly when citations are available; use markdown blockquotes and put the quoted sentence in *italics*.\n"
            "- Add visible inline citations in the answer body using markdown links like `[1](#src:utt_123)` or `[cite](#src:utt_123)` immediately after the sentence/quote they support.\n"
            "- Use only utterance_ids that appear in the tool citations.\n"
            "- Use short section headings for the main themes using markdown like `### The Climate Change Clash`.\n"
            "- Do NOT include a section called 'Key connections' and do NOT show technical arrow notation like `A -> PREDICATE -> B`.\n"
            "- Do NOT start your answer with filler like 'Wuhloss,'; start directly with the point.\n"
            "- Your `answer` field may contain markdown (bullets, bold, blockquotes).\n"
            "- When you make a claim, include at least one citation by listing its `utterance_id` in `cite_utterance_ids`.\n"
            "- If evidence is insufficient, say so clearly and ask one precise follow-up.\n"
            "- Return JSON only matching the response schema."
        )

    def _tool_declarations(self) -> list[types.Tool]:
        tool_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "hops": {"type": "integer", "default": 1},
                "seed_k": {"type": "integer", "default": 8},
                "max_edges": {"type": "integer", "default": 60},
                "max_citations": {"type": "integer", "default": 12},
            },
            "required": ["query"],
        }
        fd = types.FunctionDeclaration(
            name="kg_hybrid_graph_rag",
            description=(
                "Hybrid Graph-RAG: vector/fulltext seed search over kg_nodes, then expand kg_edges N hops "
                "and return a compact subgraph plus provenance citations with youtube timecoded URLs."
            ),
            parameters_json_schema=tool_schema,
        )
        return [types.Tool(function_declarations=[fd])]

    def _messages_to_contents(
        self, history: list[dict[str, str]], user_message: str
    ) -> list[types.Content]:
        contents: list[types.Content] = []
        for msg in history:
            role = msg.get("role")
            content = msg.get("content") or ""
            if not content:
                continue
            gemini_role = "user" if role == "user" else "model"
            contents.append(
                types.Content(
                    role=gemini_role, parts=[types.Part.from_text(text=content)]
                )
            )

        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
        )
        return contents

    def _extract_function_calls(self, response: Any) -> list[_ToolCall]:
        fcs = getattr(response, "function_calls", None)
        if not fcs:
            return []
        out: list[_ToolCall] = []
        for fc in fcs:
            name = getattr(fc, "name", None)
            args = getattr(fc, "args", None)
            if not name:
                continue
            out.append(_ToolCall(name=str(name), args=dict(args or {})))
        return out

    async def _call_llm(self, contents: list[types.Content]) -> Any:
        try:
            config = types.GenerateContentConfig(
                system_instruction=self._system_prompt(),
                tools=self._tool_declarations(),
                temperature=0.2,
                max_output_tokens=2048,
                response_mime_type="application/json",
                response_schema=AGENT_RESPONSE_SCHEMA,
            )
            return await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception:
            # Some model/tool combinations reject response_schema. Fall back to
            # best-effort JSON (still enforced in the system prompt).
            config = types.GenerateContentConfig(
                system_instruction=self._system_prompt(),
                tools=self._tool_declarations(),
                temperature=0.2,
                max_output_tokens=2048,
            )
            return await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

    async def run(
        self, *, user_message: str, history: list[dict[str, str]]
    ) -> dict[str, Any]:
        contents = self._messages_to_contents(history, user_message)
        last_retrieval: dict[str, Any] | None = None

        response = await self._call_llm(contents)
        iterations = 0

        while True:
            # Preserve the model-provided content object when continuing.
            # Reconstructing functionCall parts can drop required fields
            # (e.g., thought signatures), causing API errors.
            candidates = getattr(response, "candidates", None)
            if candidates:
                cand0 = candidates[0]
                model_content = getattr(cand0, "content", None)
                if model_content is not None:
                    contents.append(model_content)

            function_calls = self._extract_function_calls(response)
            if not function_calls:
                break

            iterations += 1
            if iterations > self.max_tool_iterations:
                break

            # Execute tools and add results
            result_parts: list[types.Part] = []
            for fc in function_calls:
                if fc.name == "kg_hybrid_graph_rag":
                    tool_result = kg_hybrid_graph_rag(
                        postgres=self.postgres,
                        embedding_client=self.embedding_client,
                        query=str(fc.args.get("query", "")),
                        hops=int(fc.args.get("hops", 1)),
                        seed_k=int(fc.args.get("seed_k", 8)),
                        max_edges=int(fc.args.get("max_edges", 60)),
                        max_citations=int(fc.args.get("max_citations", 12)),
                    )
                    last_retrieval = tool_result
                    result_parts.append(
                        types.Part.from_function_response(
                            name=fc.name,
                            response=tool_result,
                        )
                    )
                else:
                    result_parts.append(
                        types.Part.from_function_response(
                            name=fc.name,
                            response={"error": f"unknown tool: {fc.name}"},
                        )
                    )

            contents.append(types.Content(role="user", parts=result_parts))
            response = await self._call_llm(contents)

        parsed = _parse_json_best_effort(getattr(response, "text", None))
        if not parsed:
            parsed = {
                "answer": getattr(response, "text", None)
                or "I couldn't generate an answer.",
                "cite_utterance_ids": [],
                "focus_node_ids": [],
            }

        parsed.setdefault("cite_utterance_ids", [])
        parsed.setdefault("focus_node_ids", [])
        parsed.setdefault("answer", "")

        parsed["cite_utterance_ids"] = _filter_to_known_citation_ids(
            list(parsed.get("cite_utterance_ids") or []),
            last_retrieval,
        )

        parsed["answer"] = _clean_answer_text(parsed.get("answer"))
        parsed["retrieval"] = last_retrieval
        return parsed
