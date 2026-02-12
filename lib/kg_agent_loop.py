from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from lib.kg_hybrid_graph_rag import kg_hybrid_graph_rag
from lib.utils.config import config


def _should_trace() -> bool:
    """Check if chat tracing is enabled."""
    return getattr(config, "chat_trace", False)


def _start_timer() -> float:
    """Start a timer and return the start time."""
    return time.perf_counter()


def _end_timer(start_time: float) -> float:
    """End a timer and return elapsed seconds."""
    return time.perf_counter() - start_time


def _format_duration(seconds: float) -> str:
    """Format duration as ms or seconds with 2 decimal places."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _format_node(node: dict[str, Any]) -> str:
    """Format a single KG node for trace logging."""
    node_id = node.get("id", "?")
    node_label = node.get("label", "?")[:30]
    node_type = node.get("type", "?")
    return f"{node_id} ({node_type}): {node_label}"


def _format_edge(edge: dict[str, Any]) -> str:
    """Format a single KG edge for trace logging."""
    edge_id = edge.get("id", "?")[:20]
    pred = edge.get("predicate", "?")[:30]
    source = edge.get("source_label", "?")[:20]
    target = edge.get("target_label", "?")[:20]
    return f"{edge_id}: {source} --[{pred}]--> {target}"


def _format_citation(citation: dict[str, Any]) -> str:
    """Format a single citation for trace logging."""
    uid = citation.get("utterance_id", "?")
    speaker = citation.get("speaker_name", "?")[:20]
    text_preview = (citation.get("text", "") or "")[:60]
    return f"{uid}: {speaker} said '{text_preview}...'"


def _format_tool_result_summary(result: dict[str, Any]) -> str:
    """Format a kg_hybrid_graph_rag result for trace logging."""
    out: list[str] = []
    out.append(f"query='{result.get('query', '?')[:50]}...'")
    out.append(f"hops={result.get('hops', '?')}")
    out.append(f"seeds_count={len(result.get('seeds', []))}")
    out.append(f"nodes_count={len(result.get('nodes', []))}")
    out.append(f"edges_count={len(result.get('edges', []))}")
    out.append(f"citations_count={len(result.get('citations', []))}")

    if result.get("nodes") and len(result["nodes"]) <= 5:
        node_labels = [f"{n.get('label', '?')[:20]}" for n in result["nodes"][:3]]
        out.append(f"nodes_preview={node_labels}")

    if result.get("citations") and len(result["citations"]) <= 5:
        cite_ids = [f"{c.get('utterance_id', '?')}" for c in result["citations"][:3]]
        out.append(f"citations_preview={cite_ids}")

    return ", ".join(out)


def _truncate_text(text: str, max_len: int = 300) -> str:
    """Truncate text to max_len with ellipsis."""
    if not text or len(text) <= max_len:
        return text or ""
    return text[: max_len - 3] + "..."


def _format_content_part_summary(part: types.Part) -> dict[str, Any]:
    """Format a content part for trace logging."""
    if part.text:
        return {"type": "text", "preview": _truncate_text(part.text, 300)}
    if part.function_call:
        fc = part.function_call
        return {
            "type": "function_call",
            "name": getattr(fc, "name", ""),
            "args_keys": list(dict(fc.args or {}).keys()) if fc.args else [],
        }
    if part.function_response:
        return {"type": "function_response", "name": part.function_response.name or ""}
    return {"type": "unknown"}


def _format_contents_summary(contents: list[types.Content]) -> list[dict[str, Any]]:
    """Format contents list for trace logging."""
    if not contents:
        return []
    result: list[dict[str, Any]] = []
    for c in contents:
        parts_list = getattr(c, "parts", None) or []
        parts_summary = [_format_content_part_summary(p) for p in parts_list]
        result.append({"role": c.role or "unknown", "parts": parts_summary})
    return result


def _serialize_content_part(part: types.Part) -> dict[str, Any]:
    """Serialize a content part to dict for raw logging."""
    if part.text:
        return {"type": "text", "content": part.text}
    if part.function_call:
        fc = part.function_call
        return {
            "type": "function_call",
            "name": getattr(fc, "name", ""),
            "args": dict(fc.args) if fc.args else {},
        }
    if part.function_response:
        return {
            "type": "function_response",
            "name": part.function_response.name or "",
            "response": part.function_response.response,
        }
    return {"type": "unknown"}


def _serialize_contents(contents: list[types.Content]) -> list[dict[str, Any]]:
    """Serialize contents list for raw logging."""
    if not contents:
        return []
    result: list[dict[str, Any]] = []
    for c in contents:
        parts_list = getattr(c, "parts", None) or []
        parts_serialized = [_serialize_content_part(p) for p in parts_list]
        result.append({"role": c.role or "unknown", "parts": parts_serialized})
    return result


def _trace_print(trace_id: str, section: str, message: str) -> None:
    """Print a trace message with consistent formatting."""
    if not _should_trace():
        return
    print(f"🔍 [TRACE {trace_id}] {section}")
    print(f"  {message}")


def _trace_section_start(trace_id: str, section: str) -> None:
    """Print a trace section header."""
    if not _should_trace():
        return
    print(f"\n{'=' * 60}")
    print(f"🔍 [TRACE {trace_id}] {section}")
    print(f"{'=' * 60}")


def _trace_section_end(trace_id: str) -> None:
    """Print a trace section footer."""
    if not _should_trace():
        return
    print(f"{'=' * 60}\n")


AGENT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "cite_utterance_ids": {"type": "array", "items": {"type": "string"}},
        "focus_node_ids": {"type": "array", "items": {"type": "string"}},
        "followup_questions": {"type": "array", "items": {"type": "string"}},
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
    raw, _ = _extract_embedded_followup_questions(raw)
    return raw


def _extract_embedded_followup_questions(text: str) -> tuple[str, list[str]]:
    lines = text.splitlines()
    marker_idx = -1
    for i, line in enumerate(lines):
        if re.search(r"follow\s*-?\s*up\s+questions", line, re.IGNORECASE):
            marker_idx = i
            break

    if marker_idx < 0:
        return text, []

    tail_lines = [ln.strip() for ln in lines[marker_idx + 1 :] if ln.strip() and ln.strip() != "-"]
    questionish = sum(1 for ln in tail_lines if ln.endswith("?"))

    # Only strip when this clearly looks like a generated follow-up section.
    if questionish < 2:
        return text, []

    followups = [re.sub(r"^[-*\d.)\s]+", "", ln).strip() for ln in tail_lines if ln.endswith("?")]
    followups = [q for q in followups if q]

    trimmed = "\n".join(lines[:marker_idx]).rstrip()
    return trimmed, followups


def _infer_citation_ids_from_bracket_numbers(
    answer: str, retrieval: dict[str, Any] | None
) -> list[str]:
    if not isinstance(retrieval, dict):
        return []

    citations = [
        c
        for c in (retrieval.get("citations") or [])
        if isinstance(c, dict) and str(c.get("utterance_id") or "").strip()
    ]
    if not citations:
        return []

    indices = [int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", answer or "")]
    out: list[str] = []
    seen: set[str] = set()
    for idx in indices:
        if idx <= 0 or idx > len(citations):
            continue
        uid = str(citations[idx - 1].get("utterance_id") or "").strip()
        if not uid or uid in seen:
            continue
        seen.add(uid)
        out.append(uid)
    return out


def _filter_to_known_citation_ids(
    cite_utterance_ids: list[str],
    retrieval: dict[str, Any] | None,
) -> list[str]:
    if not cite_utterance_ids:
        return []
    if not isinstance(retrieval, dict):
        return [str(x or "").strip() for x in cite_utterance_ids if str(x or "").strip()]

    known = {
        str(c.get("utterance_id") or "").strip()
        for c in (retrieval.get("citations") or [])
        if isinstance(c, dict)
    }
    known.discard("")
    if not known:
        return [str(x or "").strip() for x in cite_utterance_ids if str(x or "").strip()]

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
        enable_thinking: bool = False,
    ) -> None:
        self.postgres = postgres
        self.embedding_client = embedding_client
        self.model = model
        self.max_tool_iterations = max_tool_iterations
        self.enable_thinking = enable_thinking

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
            "You MUST ground everything in retrieved evidence from knowledge graph and transcript utterances.\n\n"
            "Rules:\n"
            "- Before answering, call the tool `kg_hybrid_graph_rag` to retrieve a compact subgraph + citations.\n"
            "- Use ONLY the tool results as your source of truth. Do not invent facts.\n"
            "- Interpret the tool results: use `edges` + `nodes` to explain relationships (who/what connects to what, agreements/disagreements, proposals, responsibilities) in plain language.\n"
            "- Prefer quoting MPs directly when citations are available; use markdown blockquotes and put the quoted sentence in *italics*.\n"
            "- Add visible inline citations in the answer body using markdown links like `[1](#src:utt_123)` or `[cite](#src:utt_123)` immediately after the sentence/quote they support.\n"
            "- Use only utterance_ids that appear in tool citations.\n"
            "- Use short section headings for main themes using markdown like `### The Climate Change Clash`.\n"
            "- Do NOT include a section called 'Key connections' and do NOT show technical arrow notation like `A -> PREDICATE -> B`.\n"
            "- Do NOT start your answer with filler like 'Wuhloss,'; start directly with the point.\n"
            "- Your `answer` field may contain markdown (bullets, bold, blockquotes).\n"
            "- When you make a claim, include at least one citation by listing its `utterance_id` in `cite_utterance_ids`.\n"
            "- If evidence is insufficient, say so clearly and ask one precise follow-up.\n"
            "- Generate 2-4 follow-up questions based on your answer that users might naturally want to ask next. These should be specific questions that explore related topics, dive deeper into mentioned entities, or ask about related legislation or debates.\n"
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
                types.Content(role=gemini_role, parts=[types.Part.from_text(text=content)])
            )

        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))
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

    async def _call_llm(self, contents: list[types.Content], is_tool_call: bool) -> Any:
        config_params: dict[str, Any] = {
            "system_instruction": self._system_prompt(),
            "tools": self._tool_declarations(),
            "temperature": 0.2,
            "max_output_tokens": 2048,
        }
        if not is_tool_call:
            # Gemini function-calling requests reject response_mime_type=response_schema
            # when tools are present. Keep tools enabled for follow-up tool calls and
            # rely on prompt + best-effort JSON parsing for structured final output.
            config_params.pop("response_schema", None)
            config_params.pop("response_mime_type", None)
        if not self.enable_thinking:
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        config = types.GenerateContentConfig(**config_params)
        return await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

    async def run(self, *, user_message: str, history: list[dict[str, str]]) -> dict[str, Any]:
        trace_id = str(uuid.uuid4())[:8]
        total_start = _start_timer()

        _trace_section_start(trace_id, "KG AGENT LOOP START")
        _trace_print(trace_id, "User Query", _truncate_text(user_message, 200))
        _trace_print(
            trace_id,
            "History",
            f"{len(history)} messages (last {min(3, len(history))} shown if tracing enabled)",
        )
        _trace_section_end(trace_id)

        contents = self._messages_to_contents(history, user_message)
        last_retrieval: dict[str, Any] | None = None

        _trace_section_start(trace_id, "ITERATION 0 - LLM CALL")
        _trace_print(trace_id, "Context Summary", f"{len(contents)} content parts")
        if _should_trace():
            for i, c in enumerate(_format_contents_summary(contents)):
                print(f"    [{i}] {c['role']}: {c['parts']}")
            print(f"\n🔍 [TRACE {trace_id}] RAW CONTENTS SENT TO LLM")
            serialized = _serialize_contents(contents)
            for i, c in enumerate(serialized):
                print(f"\n  [{i}] Role: {c['role']}")
                for j, p in enumerate(c.get("parts", [])):
                    p_type = p.get("type", "unknown")
                    print(f"    [{j}] Type: {p_type}")
                    if p_type == "text":
                        print(f"        Content: {p.get('content', '')}")
                    elif p_type == "function_call":
                        print(f"        Name: {p.get('name', '')}")
                        print(f"        Args: {p.get('args', {})}")
                    elif p_type == "function_response":
                        print(f"        Name: {p.get('name', '')}")
                        print(f"        Response type: {type(p.get('response', {})).__name__}")
            print(f"\n{'=' * 60}\n")
        llm_start = _start_timer()
        response = await self._call_llm(contents, is_tool_call=True)
        llm_duration = _end_timer(llm_start)
        _trace_print(trace_id, "Duration", _format_duration(llm_duration))
        if _should_trace():
            response_text = getattr(response, "text", None) or ""
            print(f"\n🔍 [TRACE {trace_id}] RAW LLM RESPONSE")
            print(f"  Length: {len(response_text)} chars")
            print(f"  Content:\n{response_text}")
            print(f"\n{'=' * 60}\n")
        _trace_section_end(trace_id)

        iterations = 0

        while True:
            _trace_section_start(trace_id, f"PARSING LLM RESPONSE (iteration {iterations})")

            candidates = getattr(response, "candidates", None)
            if candidates:
                cand0 = candidates[0]
                model_content = getattr(cand0, "content", None)
                if model_content is not None:
                    contents.append(model_content)

            function_calls = self._extract_function_calls(response)

            if not function_calls:
                _trace_print(trace_id, "Function Calls", "None - loop complete")
                _trace_section_end(trace_id)
                break

            _trace_print(
                trace_id,
                "Function Calls",
                f"{len(function_calls)} call(s): {[fc.name for fc in function_calls]}",
            )

            _trace_section_end(trace_id)

            iterations += 1
            if iterations > self.max_tool_iterations:
                _trace_print(
                    trace_id,
                    "Limit",
                    f"Max tool iterations ({self.max_tool_iterations}) reached",
                )
                break

            _trace_section_start(trace_id, f"EXECUTING TOOLS (iteration {iterations})")
            result_parts: list[types.Part] = []
            for fc in function_calls:
                if fc.name == "kg_hybrid_graph_rag":
                    _trace_print(
                        trace_id,
                        "Tool Call",
                        f"kg_hybrid_graph_rag(query={_truncate_text(str(fc.args.get('query', '')), 100)}, hops={fc.args.get('hops', 1)}, seed_k={fc.args.get('seed_k', 8)})",
                    )
                    tool_start = _start_timer()
                    tool_result = kg_hybrid_graph_rag(
                        postgres=self.postgres,
                        embedding_client=self.embedding_client,
                        query=str(fc.args.get("query", "")),
                        hops=int(fc.args.get("hops", 1)),
                        seed_k=int(fc.args.get("seed_k", 8)),
                        max_edges=int(fc.args.get("max_edges", 60)),
                        max_citations=int(fc.args.get("max_citations", 12)),
                    )
                    tool_duration = _end_timer(tool_start)
                    last_retrieval = tool_result

                    if _should_trace():
                        _trace_print(
                            trace_id,
                            "Tool Duration",
                            _format_duration(tool_duration),
                        )
                        _trace_print(
                            trace_id,
                            "Tool Result",
                            _format_tool_result_summary(tool_result),
                        )

                    result_parts.append(
                        types.Part.from_function_response(
                            name=fc.name,
                            response=tool_result,
                        )
                    )
                else:
                    _trace_print(
                        trace_id,
                        "Tool Error",
                        f"unknown tool: {fc.name}",
                    )
                    result_parts.append(
                        types.Part.from_function_response(
                            name=fc.name,
                            response={"error": f"unknown tool: {fc.name}"},
                        )
                    )

            contents.append(types.Content(role="user", parts=result_parts))
            if _should_trace() and result_parts:
                _trace_print(
                    trace_id,
                    "Tool Response to LLM",
                    f"function_response(name='kg_hybrid_graph_rag') with {len(result_parts)} part(s)",
                )
                response_summary = (
                    result_parts[0].function_response.response
                    if result_parts[0].function_response
                    else {}
                )
                if isinstance(response_summary, dict):
                    for key in ["query", "hops", "nodes", "edges", "citations"]:
                        val = response_summary.get(key)
                        if isinstance(val, list):
                            _trace_print(trace_id, f"  {key}_count", f"{len(val)}")
                        elif val is not None:
                            _trace_print(trace_id, f"  {key}", str(val)[:100])

                    print(f"\n🔍 [TRACE {trace_id}] ACTUAL KG DATA SENT TO LLM")
                    nodes = response_summary.get("nodes", [])[:5]
                    edges = response_summary.get("edges", [])[:5]
                    citations = response_summary.get("citations", [])[:5]

                    if nodes:
                        print(
                            f"  Nodes (first {len(nodes)} of {len(response_summary.get('nodes', []))}):"
                        )
                        for i, n in enumerate(nodes):
                            print(f"    [{i}] {_format_node(n)}")

                    if edges:
                        print(
                            f"\n  Edges (first {len(edges)} of {len(response_summary.get('edges', []))}):"
                        )
                        for i, e in enumerate(edges):
                            print(f"    [{i}] {_format_edge(e)}")

                    if citations:
                        print(
                            f"\n  Citations (first {len(citations)} of {len(response_summary.get('citations', []))}):"
                        )
                        for i, c in enumerate(citations):
                            print(f"    [{i}] {_format_citation(c)}")

                    print(f"\n{'=' * 60}\n")

            _trace_section_end(trace_id)

            _trace_section_start(trace_id, f"ITERATION {iterations} - LLM CALL")
            _trace_print(trace_id, "Context Summary", f"{len(contents)} content parts")
            if _should_trace():
                for i, c in enumerate(_format_contents_summary(contents)):
                    print(f"    [{i}] {c['role']}: {len(c['parts'])} part(s)")
                    for j, p in enumerate(c["parts"][:2]):
                        p_type = p.get("type", "unknown")
                        preview = p.get("preview", "")
                        if preview:
                            print(f"        [{j}] {p_type}: {preview}")
                print(f"\n🔍 [TRACE {trace_id}] RAW CONTENTS SENT TO LLM")
                serialized = _serialize_contents(contents)
                for i, c in enumerate(serialized):
                    print(f"\n  [{i}] Role: {c['role']}")
                    for j, p in enumerate(c.get("parts", [])):
                        p_type = p.get("type", "unknown")
                        print(f"    [{j}] Type: {p_type}")
                        if p_type == "text":
                            print(f"        Content: {p.get('content', '')}")
                        elif p_type == "function_call":
                            print(f"        Name: {p.get('name', '')}")
                            print(f"        Args: {p.get('args', {})}")
                        elif p_type == "function_response":
                            print(f"        Name: {p.get('name', '')}")
                            print(f"        Response type: {type(p.get('response', {})).__name__}")
                print(f"\n{'=' * 60}\n")
            llm_start = _start_timer()
            response = await self._call_llm(contents, is_tool_call=False)
            llm_duration = _end_timer(llm_start)
            _trace_print(trace_id, "Duration", _format_duration(llm_duration))
            if _should_trace():
                response_text = getattr(response, "text", None) or ""
                print(f"\n🔍 [TRACE {trace_id}] RAW LLM RESPONSE")
                print(f"  Length: {len(response_text)} chars")
                print(f"  Content:\n{response_text}")
                print(f"\n{'=' * 60}\n")
            _trace_section_end(trace_id)

        _trace_section_start(trace_id, "FINAL ANSWER PARSING")
        parsed = _parse_json_best_effort(getattr(response, "text", None))
        if not parsed:
            parsed = {
                "answer": getattr(response, "text", None) or "I couldn't generate an answer.",
                "cite_utterance_ids": [],
                "focus_node_ids": [],
                "followup_questions": [],
            }

        parsed.setdefault("cite_utterance_ids", [])
        parsed.setdefault("focus_node_ids", [])
        parsed.setdefault("answer", "")
        parsed.setdefault("followup_questions", [])

        parsed["followup_questions"] = [
            str(q).strip()
            for q in list(parsed.get("followup_questions") or [])
            if str(q or "").strip()
        ][:4]

        cleaned_answer, embedded_followups = _extract_embedded_followup_questions(
            str(parsed.get("answer") or "")
        )
        if embedded_followups:
            merged_followups = list(parsed.get("followup_questions") or []) + embedded_followups
            deduped: list[str] = []
            seen_followups: set[str] = set()
            for q in merged_followups:
                normalized_q = str(q).strip()
                if not normalized_q or normalized_q in seen_followups:
                    continue
                seen_followups.add(normalized_q)
                deduped.append(normalized_q)
            parsed["followup_questions"] = deduped[:4]
        parsed["answer"] = cleaned_answer

        cite_ids = _filter_to_known_citation_ids(
            list(parsed.get("cite_utterance_ids") or []),
            last_retrieval,
        )
        inferred_ids = _infer_citation_ids_from_bracket_numbers(
            parsed.get("answer", ""), last_retrieval
        )
        for inferred in inferred_ids:
            if inferred not in cite_ids:
                cite_ids.append(inferred)
        parsed["cite_utterance_ids"] = cite_ids

        parsed["answer"] = _clean_answer_text(parsed.get("answer"))
        parsed["retrieval"] = last_retrieval

        total_duration = _end_timer(total_start)
        if _should_trace():
            _trace_print(
                trace_id,
                "Final Answer Summary",
                f"length={len(parsed.get('answer', ''))} chars, "
                f"cite_ids={parsed.get('cite_utterance_ids', [])[:3]}{'...' if len(parsed.get('cite_utterance_ids', [])) > 3 else ''}, "
                f"focus_nodes={parsed.get('focus_node_ids', [])[:3]}{'...' if len(parsed.get('focus_node_ids', [])) > 3 else ''}",
            )
            _trace_print(trace_id, "Total Duration", _format_duration(total_duration))
            _trace_section_end(trace_id)

        return parsed
