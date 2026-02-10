"""Cerebras-only KG extractor using gpt-oss-120b with two-pass refinement.

This module implements extraction of a ConceptWindow using Cerebras gpt-oss-120b
with a two-pass approach and reasoning for improved recall.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from lib.db.postgres_client import PostgresClient
from lib.embeddings.google_client import GoogleEmbeddingClient
from lib.knowledge_graph.window_builder import ConceptWindow, Window
from lib.knowledge_graph.oss_two_pass import (
    RefineMode,
    TwoPassMode,
    build_oss_additions_prompt,
    build_oss_draft_prompt,
    build_refine_prompt,
    merge_oss_additions,
    normalize_evidence_in_data,
    normalize_utterance_ids_in_data,
    validate_kg_llm_data,
)

load_dotenv()


DEFAULT_MODEL = "gpt-oss-120b"
DEFAULT_TWO_PASS = TwoPassMode.ALWAYS
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_REASONING_FORMAT = "hidden"
DEFAULT_MAX_COMPLETION_TOKENS = 16384


PREDICATES = [
    "AMENDS",
    "GOVERNS",
    "MODERNIZES",
    "AIMS_TO_REDUCE",
    "REQUIRES_APPROVAL",
    "IMPLEMENTED_BY",
    "RESPONSIBLE_FOR",
    "ASSOCIATED_WITH",
    "CAUSES",
    "ADDRESSES",
    "PROPOSES",
    "RESPONDS_TO",
    "AGREES_WITH",
    "DISAGREES_WITH",
    "QUESTIONS",
]


NODE_TYPES = [
    "foaf:Person",
    "schema:Legislation",
    "schema:Organization",
    "schema:Place",
    "skos:Concept",
]


@dataclass
class ExtractionResult:
    """Result of extraction from a single window."""

    window: Window
    nodes_new: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    raw_response: str
    parse_success: bool
    error: str | None = None

    # Two-pass metrics
    pass1_elapsed_s: float | None = None
    pass2_elapsed_s: float | None = None
    pass2_trigger: str | None = None
    pass1_parse_success: bool | None = None
    pass1_edge_count: int | None = None
    pass1_violations_count: int | None = None
    prompt_pass1: str | None = None
    prompt_pass2: str | None = None
    raw_response_pass1: str | None = None
    raw_response_pass2: str | None = None
    reasoning_pass1: str | None = None
    reasoning_pass2: str | None = None
    pass1_error: str | None = None
    pass2_error: str | None = None


class OssKGExtractor:
    """Extract knowledge graph entities and relationships using Cerebras gpt-oss-120b."""

    def __init__(
        self,
        postgres_client: PostgresClient,
        embedding_client: GoogleEmbeddingClient,
        model: str = DEFAULT_MODEL,
        two_pass: TwoPassMode = DEFAULT_TWO_PASS,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        reasoning_format: str = DEFAULT_REASONING_FORMAT,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    ):
        self.postgres = postgres_client
        self.embedding = embedding_client

        api_key = self._get_api_key()
        self.client = Cerebras(api_key=api_key)
        self.model = model
        self.two_pass = two_pass
        self.reasoning_effort = reasoning_effort
        self.reasoning_format = reasoning_format
        self.max_completion_tokens = max_completion_tokens

    def _get_api_key(self) -> str:
        """Get Cerebras API key from environment."""
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY environment variable is not set")
        return api_key

    def _get_known_nodes_table(
        self, window: ConceptWindow, youtube_video_id: str, top_k: int = 25
    ) -> str:
        """Get candidate nodes for the window."""
        from lib.knowledge_graph.window_builder import WindowBuilder

        window_builder = WindowBuilder(self.postgres, self.embedding)
        candidates = window_builder.get_candidate_nodes(
            window.text, window.speaker_ids, youtube_video_id, top_k
        )
        return window_builder.format_known_nodes(candidates)

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON response from LLM, handling various formats."""
        response = response.strip()

        # Try to extract JSON from markdown code blocks
        if "```json" in response:
            match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        elif "```" in response:
            match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                return json.loads(match.group(1))

        # Try brace matching for top-level JSON object
        brace_start = response.find("{")
        if brace_start != -1:
            brace_count = 0
            brace_end = -1
            for i in range(brace_start, len(response)):
                if response[i] == "{":
                    brace_count += 1
                elif response[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        brace_end = i + 1
                        break
            if brace_end != -1:
                json_str = response[brace_start:brace_end]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            raise ValueError("Could not parse JSON from response")

    def _call_cerebras(
        self,
        prompt: str,
        use_reasoning: bool = True,
    ) -> tuple[str, str | None]:
        """Call Cerebras API with retry logic.

        Returns:
            (content, reasoning) tuple
        """
        system_prompt = (
            "You are extracting knowledge graph entities and relationships from parliamentary transcripts. "
            "Return JSON only. Do not include markdown."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_completion_tokens": self.max_completion_tokens,
        }

        if use_reasoning:
            kwargs["reasoning_effort"] = self.reasoning_effort
            kwargs["reasoning_format"] = self.reasoning_format

        response = self._retry_call(kwargs)

        reasoning = None
        content = ""

        msg = response.choices[0].message
        content = msg.content
        reasoning = getattr(msg, "reasoning", None)

        return content, reasoning

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _retry_call(self, kwargs: dict) -> Any:
        """Retryable Cerebras API call."""
        response = self.client.chat.completions.create(**kwargs)
        return response

    def extract_from_concept_window(
        self, window: ConceptWindow, youtube_video_id: str, top_k: int = 25
    ) -> ExtractionResult:
        """Extract knowledge graph from a concept window using two-pass approach."""
        known_nodes_table = self._get_known_nodes_table(window, youtube_video_id, top_k)
        window_utterance_ids = {u.id for u in window.utterances}

        # Pass 1: Recall-oriented draft
        target_edges = len(window.utterances) + 2
        prompt_pass1 = build_oss_draft_prompt(
            window_text=window.text,
            known_nodes_table=known_nodes_table,
            predicates=PREDICATES,
            node_types=NODE_TYPES,
            target_edges=target_edges,
        )

        pass1_start = time.time()
        try:
            raw_response_pass1, reasoning_pass1 = self._call_cerebras(
                prompt_pass1, use_reasoning=True
            )
            data_pass1 = self._parse_json_response(raw_response_pass1)

            # Normalize pass1 output
            normalize_utterance_ids_in_data(
                data_pass1, youtube_video_id=youtube_video_id
            )
            normalize_evidence_in_data(data_pass1, window_text=window.text)

            pass1_parse_success = True
            pass1_error = None

        except Exception as e:
            import traceback

            traceback.print_exc()
            pass1_parse_success = False
            pass1_error = f"{type(e).__name__}: {e}"
            data_pass1 = {"nodes_new": [], "edges": []}
            raw_response_pass1 = prompt_pass1
            reasoning_pass1 = None

        pass1_elapsed_s = time.time() - pass1_start

        # Validate pass1 output
        if pass1_parse_success:
            validation_result = validate_kg_llm_data(
                data_pass1,
                window_text=window.text,
                window_utterance_ids=window_utterance_ids,
                window_speaker_ids=window.speaker_ids,
                allowed_predicates=set(PREDICATES),
                allowed_node_types=set(NODE_TYPES),
            )
            pass1_violations_count = validation_result.violations_count
            pass1_edge_count = len(data_pass1.get("edges", []))
        else:
            validation_result = None
            pass1_violations_count = 0
            pass1_edge_count = 0

        # Determine if pass 2 is needed
        should_run_pass2, pass2_trigger = (
            (True, "always") if self.two_pass == TwoPassMode.ALWAYS else (False, None)
        )

        # Pass 2: Refine or additions only
        pass2_elapsed_s = None
        raw_response_pass2 = None
        reasoning_pass2 = None
        pass2_error = None
        prompt_pass2 = None
        final_data = data_pass1

        if should_run_pass2 and pass1_parse_success:
            pass2_start = time.time()
            try:
                if pass1_violations_count == 0:
                    # Additions-only prompt (recall mode)
                    prompt_pass2 = build_oss_additions_prompt(
                        window_text=window.text,
                        known_nodes_table=known_nodes_table,
                        predicates=PREDICATES,
                        node_types=NODE_TYPES,
                        draft_json=json.dumps(data_pass1, indent=2),
                        target_edges=target_edges,
                        max_added_edges=3,
                    )
                    raw_response_pass2, reasoning_pass2 = self._call_cerebras(
                        prompt_pass2, use_reasoning=True
                    )
                    data_pass2 = self._parse_json_response(raw_response_pass2)

                    # Normalize pass2 output
                    normalize_utterance_ids_in_data(
                        data_pass2, youtube_video_id=youtube_video_id
                    )
                    normalize_evidence_in_data(data_pass2, window_text=window.text)

                    # Merge additions
                    final_data = merge_oss_additions(data_pass1, data_pass2)
                else:
                    # Repair prompt (deletion allowed)
                    prompt_pass2 = build_refine_prompt(
                        window_text=window.text,
                        known_nodes_table=known_nodes_table,
                        predicates=PREDICATES,
                        node_types=NODE_TYPES,
                        draft_json=json.dumps(data_pass1, indent=2),
                        issues=validation_result.issues if validation_result else [],
                        refine_mode=RefineMode.AUDIT_REPAIR,
                        max_added_edges=3,
                    )
                    raw_response_pass2, reasoning_pass2 = self._call_cerebras(
                        prompt_pass2, use_reasoning=True
                    )
                    final_data = self._parse_json_response(raw_response_pass2)

                    # Normalize pass2 output
                    normalize_utterance_ids_in_data(
                        final_data, youtube_video_id=youtube_video_id
                    )
                    normalize_evidence_in_data(final_data, window_text=window.text)

                pass2_error = None

            except Exception as e:
                pass2_error = str(e)
                final_data = data_pass1

            pass2_elapsed_s = time.time() - pass2_start

        # Build final result
        utterance_timestamps = {
            u.id: (u.timestamp_str, u.seconds_since_start) for u in window.utterances
        }

        # Process edges with timestamps
        edges = []
        for edge_data in final_data.get("edges", []):
            try:
                source_ref = edge_data["source_ref"]
                predicate = edge_data["predicate"]
                target_ref = edge_data["target_ref"]
                evidence = edge_data["evidence"]
            except Exception:
                continue

            utterance_ids = edge_data.get("utterance_ids")
            if not isinstance(utterance_ids, list) or not utterance_ids:
                continue

            earliest_timestamp_str = None
            earliest_seconds = None

            for uid in utterance_ids:
                if uid in utterance_timestamps:
                    ts_str, ts_seconds = utterance_timestamps[uid]
                    if earliest_seconds is None or ts_seconds < earliest_seconds:
                        earliest_timestamp_str = ts_str
                        earliest_seconds = ts_seconds

            edges.append(
                {
                    "source_ref": source_ref,
                    "predicate": predicate,
                    "target_ref": target_ref,
                    "evidence": evidence,
                    "utterance_ids": utterance_ids,
                    "earliest_timestamp": earliest_timestamp_str
                    or window.earliest_timestamp,
                    "earliest_seconds": earliest_seconds or window.earliest_seconds,
                    "confidence": float(edge_data.get("confidence", 0.5)),
                }
            )

        nodes_new = final_data.get("nodes_new", [])

        return ExtractionResult(
            window=window,
            nodes_new=nodes_new,
            edges=edges,
            raw_response=raw_response_pass2 or raw_response_pass1,
            parse_success=pass1_parse_success
            and (pass2_error is None if should_run_pass2 else True),
            error=pass1_error or pass2_error,
            pass1_elapsed_s=pass1_elapsed_s,
            pass2_elapsed_s=pass2_elapsed_s,
            pass2_trigger=pass2_trigger,
            pass1_parse_success=pass1_parse_success,
            pass1_edge_count=pass1_edge_count,
            pass1_violations_count=pass1_violations_count,
            prompt_pass1=prompt_pass1,
            prompt_pass2=prompt_pass2 if should_run_pass2 else None,
            raw_response_pass1=raw_response_pass1,
            raw_response_pass2=raw_response_pass2,
            reasoning_pass1=reasoning_pass1,
            reasoning_pass2=reasoning_pass2,
            pass1_error=pass1_error,
            pass2_error=pass2_error,
        )
