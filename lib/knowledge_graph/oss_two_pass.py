"""Two-pass refinement helpers to emulate "thinking" for gpt-oss-120b.

We keep this logic separate and deterministic so it can be tested without
network calls or database writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from lib.knowledge_graph.model_compare import normalize_speaker_ref


def normalize_utterance_ids_in_data(
    data: dict[str, Any], *, youtube_video_id: str
) -> None:
    """Normalize utterance_ids to full "{youtube_video_id}:<seconds>" strings.

    Some models sometimes output bare seconds like "1851". The transcript windows
    use full utterance ids (e.g. "Syxyah7QIaM:1851"). This normalizer is
    deterministic and keeps provenance compatible with downstream parsing.
    """

    def normalize_list(edge: dict[str, Any]) -> None:
        utterance_ids = edge.get("utterance_ids")
        if not isinstance(utterance_ids, list):
            return
        out: list[str] = []
        for uid in utterance_ids:
            s = str(uid).strip()
            if s.isdigit():
                out.append(f"{youtube_video_id}:{s}")
            else:
                out.append(s)
        edge["utterance_ids"] = out

    for key in ("edges", "edges_add"):
        edges = data.get(key)
        if not isinstance(edges, list):
            continue
        for e in edges:
            if isinstance(e, dict):
                normalize_list(e)


def normalize_evidence_in_data(data: dict[str, Any], *, window_text: str) -> None:
    """Normalize evidence fields to be exact substrings of window_text.

    Some models paraphrase slightly (e.g., remove repeated words) which breaks the
    strict substring rule. This fixer only runs when evidence is NOT already a
    substring and uses the referenced utterance_ids to choose a relevant snippet
    that is guaranteed to appear in the window.
    """

    if not window_text:
        return

    # Map utterance_id -> utterance content (text after metadata bracket)
    utterance_content: dict[str, str] = {}
    for line in window_text.splitlines():
        if not line.startswith("[utterance_id="):
            continue
        end_bracket = line.find("] ")
        if end_bracket == -1:
            continue
        header = line[: end_bracket + 1]
        uid_start = header.find("utterance_id=")
        if uid_start == -1:
            continue
        uid_start += len("utterance_id=")
        uid_end = header.find(" ", uid_start)
        if uid_end == -1:
            continue
        uid = header[uid_start:uid_end]
        utterance_content[uid] = line[end_bracket + 2 :]

    def fix_edge(edge: dict[str, Any]) -> None:
        evidence = str(edge.get("evidence", ""))
        if evidence and evidence in window_text:
            return

        utterance_ids = edge.get("utterance_ids")
        if not isinstance(utterance_ids, list) or not utterance_ids:
            return

        uid = str(utterance_ids[0]).strip()
        content = utterance_content.get(uid)
        if not content:
            return

        if evidence and evidence in content:
            return

        words = [w for w in evidence.replace("\n", " ").split() if w]
        start_idx = -1
        for n in (6, 5, 4, 3):
            if len(words) < n:
                continue
            for i in range(0, len(words) - n + 1):
                anchor = " ".join(words[i : i + n])
                pos = content.find(anchor)
                if pos != -1:
                    start_idx = pos
                    break
            if start_idx != -1:
                break

        if start_idx == -1:
            start_idx = 0

        max_len = min(220, max(60, len(evidence) + 40))
        snippet = content[start_idx : start_idx + max_len].rstrip()
        if snippet:
            edge["evidence"] = snippet

    for key in ("edges", "edges_add"):
        edges = data.get(key)
        if not isinstance(edges, list):
            continue
        for e in edges:
            if isinstance(e, dict):
                fix_edge(e)


class TwoPassMode(str, Enum):
    NONE = "none"
    ALWAYS = "always"
    ON_FAIL = "on_fail"
    ON_LOW_EDGES = "on_low_edges"
    ON_VIOLATIONS = "on_violations"


class RefineMode(str, Enum):
    AUDIT_REPAIR = "audit_repair"
    MISSING_ONLY = "missing_only"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    edge_index: int | None = None
    node_index: int | None = None


@dataclass(frozen=True)
class ValidationResult:
    edge_count: int
    node_count: int
    issues: list[ValidationIssue]

    @property
    def violations_count(self) -> int:
        return len(self.issues)


def validate_kg_llm_data(
    data: dict[str, Any],
    *,
    window_text: str,
    window_utterance_ids: set[str],
    window_speaker_ids: list[str],
    allowed_predicates: set[str],
    allowed_node_types: set[str],
) -> ValidationResult:
    """Validate LLM KG output against local constraints.

    This is intentionally stricter than "best effort" parsing: if evidence isn't
    anchored or utterance IDs are invalid, we flag it so pass 2 can repair/delete.
    """
    issues: list[ValidationIssue] = []

    nodes = data.get("nodes_new")
    edges = data.get("edges")
    if not isinstance(nodes, list):
        issues.append(
            ValidationIssue(
                code="nodes_new_not_list", message="nodes_new must be a list"
            )
        )
        nodes = []
    if not isinstance(edges, list):
        issues.append(
            ValidationIssue(code="edges_not_list", message="edges must be a list")
        )
        edges = []

    for i, n in enumerate(nodes):
        if not isinstance(n, dict):
            issues.append(
                ValidationIssue(
                    code="node_not_object",
                    message="node must be an object",
                    node_index=i,
                )
            )
            continue
        temp_id = str(n.get("temp_id", "")).strip()
        node_type = str(n.get("type", "")).strip()
        label = str(n.get("label", "")).strip()
        if not temp_id:
            issues.append(
                ValidationIssue(
                    code="node_missing_temp_id",
                    message="node missing temp_id",
                    node_index=i,
                )
            )
        if node_type not in allowed_node_types:
            issues.append(
                ValidationIssue(
                    code="node_type_invalid",
                    message=f"node type must be one of: {sorted(allowed_node_types)}",
                    node_index=i,
                )
            )
        if not label:
            issues.append(
                ValidationIssue(
                    code="node_missing_label",
                    message="node missing label",
                    node_index=i,
                )
            )

    for i, e in enumerate(edges):
        if not isinstance(e, dict):
            issues.append(
                ValidationIssue(
                    code="edge_not_object",
                    message="edge must be an object",
                    edge_index=i,
                )
            )
            continue

        source_ref = str(e.get("source_ref", "")).strip()
        predicate = str(e.get("predicate", "")).strip()
        target_ref = str(e.get("target_ref", "")).strip()
        evidence = str(e.get("evidence", ""))
        utterance_ids = e.get("utterance_ids")

        if not source_ref:
            issues.append(
                ValidationIssue(
                    code="edge_missing_source_ref",
                    message="edge missing source_ref",
                    edge_index=i,
                )
            )
        if predicate not in allowed_predicates:
            issues.append(
                ValidationIssue(
                    code="edge_predicate_invalid",
                    message=f"predicate must be one of: {sorted(allowed_predicates)}",
                    edge_index=i,
                )
            )
        if not target_ref:
            issues.append(
                ValidationIssue(
                    code="edge_missing_target_ref",
                    message="edge missing target_ref",
                    edge_index=i,
                )
            )

        if not evidence.strip():
            issues.append(
                ValidationIssue(
                    code="edge_missing_evidence",
                    message="edge missing evidence",
                    edge_index=i,
                )
            )
        elif evidence not in window_text:
            issues.append(
                ValidationIssue(
                    code="edge_evidence_not_substring",
                    message="evidence must be a direct substring of the transcript window",
                    edge_index=i,
                )
            )

        if not isinstance(utterance_ids, list) or not utterance_ids:
            issues.append(
                ValidationIssue(
                    code="edge_missing_utterance_ids",
                    message="edge must include non-empty utterance_ids list",
                    edge_index=i,
                )
            )
        else:
            bad = [
                str(uid)
                for uid in utterance_ids
                if str(uid) not in window_utterance_ids
            ]
            if bad:
                issues.append(
                    ValidationIssue(
                        code="edge_bad_utterance_ids",
                        message=f"utterance_ids not in window: {bad}",
                        edge_index=i,
                    )
                )

        # Speaker sanity checks (only if it looks like a speaker ref).
        for ref_name, ref_value in [
            ("source_ref", source_ref),
            ("target_ref", target_ref),
        ]:
            if ref_value.startswith("speaker_") or ref_value.startswith("s_"):
                norm = normalize_speaker_ref(ref_value, window_speaker_ids)
                if norm is None:
                    issues.append(
                        ValidationIssue(
                            code="edge_invalid_speaker_ref",
                            message=f"{ref_name} references speaker not present in window",
                            edge_index=i,
                        )
                    )

    return ValidationResult(
        edge_count=len(edges),
        node_count=len(nodes),
        issues=issues,
    )


def should_run_second_pass(
    *,
    mode: TwoPassMode,
    pass1_parse_success: bool,
    edge_count: int,
    violations_count: int,
    min_edges: int,
) -> tuple[bool, str | None]:
    if mode == TwoPassMode.NONE:
        return False, None
    if mode == TwoPassMode.ALWAYS:
        return True, "always"
    if mode == TwoPassMode.ON_FAIL:
        return (
            not pass1_parse_success
        ), "parse_fail" if not pass1_parse_success else None
    if mode == TwoPassMode.ON_LOW_EDGES:
        return (pass1_parse_success and edge_count < min_edges), (
            "low_edges" if pass1_parse_success and edge_count < min_edges else None
        )
    if mode == TwoPassMode.ON_VIOLATIONS:
        return (pass1_parse_success and violations_count > 0), (
            "violations" if pass1_parse_success and violations_count > 0 else None
        )
    return False, None


def build_refine_prompt(
    *,
    window_text: str,
    known_nodes_table: str,
    predicates: list[str],
    node_types: list[str],
    draft_json: str,
    issues: list[ValidationIssue],
    refine_mode: RefineMode,
    max_added_edges: int,
) -> str:
    """Build a second-pass prompt that audits and repairs the draft JSON.

    Important: this prompt explicitly allows deletion of invalid edges.
    """
    issues_lines = [
        f"- {iss.code}: {iss.message}"
        + (f" (edge_index={iss.edge_index})" if iss.edge_index is not None else "")
        + (f" (node_index={iss.node_index})" if iss.node_index is not None else "")
        for iss in issues
    ]
    issues_block = "\n".join(issues_lines) if issues_lines else "- (none detected)"

    predicates_str = ", ".join(predicates)
    node_types_str = ", ".join(node_types)

    missed_sweep = (
        "It is likely the draft missed some relationships. After fixing issues, re-read the transcript window "
        "and do a second sweep to see if there are any clearly-supported, substantive relationships not yet captured. "
        "Add only a small number of additional edges, and only when the evidence is unambiguous."
    )

    if refine_mode == RefineMode.MISSING_ONLY:
        refine_instructions = (
            "You MUST keep all valid edges and nodes from the draft. "
            "You MUST repair issues if possible, otherwise delete invalid items. "
            + missed_sweep
            + " You MAY add up to "
            f"{max_added_edges} additional edges if they are high-signal, fully supported, and high-confidence."
        )
    else:
        refine_instructions = (
            "You MUST produce a corrected KG JSON. "
            "You MUST repair issues if possible, otherwise delete invalid items. "
            + missed_sweep
            + " You MAY add up to "
            f"{max_added_edges} additional high-signal edges if they are fully supported and high-confidence."
        )

    return f"""You are a strict JSON editor for knowledge graph extraction.

TRANSCRIPT WINDOW:
{window_text}

KNOWN NODES (use these IDs when possible):
{known_nodes_table}

ALLOWED PREDICATES:
{predicates_str}

ALLOWED NODE TYPES:
{node_types_str}

DRAFT JSON (from pass 1):
{draft_json}

VALIDATION ISSUES DETECTED:
{issues_block}

INSTRUCTIONS:
1. Return valid JSON only - no markdown, no comments.
2. Evidence MUST be a direct substring quote from the transcript window.
3. Utterance IDs MUST refer to the provided utterances (copy the full value from "utterance_id=..." exactly; do NOT shorten to bare seconds).
4. Predicate MUST be from ALLOWED PREDICATES.
5. Node type MUST be from ALLOWED NODE TYPES.
6. {refine_instructions}
7. Avoid trivial connections. Prefer governance/causal/proposal/response relationships with explicit wording.
8. Be conservative: do not add speculative edges; if unsure, omit.

OUTPUT FORMAT:
{{
  "nodes_new": [{{"temp_id": "n1", "type": "skos:Concept", "label": "...", "aliases": ["..."]}}],
  "edges": [{{"source_ref": "...", "predicate": "...", "target_ref": "...", "evidence": "...", "utterance_ids": ["<video_id>:<seconds>", "<video_id>:<seconds>"], "confidence": 0.7}}]
}}

Return the corrected JSON now."""


def build_oss_draft_prompt(
    *,
    window_text: str,
    known_nodes_table: str,
    predicates: list[str],
    node_types: list[str],
    target_edges: int,
) -> str:
    """Build a recall-oriented prompt for gpt-oss-120b.

    This prompt is intentionally more explicit than the Gemini prompt to boost recall.
    """
    predicates_str = ", ".join(predicates)
    node_types_str = ", ".join(node_types)
    target = max(4, int(target_edges))

    return f"""You are extracting knowledge graph entities and relationships from parliamentary transcripts.

TRANSCRIPT WINDOW:
{window_text}

KNOWN NODES (use these IDs when possible):
{known_nodes_table}

RULES (must follow):
1. If a node matches a Known Node, you MUST use the existing id (do not create a new node).
2. For new nodes, assign a temporary id like \"n1\", \"n2\", etc.
3. Predicate must be from this list: {predicates_str}
4. Node type must be from this list: {node_types_str} (use \"skos:Concept\" for abstract concepts)
5. Evidence must be a direct substring quote from the transcript window.
 6. Utterance IDs must refer to the provided utterances (copy the full value from "utterance_id=..." exactly; do NOT shorten to bare seconds).
7. Return valid JSON only - no markdown, no comments.

RECALL OBJECTIVE:
- Aim to extract about {target} substantive edges if the window supports it.
- If you find fewer than {max(8, target - 2)} edges, do a second sweep over the transcript window before answering.
- Expand enumerations: if the text mentions multiple concrete items (e.g., PSVs, trucks), create separate nodes and edges for each.

PREDICATE GUIDANCE (choose strongest applicable):
- Prefer CAUSES when explicit impact/causation is stated (e.g., \"impacted\", \"because\", \"led to\").
- Prefer PROPOSES / MODERNIZES for plans, upgrades, or intended changes.
- Prefer RESPONSIBLE_FOR / IMPLEMENTED_BY for responsibility/implementation statements.
- Use ASSOCIATED_WITH only when no stronger predicate fits.
- Avoid collapsing everything into ADDRESSES if a more specific predicate fits.

OUTPUT FORMAT:
{{
  \"nodes_new\": [
    {{\"temp_id\": \"n1\", \"type\": \"skos:Concept\", \"label\": \"...\", \"aliases\": [\"...\"]}}
  ],
  \"edges\": [
    {{
      \"source_ref\": \"speaker_s_...\",
      \"predicate\": \"PROPOSES\",
      \"target_ref\": \"n1\",
      \"evidence\": \"...\",
      \"utterance_ids\": [\"<video_id>:<seconds>\", \"<video_id>:<seconds>\"],
      \"confidence\": 0.72
    }}
  ]
}}

Return JSON only."""


def build_oss_additions_prompt(
    *,
    window_text: str,
    known_nodes_table: str,
    predicates: list[str],
    node_types: list[str],
    draft_json: str,
    target_edges: int,
    max_added_edges: int,
) -> str:
    """Build a second-pass prompt that only adds missing edges/nodes.

    Output is deltas only so we can merge deterministically.
    """
    predicates_str = ", ".join(predicates)
    node_types_str = ", ".join(node_types)
    target = max(4, int(target_edges))
    max_add = max(0, int(max_added_edges))

    return f"""You are improving a knowledge graph extraction.

TRANSCRIPT WINDOW:
{window_text}

KNOWN NODES (use these IDs when possible):
{known_nodes_table}

ALLOWED PREDICATES:
{predicates_str}

ALLOWED NODE TYPES:
{node_types_str}

CURRENT DRAFT JSON:
{draft_json}

TASK:
1. Re-read the transcript window and look for substantive relationships that are CLEARLY supported but missing from the draft.
2. Be conservative: add edges only when the evidence is unambiguous.
3. Expand enumerations into multiple edges when the transcript lists multiple concrete items.
4. Add at most {max_add} new edges.
5. Aim to move the draft toward about {target} edges total if the window supports it.

STRICT RULES:
- Evidence MUST be a direct substring quote from the transcript window.
- Utterance IDs MUST refer to the provided utterances (copy the full value from "utterance_id=..." exactly; do NOT shorten to bare seconds).
- Predicate MUST be from ALLOWED PREDICATES.
- Node type MUST be from ALLOWED NODE TYPES.
- Return JSON only.

OUTPUT FORMAT (deltas only):
{{
  \"nodes_new_add\": [
    {{\"temp_id\": \"a1\", \"type\": \"skos:Concept\", \"label\": \"...\", \"aliases\": [\"...\"]}}
  ],
  \"edges_add\": [
    {{
      \"source_ref\": \"speaker_s_...\",
      \"predicate\": \"CAUSES\",
      \"target_ref\": \"a1\",
      \"evidence\": \"...\",
      \"utterance_ids\": [\"<video_id>:<seconds>\", \"<video_id>:<seconds>\"],
      \"confidence\": 0.8
    }}
  ],
  \"edges_delete\": []
}}

Return the JSON now."""


def merge_oss_additions(
    base: dict[str, Any],
    additions: dict[str, Any],
) -> dict[str, Any]:
    """Merge pass-2 delta output into pass-1 base output.

    Remaps added temp IDs to avoid collisions with base node temp_ids.
    """
    base_nodes_any = base.get("nodes_new")
    base_nodes: list[Any] = base_nodes_any if isinstance(base_nodes_any, list) else []
    base_edges_any = base.get("edges")
    base_edges: list[Any] = base_edges_any if isinstance(base_edges_any, list) else []

    add_nodes_any = additions.get("nodes_new_add")
    add_nodes: list[Any] = add_nodes_any if isinstance(add_nodes_any, list) else []
    add_edges_any = additions.get("edges_add")
    add_edges: list[Any] = add_edges_any if isinstance(add_edges_any, list) else []
    del_edges_any = additions.get("edges_delete")
    del_edges: list[Any] = del_edges_any if isinstance(del_edges_any, list) else []

    existing_ids = {
        str(n.get("temp_id"))
        for n in base_nodes
        if isinstance(n, dict) and n.get("temp_id")
    }

    # Build remap for added nodes.
    remap: dict[str, str] = {}
    counter = 1
    for n in add_nodes:
        if not isinstance(n, dict):
            continue
        old = str(n.get("temp_id", "")).strip() or f"a{counter}"
        new = old
        while new in existing_ids or new in remap.values():
            counter += 1
            new = f"a{counter}"
        remap[old] = new
        existing_ids.add(new)

    def remap_ref(ref: Any) -> Any:
        if not isinstance(ref, str):
            return ref
        return remap.get(ref, ref)

    merged_nodes: list[dict[str, Any]] = []
    for n in base_nodes:
        if isinstance(n, dict):
            merged_nodes.append(n)

    for n in add_nodes:
        if not isinstance(n, dict):
            continue
        old = str(n.get("temp_id", "")).strip()
        new = remap.get(old, old)
        n2 = dict(n)
        n2["temp_id"] = new
        merged_nodes.append(n2)

    # Apply deletions by simple signature match.
    delete_sigs = set()
    for d in del_edges:
        if not isinstance(d, dict):
            continue
        delete_sigs.add(
            (
                str(d.get("source_ref", "")),
                str(d.get("predicate", "")),
                str(d.get("target_ref", "")),
            )
        )

    merged_edges: list[dict[str, Any]] = []
    for e in base_edges:
        if not isinstance(e, dict):
            continue
        sig = (
            str(e.get("source_ref", "")),
            str(e.get("predicate", "")),
            str(e.get("target_ref", "")),
        )
        if sig in delete_sigs:
            continue
        merged_edges.append(e)

    for e in add_edges:
        if not isinstance(e, dict):
            continue
        e2 = dict(e)
        e2["source_ref"] = remap_ref(e2.get("source_ref"))
        e2["target_ref"] = remap_ref(e2.get("target_ref"))
        merged_edges.append(e2)

    return {"nodes_new": merged_nodes, "edges": merged_edges}
