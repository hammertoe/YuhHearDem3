from __future__ import annotations

from lib.knowledge_graph.oss_two_pass import (
    RefineMode,
    TwoPassMode,
    build_refine_prompt,
    should_run_second_pass,
    validate_kg_llm_data,
)


def test_should_run_second_pass_on_low_edges() -> None:
    should, reason = should_run_second_pass(
        mode=TwoPassMode.ON_LOW_EDGES,
        pass1_parse_success=True,
        edge_count=2,
        violations_count=0,
        min_edges=4,
    )
    assert should is True
    assert reason == "low_edges"


def test_validate_should_flag_evidence_not_substring() -> None:
    window_text = "[utterance_id=v:1 t=0:00:01 speaker_id=s_a] Hello world"
    data = {
        "nodes_new": [],
        "edges": [
            {
                "source_ref": "s_a",
                "predicate": "PROPOSES",
                "target_ref": "n1",
                "evidence": "Not in window",
                "utterance_ids": ["v:1"],
                "confidence": 0.8,
            }
        ],
    }
    res = validate_kg_llm_data(
        data,
        window_text=window_text,
        window_utterance_ids={"v:1"},
        window_speaker_ids=["s_a"],
        allowed_predicates={"PROPOSES"},
        allowed_node_types={"skos:Concept"},
    )
    codes = {i.code for i in res.issues}
    assert "edge_evidence_not_substring" in codes


def test_refine_prompt_mentions_deletion_and_limits_added_edges() -> None:
    prompt = build_refine_prompt(
        window_text="win",
        known_nodes_table="tbl",
        predicates=["PROPOSES"],
        node_types=["skos:Concept"],
        draft_json='{"nodes_new":[],"edges":[]}',
        issues=[],
        refine_mode=RefineMode.AUDIT_REPAIR,
        max_added_edges=7,
    )
    assert "delete" in prompt.lower()
    assert "7 additional" in prompt
