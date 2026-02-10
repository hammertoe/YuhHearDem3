from __future__ import annotations

from lib.knowledge_graph.oss_two_pass import merge_oss_additions


def test_merge_should_remap_added_temp_ids_and_update_edges() -> None:
    base = {
        "nodes_new": [
            {"temp_id": "n1", "type": "skos:Concept", "label": "A"},
        ],
        "edges": [
            {
                "source_ref": "speaker_s_a_1",
                "predicate": "ADDRESSES",
                "target_ref": "n1",
                "evidence": "A",
                "utterance_ids": ["v:1"],
                "confidence": 0.5,
            }
        ],
    }

    additions = {
        "nodes_new_add": [
            {"temp_id": "n1", "type": "skos:Concept", "label": "B"},
            {"temp_id": "a1", "type": "skos:Concept", "label": "C"},
        ],
        "edges_add": [
            {
                "source_ref": "speaker_s_a_1",
                "predicate": "CAUSES",
                "target_ref": "n1",
                "evidence": "B",
                "utterance_ids": ["v:1"],
                "confidence": 0.9,
            },
            {
                "source_ref": "speaker_s_a_1",
                "predicate": "CAUSES",
                "target_ref": "a1",
                "evidence": "C",
                "utterance_ids": ["v:1"],
                "confidence": 0.9,
            },
        ],
        "edges_delete": [],
    }

    merged = merge_oss_additions(base, additions)
    temp_ids = {n["temp_id"] for n in merged["nodes_new"]}
    assert "n1" in temp_ids
    assert "a1" in temp_ids
    # Collision for added "n1" must be remapped.
    assert len(temp_ids) == 3

    # Ensure the added edge targeting old n1 got remapped.
    base_n1_edges = [e for e in merged["edges"] if e.get("target_ref") == "n1"]
    assert len(base_n1_edges) == 1
