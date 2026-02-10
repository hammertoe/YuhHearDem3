from __future__ import annotations

from lib.knowledge_graph.model_compare import (
    canonicalize_edges,
    canonicalize_nodes,
    edge_signature_loose,
    normalize_speaker_ref,
)


def test_normalize_speaker_ref_should_prefix_bare_speaker_id() -> None:
    window_speaker_ids = ["s_alice_1", "s_bob_1"]
    assert normalize_speaker_ref("s_alice_1", window_speaker_ids) == "speaker_s_alice_1"


def test_normalize_speaker_ref_should_reject_unknown_speaker() -> None:
    window_speaker_ids = ["s_alice_1"]
    assert normalize_speaker_ref("speaker_s_bob_1", window_speaker_ids) is None


def test_canonicalize_should_map_temp_ids_and_keep_speaker_ids() -> None:
    nodes_new = [
        {"temp_id": "n1", "type": "skos:Concept", "label": "Fixed penalty regime"}
    ]
    edges = [
        {
            "source_ref": "s_alice_1",
            "predicate": "PROPOSES",
            "target_ref": "n1",
            "evidence": "fixed penalties",
            "utterance_ids": ["v:1"],
            "earliest_seconds": 1,
        }
    ]
    temp_to_canonical = canonicalize_nodes(nodes_new)
    canon = canonicalize_edges(
        edges,
        temp_to_canonical=temp_to_canonical,
        window_speaker_ids=["s_alice_1"],
    )
    assert len(canon) == 1
    assert canon[0]["source_id"] == "speaker_s_alice_1"
    assert canon[0]["target_id"].startswith("kg_")
    assert edge_signature_loose(canon[0])[1] == "PROPOSES"
