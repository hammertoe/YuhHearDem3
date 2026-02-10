from __future__ import annotations

from lib.knowledge_graph.oss_two_pass import normalize_evidence_in_data


def test_normalize_evidence_should_replace_when_not_substring() -> None:
    window_text = "\n".join(
        [
            "[utterance_id=vid:10 t=0:00:10 speaker_id=s_a] we we are at a point now where the the future looks bright",
            "[utterance_id=vid:11 t=0:00:11 speaker_id=s_a] other",
        ]
    )
    data = {
        "edges": [
            {
                "source_ref": "speaker_s_a",
                "predicate": "PROPOSES",
                "target_ref": "n1",
                "evidence": "we are at a point now where the future looks bright",
                "utterance_ids": ["vid:10"],
                "confidence": 0.5,
            }
        ]
    }

    assert data["edges"][0]["evidence"] not in window_text
    normalize_evidence_in_data(data, window_text=window_text)
    assert data["edges"][0]["evidence"] in window_text
