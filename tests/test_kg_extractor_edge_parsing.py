from __future__ import annotations

from lib.knowledge_graph.kg_extractor import KGExtractor
from lib.knowledge_graph.window_builder import ConceptWindow, Utterance


def test_parse_edges_should_skip_edge_missing_utterance_ids() -> None:
    extractor = KGExtractor.__new__(KGExtractor)

    window = ConceptWindow(
        utterances=[
            Utterance(
                id="video1:10",
                timestamp_str="0:00:10",
                seconds_since_start=10,
                speaker_id="s_a",
                text="Some sufficiently long utterance.",
            )
        ],
        window_index=0,
    )

    utterance_timestamps = {"video1:10": ("0:00:10", 10)}
    data = {
        "edges": [
            {
                "source_ref": "speaker_s_a",
                "predicate": "PROPOSES",
                "target_ref": "n1",
                "evidence": "Some",
                "confidence": 0.7,
                # missing utterance_ids
            },
            {
                "source_ref": "speaker_s_a",
                "predicate": "PROPOSES",
                "target_ref": "n1",
                "evidence": "Some",
                "confidence": 0.7,
                "utterance_ids": ["video1:10"],
            },
        ]
    }

    edges = extractor._parse_edges_from_llm_data(data, utterance_timestamps, window)
    assert len(edges) == 1
    assert edges[0].utterance_ids == ["video1:10"]
    assert edges[0].earliest_seconds == 10
