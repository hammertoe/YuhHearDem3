from __future__ import annotations

from lib.knowledge_graph.oss_two_pass import normalize_utterance_ids_in_data


def test_normalize_utterance_ids_should_prefix_bare_seconds() -> None:
    data = {
        "nodes_new": [],
        "edges": [
            {"utterance_ids": ["1851", "Syxyah7QIaM:1873"]},
        ],
    }
    normalize_utterance_ids_in_data(data, youtube_video_id="Syxyah7QIaM")
    assert data["edges"][0]["utterance_ids"] == ["Syxyah7QIaM:1851", "Syxyah7QIaM:1873"]


def test_normalize_utterance_ids_should_handle_edges_add() -> None:
    data = {
        "edges_add": [
            {"utterance_ids": [1857]},
        ]
    }
    normalize_utterance_ids_in_data(data, youtube_video_id="vid")
    assert data["edges_add"][0]["utterance_ids"] == ["vid:1857"]
