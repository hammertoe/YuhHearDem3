from __future__ import annotations

from typing import Any


class _FakePostgres:
    def __init__(self) -> None:
        self.queries: list[tuple[str, tuple[Any, ...] | None]] = []

    def execute_query(self, sql: str, params: tuple[Any, ...] | None = None):
        self.queries.append((sql, params))

        if "FROM kg_nodes" in sql and "embedding <=>" in sql:
            # (id, type, label, aliases, distance)
            return [
                (
                    "kg_a",
                    "skos:Concept",
                    "Water Management",
                    ["water", "water policy"],
                    0.12,
                ),
            ]

        if "FROM kg_nodes" in sql and "plainto_tsquery" in sql:
            # (id, type, label, aliases, rank)
            return [
                (
                    "kg_a",
                    "skos:Concept",
                    "Water Management",
                    ["water", "water policy"],
                    0.9,
                ),
            ]

        if "FROM kg_aliases" in sql:
            return []

        if "FROM kg_edges" in sql:
            # (id, source_id, predicate, predicate_raw, target_id,
            #  youtube_video_id, earliest_timestamp_str, earliest_seconds,
            #  utterance_ids, evidence, speaker_ids, confidence)
            return [
                (
                    "kge_1",
                    "kg_a",
                    "DISCUSSES",
                    "discusses",
                    "kg_b",
                    "Syxyah7QIaM",
                    "00:20:35",
                    1235,
                    ["utt_1"],
                    "They discussed water management policy.",
                    ["s_test_1"],
                    0.77,
                )
            ]

        if "FROM kg_nodes" in sql and "WHERE id IN" in sql:
            # (id, label, type)
            return [
                ("kg_a", "Water Management", "skos:Concept"),
                ("kg_b", "National Water Authority", "schema:Organization"),
            ]

        if "FROM sentences" in sql:
            # (id, text, seconds_since_start, timestamp_str, youtube_video_id,
            #  video_date, video_title, speaker_id, full_name, normalized_name, title, position, speaker_title)
            return [
                (
                    "utt_1",
                    "We need to address water management.",
                    1235,
                    "00:20:35",
                    "Syxyah7QIaM",
                    "2025-01-01",
                    "Parliament Sitting",
                    "s_santia_bradshaw_1",
                    None,
                    "santia bradshaw",
                    None,
                    None,
                    "Minister",
                )
            ]

        if "FROM order_papers" in sql and "jsonb_array_elements" in sql:
            # (sitting_date, name, title, role)
            return [("2026-01-06", "Santia Bradshaw", "Hon.", "Minister")]

        return []


class _FakeEmbedding:
    def generate_query_embedding(self, _query: str) -> list[float]:
        return [0.0] * 768


def test_build_youtube_url_with_timecode():
    from lib.kg_hybrid_graph_rag import build_youtube_url

    assert (
        build_youtube_url("Syxyah7QIaM", 1235)
        == "https://www.youtube.com/watch?v=Syxyah7QIaM&t=1235s"
    )


def test_kg_hybrid_graph_rag_returns_compact_subgraph():
    from lib.kg_hybrid_graph_rag import kg_hybrid_graph_rag

    postgres = _FakePostgres()
    embedding = _FakeEmbedding()

    out = kg_hybrid_graph_rag(
        postgres=postgres,
        embedding_client=embedding,
        query="water management",
        hops=1,
        seed_k=5,
        max_edges=20,
        max_citations=5,
    )

    assert out["query"] == "water management"
    assert out["hops"] == 1
    assert len(out["seeds"]) >= 1
    assert any(s["id"] == "kg_a" for s in out["seeds"])

    node_ids = {n["id"] for n in out["nodes"]}
    assert {"kg_a", "kg_b"}.issubset(node_ids)

    assert len(out["edges"]) == 1
    assert out["edges"][0]["id"] == "kge_1"
    assert out["edges"][0]["source_label"] == "Water Management"
    assert out["edges"][0]["target_label"] == "National Water Authority"

    assert len(out["citations"]) == 1
    c = out["citations"][0]
    assert c["utterance_id"] == "utt_1"
    assert c["youtube_video_id"] == "Syxyah7QIaM"
    assert c["youtube_url"].endswith("&t=1235s")
    assert c["speaker_name"] == "The Honourable Santia Bradshaw"
    assert c["speaker_title"] == "Minister"


def test_kg_hybrid_graph_rag_falls_back_to_speakers_position_when_no_session_role() -> (
    None
):
    from lib.kg_hybrid_graph_rag import kg_hybrid_graph_rag

    class _FakePostgresNoRole(_FakePostgres):
        def execute_query(self, sql: str, params: tuple[Any, ...] | None = None):
            if "FROM sentences" in sql:
                return [
                    (
                        "utt_1",
                        "We need to address water management.",
                        1235,
                        "00:20:35",
                        "Syxyah7QIaM",
                        "2025-01-01",
                        "Parliament Sitting",
                        "s_santia_bradshaw_1",
                        None,
                        "santia bradshaw",
                        None,
                        "Minister of Transport",
                        None,
                    )
                ]
            return super().execute_query(sql, params)

    out = kg_hybrid_graph_rag(
        postgres=_FakePostgresNoRole(),
        embedding_client=_FakeEmbedding(),
        query="water management",
        hops=1,
        seed_k=5,
        max_edges=20,
        max_citations=5,
    )

    assert out["citations"][0]["speaker_title"] == "Minister of Transport"


def test_format_speaker_name_prefers_full_name_and_title_cases_normalized() -> None:
    from lib.kg_hybrid_graph_rag import format_speaker_name

    assert (
        format_speaker_name(
            full_name=None, normalized_name="hon santia bradshaw", speaker_id=""
        )
        == "The Honourable Santia Bradshaw"
    )
    assert (
        format_speaker_name(
            full_name="Ralph Thorne", normalized_name="mr ralph thorne", speaker_id=""
        )
        == "Ralph Thorne"
    )
