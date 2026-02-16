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
            #  utterance_ids, evidence, speaker_ids, confidence, edge_rank_score)
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
                    0.12,
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

        if "FROM bill_excerpts be" in sql:
            # (id, bill_id, chunk_index, text, source_url, distance, bill_number, title, page_number)
            return [
                (
                    "bex_1",
                    "bill_water_1",
                    2,
                    "Part IV addresses water quality and potable water systems.",
                    "https://www.barbadosparliament.com/uploads/bill_resolution/sample.pdf",
                    0.09,
                    "BILL-123",
                    "Water Services Bill, 2026",
                    12,
                )
            ]

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


def test_kg_hybrid_graph_rag_respects_bill_citation_limit() -> None:
    from lib.kg_hybrid_graph_rag import kg_hybrid_graph_rag_with_bills

    postgres = _FakePostgres()
    embedding = _FakeEmbedding()

    out = kg_hybrid_graph_rag_with_bills(
        postgres=postgres,
        embedding_client=embedding,
        query="water management",
        hops=1,
        seed_k=5,
        max_edges=20,
        max_citations=5,
        max_bill_citations=1,
    )

    assert len(out["bill_citations"]) == 1


def test_kg_hybrid_graph_rag_keeps_vector_seeds_when_fulltext_is_generic() -> None:
    from lib.kg_hybrid_graph_rag import kg_hybrid_graph_rag

    class _FakePostgresSeedBias(_FakePostgres):
        def execute_query(self, sql: str, params: tuple[Any, ...] | None = None):
            if "FROM kg_nodes" in sql and "embedding <=>" in sql:
                return [
                    (
                        "kg_water",
                        "skos:Concept",
                        "water management",
                        ["water"],
                        0.1,
                    )
                ]

            if "FROM kg_nodes" in sql and "plainto_tsquery" in sql:
                return [
                    (
                        "kg_ministers",
                        "foaf:Group",
                        "ministers",
                        ["minister"],
                        5.0,
                    ),
                    (
                        "kg_ministers_2",
                        "skos:Concept",
                        "ministerial powers",
                        ["ministers"],
                        4.2,
                    ),
                ]

            if "FROM kg_nodes" in sql and "WHERE id IN" in sql:
                return [
                    ("kg_water", "water management", "skos:Concept"),
                    ("kg_ministers", "ministers", "foaf:Group"),
                    ("kg_ministers_2", "ministerial powers", "skos:Concept"),
                ]

            if "FROM kg_edges" in sql:
                return []

            if "FROM sentences" in sql:
                return []

            if "FROM order_papers" in sql and "jsonb_array_elements" in sql:
                return []

            return super().execute_query(sql, params)

    out = kg_hybrid_graph_rag(
        postgres=_FakePostgresSeedBias(),
        embedding_client=_FakeEmbedding(),
        query="What did ministers say about water management recently",
        hops=1,
        seed_k=5,
        max_edges=10,
        max_citations=5,
    )

    assert any("water" in (s.get("label") or "").lower() for s in out["seeds"])


def test_kg_hybrid_graph_rag_falls_back_to_speakers_position_when_no_session_role() -> None:
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
        format_speaker_name(full_name=None, normalized_name="hon santia bradshaw", speaker_id="")
        == "The Honourable Santia Bradshaw"
    )
    assert (
        format_speaker_name(
            full_name="Ralph Thorne", normalized_name="mr ralph thorne", speaker_id=""
        )
        == "Ralph Thorne"
    )


def test_kg_hybrid_graph_rag_with_bills_should_include_page_fragment_and_match_terms() -> None:
    from lib.kg_hybrid_graph_rag import kg_hybrid_graph_rag_with_bills

    out = kg_hybrid_graph_rag_with_bills(
        postgres=_FakePostgres(),
        embedding_client=_FakeEmbedding(),
        query="water quality systems",
        hops=1,
        seed_k=5,
        max_edges=20,
        max_citations=5,
        max_bill_citations=3,
    )

    assert len(out["bill_citations"]) == 1
    bill = out["bill_citations"][0]
    assert bill["page_number"] == 12
    assert bill["source_url"].endswith("#page=12")
    assert "water" in bill["matched_terms"]


def test_fuse_candidates_rrf_should_penalize_generic_governance_when_query_is_topical() -> None:
    from lib.kg_hybrid_graph_rag import _extract_query_intent, _fuse_candidates_rrf

    vector_candidates = [
        {
            "id": "kg_ministers",
            "type": "foaf:Group",
            "label": "ministers",
            "aliases": ["minister"],
        },
        {
            "id": "kg_water",
            "type": "skos:Concept",
            "label": "water management",
            "aliases": ["water"],
        },
    ]
    fulltext_candidates = []
    alias_candidates = []

    query = "What did ministers say about water management recently"
    intent = _extract_query_intent(query)

    fused = _fuse_candidates_rrf(
        vector_candidates=vector_candidates,
        fulltext_candidates=fulltext_candidates,
        alias_candidates=alias_candidates,
        query=query,
        intent=intent,
    )

    assert fused[0]["id"] == "kg_water"
    ministers = next(item for item in fused if item["id"] == "kg_ministers")
    assert ministers["boost"] < 0.0


def test_kg_hybrid_graph_rag_should_keep_edges_when_threshold_applies() -> None:
    from lib.kg_hybrid_graph_rag import kg_hybrid_graph_rag

    out = kg_hybrid_graph_rag(
        postgres=_FakePostgres(),
        embedding_client=_FakeEmbedding(),
        query="water management",
        hops=1,
        seed_k=5,
        max_edges=20,
        max_citations=5,
        edge_rank_threshold=0.05,
    )

    assert len(out["edges"]) == 1
    assert out["edges"][0]["id"] == "kge_1"
