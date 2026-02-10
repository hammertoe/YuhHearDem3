from __future__ import annotations

from dataclasses import dataclass

from lib.id_generators import generate_kg_node_id
from lib.knowledge_graph.kg_extractor import (
    ExtractedEdge,
    ExtractedNode,
    ExtractionResult,
    KGExtractor,
)
from lib.knowledge_graph.window_builder import ConceptWindow, Utterance


@dataclass
class _FakeEmbeddingClient:
    def generate_embeddings_batch(
        self, texts: list[str], task_type: str
    ) -> list[list[float]]:
        return [[0.0] * 3 for _ in texts]


class _FakePostgres:
    def __init__(self) -> None:
        self.kg_nodes: set[str] = set()
        self.inserted_edges: list[tuple] = []

    def execute_batch(self, query: str, params_list: list[tuple]) -> None:
        if "INSERT INTO kg_nodes" in query:
            for row in params_list:
                self.kg_nodes.add(row[0])
            return

        if "INSERT INTO kg_edges" in query:
            self.inserted_edges.extend(params_list)
            return

        # kg_aliases / UPDATE kg_nodes embedding are not needed for these unit tests.

    def execute_query(self, query: str, params: tuple | None = None) -> list[tuple]:
        if "FROM speakers" in query:
            return []

        if "FROM kg_nodes" in query and "WHERE id = ANY" in query:
            node_ids = []
            if params:
                node_ids = params[0]
            return [(nid,) for nid in node_ids if nid in self.kg_nodes]

        return []


def _make_extractor(fake_pg: _FakePostgres) -> KGExtractor:
    extractor = KGExtractor.__new__(KGExtractor)
    extractor.postgres = fake_pg
    extractor.embedding = _FakeEmbeddingClient()
    extractor._embed_new_nodes = lambda _node_ids: None  # type: ignore[method-assign]
    return extractor


def test_canonicalize_should_skip_edge_when_speaker_ref_not_in_window() -> None:
    pg = _FakePostgres()
    extractor = _make_extractor(pg)

    window = ConceptWindow(
        utterances=[
            Utterance(
                id="video1:1",
                timestamp_str="0:00:01",
                seconds_since_start=1,
                speaker_id="s_real_1",
                text="Hello world, this is long enough.",
            )
        ],
        window_index=0,
    )

    result = ExtractionResult(
        window=window,
        nodes_new=[],
        edges=[
            ExtractedEdge(
                source_ref="s_fake_1",
                predicate="QUESTIONS",
                target_ref="s_real_1",
                evidence="Hello world",
                utterance_ids=["video1:1"],
                earliest_timestamp="0:00:01",
                earliest_seconds=1,
                confidence=0.5,
            )
        ],
        raw_response="{}",
        parse_success=True,
    )

    stats = extractor.canonicalize_and_store(
        [result], youtube_video_id="video1", kg_run_id="run1", extractor_model="m"
    )

    assert stats["edges"] == 0
    assert stats["edges_skipped_invalid_speaker_ref"] == 1
    assert pg.inserted_edges == []


def test_canonicalize_should_prefix_bare_speaker_id_and_store_edge() -> None:
    pg = _FakePostgres()
    extractor = _make_extractor(pg)

    window = ConceptWindow(
        utterances=[
            Utterance(
                id="video1:1",
                timestamp_str="0:00:01",
                seconds_since_start=1,
                speaker_id="s_real_1",
                text="Hello world, this is long enough.",
            )
        ],
        window_index=0,
    )

    result = ExtractionResult(
        window=window,
        nodes_new=[
            ExtractedNode(
                temp_id="n1",
                type="skos:Concept",
                label="Test Concept",
                aliases=[],
            )
        ],
        edges=[
            ExtractedEdge(
                source_ref="s_real_1",
                predicate="PROPOSES",
                target_ref="n1",
                evidence="Hello world",
                utterance_ids=["video1:1"],
                earliest_timestamp="0:00:01",
                earliest_seconds=1,
                confidence=0.8,
            )
        ],
        raw_response="{}",
        parse_success=True,
    )

    stats = extractor.canonicalize_and_store(
        [result], youtube_video_id="video1", kg_run_id="run1", extractor_model="m"
    )

    assert stats["edges"] == 1
    assert len(pg.inserted_edges) == 1

    _edge_id, source_id, _pred, target_id, *_rest = pg.inserted_edges[0]
    assert source_id == "speaker_s_real_1"
    assert target_id == generate_kg_node_id("skos:Concept", "Test Concept")
