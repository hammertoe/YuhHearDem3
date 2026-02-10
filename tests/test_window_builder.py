"""Unit tests for Window builder."""

import pytest

from lib.knowledge_graph.window_builder import (
    ConceptWindow,
    DiscourseWindow,
    Utterance,
    Window,
    WindowBuilder,
)


@pytest.fixture
def mock_postgres():
    """Mock PostgresClient."""

    class MockPostgres:
        def __init__(self):
            self._utterances = []

        def set_utterances(self, utterances):
            """Set utterances to return."""
            self._utterances = [
                (
                    u.id,
                    u.timestamp_str,
                    u.seconds_since_start,
                    u.speaker_id,
                    u.text,
                )
                for u in utterances
            ]

        def execute_query(self, query, params=None):
            return self._utterances

    return MockPostgres()


@pytest.fixture
def sample_utterances():
    """Sample utterances for testing."""
    return [
        Utterance(
            id="video1:10",
            timestamp_str="0:00:10",
            seconds_since_start=10,
            speaker_id="speaker_a",
            text="Speaker A says something important.",
        ),
        Utterance(
            id="video1:20",
            timestamp_str="0:00:20",
            seconds_since_start=20,
            speaker_id="speaker_a",
            text="Speaker A continues speaking.",
        ),
        Utterance(
            id="video1:30",
            timestamp_str="0:00:30",
            seconds_since_start=30,
            speaker_id="speaker_b",
            text="Speaker B responds.",
        ),
        Utterance(
            id="video1:40",
            timestamp_str="0:00:40",
            seconds_since_start=40,
            speaker_id="speaker_b",
            text="Speaker B adds more.",
        ),
        Utterance(
            id="video1:50",
            timestamp_str="0:00:50",
            seconds_since_start=50,
            speaker_id="speaker_a",
            text="Speaker A responds back.",
        ),
    ]


def test_utterance_creation():
    """Test Utterance creation."""
    utterance = Utterance(
        id="video1:10",
        timestamp_str="0:00:10",
        seconds_since_start=10,
        speaker_id="speaker_a",
        text="Hello world.",
    )

    assert utterance.id == "video1:10"
    assert utterance.timestamp_str == "0:00:10"
    assert utterance.seconds_since_start == 10
    assert utterance.speaker_id == "speaker_a"
    assert utterance.text == "Hello world."


def test_utterance_from_row():
    """Test Utterance creation from database row."""
    row = ("video1:10", "0:00:10", 10, "speaker_a", "Hello world.")
    utterance = Utterance.from_row(row)

    assert utterance.id == "video1:10"
    assert utterance.timestamp_str == "0:00:10"
    assert utterance.seconds_since_start == 10
    assert utterance.speaker_id == "speaker_a"
    assert utterance.text == "Hello world."


def test_window_properties():
    """Test Window properties."""
    utterances = [
        Utterance(
            id="video1:10",
            timestamp_str="0:00:10",
            seconds_since_start=10,
            speaker_id="speaker_a",
            text="Hello.",
        ),
        Utterance(
            id="video1:20",
            timestamp_str="0:00:20",
            seconds_since_start=20,
            speaker_id="speaker_b",
            text="Hi there.",
        ),
    ]

    window = Window(utterances=utterances)

    assert "Hello." in window.text
    assert "Hi there." in window.text
    assert "utterance_id=video1:10" in window.text
    assert window.utterance_ids == ["video1:10", "video1:20"]
    assert set(window.speaker_ids) == {"speaker_a", "speaker_b"}
    assert window.earliest_timestamp == "0:00:10"
    assert window.earliest_seconds == 10


def test_concept_window_creation():
    """Test ConceptWindow creation."""
    window = ConceptWindow(
        utterances=[],
        window_size=10,
        stride=6,
        window_index=0,
    )

    assert window.window_type == "concept"
    assert window.window_size == 10
    assert window.stride == 6
    assert window.window_index == 0


def test_discourse_window_creation():
    """Test DiscourseWindow creation."""
    window = DiscourseWindow(
        utterances=[],
        transition_from="speaker_a",
        transition_to="speaker_b",
        window_index=0,
    )

    assert window.window_type == "discourse"
    assert window.transition_from == "speaker_a"
    assert window.transition_to == "speaker_b"
    assert window.window_index == 0


def test_build_concept_windows(mock_postgres, sample_utterances):
    """Test building concept windows."""
    builder = WindowBuilder(mock_postgres)

    windows = builder.build_concept_windows(
        sample_utterances,
        window_size=3,
        stride=2,
        filter_short=False,
    )

    assert len(windows) == 2
    assert all(isinstance(w, ConceptWindow) for w in windows)

    first_window = windows[0]
    assert len(first_window.utterances) == 3
    assert first_window.utterances[0].speaker_id == "speaker_a"


def test_build_concept_windows_with_filter(mock_postgres):
    """Test building concept windows with filtering."""
    short_utterance = Utterance(
        id="video1:5",
        timestamp_str="0:00:05",
        seconds_since_start=5,
        speaker_id="speaker_a",
        text="Yes.",
    )

    utterances = [
        short_utterance,
        Utterance(
            id="video1:10",
            timestamp_str="0:00:10",
            seconds_since_start=10,
            speaker_id="speaker_a",
            text="A longer utterance with more content.",
        ),
    ]

    builder = WindowBuilder(mock_postgres)

    windows = builder.build_concept_windows(
        utterances,
        window_size=2,
        stride=2,
        filter_short=True,
    )

    assert len(windows) == 0


def test_build_discourse_windows(mock_postgres, sample_utterances):
    """Test building discourse windows."""
    builder = WindowBuilder(mock_postgres)

    windows = builder.build_discourse_windows(
        sample_utterances,
        context_size=1,
    )

    transitions = [
        ("speaker_a", "speaker_b"),
        ("speaker_b", "speaker_a"),
    ]

    assert len(windows) == 2
    assert all(isinstance(w, DiscourseWindow) for w in windows)

    for i, window in enumerate(windows):
        assert window.transition_from == transitions[i][0]
        assert window.transition_to == transitions[i][1]


def test_build_all_windows(mock_postgres, sample_utterances):
    """Test building all windows."""
    mock_postgres.set_utterances(sample_utterances)
    builder = WindowBuilder(mock_postgres)

    all_windows = builder.build_all_windows(
        "video1",
        window_size=3,
        stride=2,
        context_size=1,
        filter_short=False,
    )

    assert "concept" in all_windows
    assert "discourse" in all_windows

    concept_windows = all_windows["concept"]
    discourse_windows = all_windows["discourse"]

    assert len(concept_windows) > 0
    assert len(discourse_windows) > 0


def test_format_known_nodes():
    """Test formatting known nodes for LLM prompt."""
    candidates = [
        {
            "id": "speaker_a",
            "type": "foaf:Person",
            "label": "John Doe",
            "aliases": ["john", "mr doe"],
        },
        {
            "id": "kg_abc123",
            "type": "skos:Concept",
            "label": "Tax Reform",
            "aliases": ["tax", "reform"],
        },
    ]

    class _DummyPostgres:
        def execute_query(self, query, params=None):
            return []

    builder = WindowBuilder(_DummyPostgres())  # type: ignore[arg-type]

    table = builder.format_known_nodes(candidates)

    assert "| ID | Type | Label | Aliases |" in table
    assert "speaker_a" in table
    assert "kg_abc123" in table
    assert "John Doe" in table
    assert "Tax Reform" in table
