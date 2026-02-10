# Tests for advanced search features.
import pytest
from unittest.mock import Mock

from lib.advanced_search_features import AdvancedSearchFeatures


@pytest.fixture
def sample_sentences_data():
    """Sample sentence data for testing."""
    return [
        {
            "id": "test_video:00:36:00",
            "text": "Let us pray.",
            "seconds_since_start": 2160,
            "timestamp_str": "00:36:00",
            "video_id": "test_video",
            "speaker_id": "s_reverend_1",
            "paragraph_id": "test_video:2160",
            "speaker_name": "reverend",
            "video_date": "2026-01-06",
            "video_title": "Test Video",
        },
        {
            "id": "test_video:00:36:34",
            "text": "Almighty God, our help in ages past.",
            "seconds_since_start": 2194,
            "timestamp_str": "00:36:34",
            "video_id": "test_video",
            "speaker_id": "s_reverend_1",
            "paragraph_id": "test_video:2160",
            "speaker_name": "reverend",
            "video_date": "2026-01-06",
            "video_title": "Test Video",
        },
    ]


def test_advanced_search_init():
    """Test advanced search initialization."""
    features = AdvancedSearchFeatures(
        postgres=Mock(),
        memgraph=Mock(),
        embedding_client=Mock(),
    )
    assert features.postgres is not None
    assert features.memgraph is not None
    assert features.embedding_client is not None
    print("✅ Advanced search initialization works")


def test_temporal_search():
    """Test temporal search functionality."""
    postgres = Mock()
    embedding_client = Mock()
    embedding_client.generate_query_embedding.return_value = [0.0] * 768

    features = AdvancedSearchFeatures(
        postgres=postgres,
        memgraph=Mock(),
        embedding_client=embedding_client,
    )

    postgres.execute_query.return_value = [
        (
            "test_video:2160",
            "Let us pray.",
            2160,
            "00:36:00",
            "test_video",
            "Test Video",
            "2026-01-06",
            "s_reverend_1",
            "reverend",
            "test_video:2160",
            0.1,
        )
    ]

    results = features.temporal_search(
        "test query", "2026-01-01", "2026-01-31", None, None, 10
    )

    assert len(results) > 0
    assert all("score" in r for r in results)
    assert results[0]["search_type"] == "temporal"
    assert results[0]["video_id"] == "test_video"
    assert results[0]["speaker_id"] == "s_reverend_1"

    print("✅ Temporal search works")


def test_trend_analysis():
    """Test trend analysis functionality."""
    postgres = Mock()
    features = AdvancedSearchFeatures(
        postgres=postgres,
        memgraph=Mock(),
        embedding_client=Mock(),
    )

    postgres.execute_query.return_value = [
        ("2026-01-01", 5),
        ("2026-01-02", 3),
        ("2026-01-03", 7),
    ]

    result = features.trend_analysis("test_entity_id", 30, 10)

    assert "entity_id" in result
    assert "trends" in result
    assert "summary" in result
    assert len(result["trends"]) > 0

    print("✅ Trend analysis works")


def test_moving_average_calculation():
    """Test moving average calculation."""
    features = AdvancedSearchFeatures()

    data = [
        {"date": "2026-01-01", "mentions": 5},
        {"date": "2026-01-02", "mentions": 7},
        {"date": "2026-01-03", "mentions": 3},
        {"date": "2026-01-04", "mentions": 8},
        {"date": "2026-01-05", "mentions": 12},
    ]

    result = features._calculate_moving_average(data, window_size=3)

    assert len(result) > 0
    assert len(result) == len(data)
    assert result[0]["date"] == "2026-01-01"
    assert result[1]["value"] == pytest.approx(6.0)

    print("✅ Moving average calculation works")


def test_multi_hop_query():
    """Test multi-hop graph traversal."""
    memgraph = Mock()
    features = AdvancedSearchFeatures(
        postgres=Mock(),
        memgraph=memgraph,
        embedding_client=Mock(),
    )

    memgraph.execute_query.return_value = [
        {
            "start_entity": "start_id",
            "related_entity": "related_1",
            "relationship_type": "DISCUSSES",
            "labels": ["Topic"],
        },
        {
            "start_entity": "start_id",
            "related_entity": "related_2",
            "relationship_type": "DISCUSSES",
            "labels": ["Topic"],
        },
    ]

    results = features.multi_hop_query("start_id", 2)

    assert len(results) == 2
    assert results[0]["related_entity"] == "related_1"
    assert results[1]["related_entity"] == "related_2"
    assert all("relationship_type" in r for r in results)

    print("✅ Multi-hop query works")


def test_complex_query_speaker_influence():
    """Test speaker influence query."""
    memgraph = Mock()
    features = AdvancedSearchFeatures(
        postgres=Mock(),
        memgraph=memgraph,
        embedding_client=Mock(),
    )

    memgraph.execute_query.return_value = [
        {
            "speaker_id": "s_speaker_1",
            "topic_id": "topic_id",
            "topic_text": "Traffic safety",
            "count": 15,
        }
    ]

    result = features.complex_query("speaker_influence", {"max_results": 20})

    assert result["query_type"] == "speaker_influence"
    assert result["count"] == 1
    assert result["results"][0]["speaker_id"] == "s_speaker_1"
    assert result["results"][0]["topic_text"] == "Traffic safety"

    print("✅ Complex query works")


def test_complex_query_bill_connections():
    """Test bill connections query."""
    memgraph = Mock()
    features = AdvancedSearchFeatures(
        postgres=Mock(),
        memgraph=memgraph,
        embedding_client=Mock(),
    )

    memgraph.execute_query.return_value = [
        {"bill_id": "bill_id", "bill_title": "Bill Title", "speaker_count": 5}
    ]

    result = features.complex_query("bill_connections", {"max_results": 20})

    assert result["query_type"] == "bill_connections"
    assert result["count"] == 1
    assert result["results"][0]["bill_id"] == "bill_id"

    print("✅ Complex query works")


def test_complex_query_controversial_topics():
    """Test controversial topics query."""
    memgraph = Mock()
    features = AdvancedSearchFeatures(
        postgres=Mock(),
        memgraph=memgraph,
        embedding_client=Mock(),
    )

    memgraph.execute_query.return_value = [
        {
            "topic": "topic_id",
            "agree_count": 10,
            "disagreeing_speaker_1": "speaker_1",
            "disagreeing_speaker_2": "speaker_2",
        }
    ]

    result = features.complex_query("controversial_topics", {"max_results": 20})

    assert result["query_type"] == "controversial_topics"
    assert result["count"] == 1
    assert result["results"][0]["agree_count"] == 10

    print("✅ Complex query works")


def test_invalid_query_type():
    """Test error handling for invalid query type."""
    features = AdvancedSearchFeatures(
        postgres=Mock(),
        memgraph=Mock(),
        embedding_client=Mock(),
    )

    result = features.complex_query("invalid_query", {"max_results": 20})

    assert result["query_type"] == "invalid_query"
    assert result["results"] == []

    print("✅ Invalid query type handling works")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
