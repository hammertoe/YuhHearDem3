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
        embedding_client=Mock(),
    )
    assert features.postgres is not None
    assert features.embedding_client is not None
    print("✅ Advanced search initialization works")


def test_temporal_search():
    """Test temporal search functionality."""
    postgres = Mock()
    embedding_client = Mock()
    embedding_client.generate_query_embedding.return_value = [0.0] * 768

    features = AdvancedSearchFeatures(
        postgres=postgres,
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

    results = features.temporal_search("test query", "2026-01-01", "2026-01-31", None, None, 10)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
