"""Tests for KG chat agent."""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import pytest

from lib.chat_agent import KGChatAgent
from lib.db.postgres_client import PostgresClient
from lib.embeddings.google_client import GoogleEmbeddingClient
from lib.id_generators import normalize_label


@pytest.fixture
def postgres_client() -> PostgresClient:
    client = PostgresClient()
    yield client
    client.close()


@pytest.fixture
def embedding_client() -> GoogleEmbeddingClient:
    client = GoogleEmbeddingClient()
    yield client


@pytest.fixture
def agent(
    postgres_client: PostgresClient, embedding_client: GoogleEmbeddingClient
) -> KGChatAgent:
    return KGChatAgent(
        postgres_client=postgres_client,
        embedding_client=embedding_client,
    )


def test_create_thread(agent: KGChatAgent):
    """Test creating a new thread."""
    title = "Test funding thread"
    thread_id = agent.create_thread(title)

    assert thread_id
    assert len(thread_id) == 36

    thread = agent.get_thread(thread_id)
    assert thread is not None
    assert thread["id"] == thread_id
    assert thread["title"] == title
    assert "messages" in thread
    assert "state" in thread


def test_create_thread_no_title(agent: KGChatAgent):
    """Test creating a thread without title."""
    thread_id = agent.create_thread(None)

    assert thread_id

    thread = agent.get_thread(thread_id)
    assert thread is not None
    assert thread["title"] is None


def test_retrieve_candidate_nodes_empty(agent: KGChatAgent):
    """Test candidate node retrieval with empty results."""
    candidates = agent._retrieve_candidate_nodes("query with no matches", {}, top_k=10)

    assert isinstance(candidates, list)
    assert len(candidates) == 0


def test_retrieve_candidate_nodes_with_embeddings(
    postgres_client: PostgresClient,
    embedding_client: GoogleEmbeddingClient,
    agent: KGChatAgent,
):
    """Test candidate node retrieval via embeddings."""
    postgres_client.execute_update(
        """
            INSERT INTO paragraphs (id, youtube_video_id, start_seconds, end_seconds,
                             text, speaker_id, start_timestamp)
            VALUES (%s, %s, %s, %s, %s, %s),
            """,
        (
            "test_para_1",
            "Syxyah7QIaM",
            0,
            10,
            "Test funding sentence",
            "s_test_speaker_1",
            "00:20:30",
        ),
    )

    postgres_client.execute_update(
        """
            INSERT INTO kg_nodes (id, label, type, aliases)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET label = EXCLUDED.label
            """,
        ("kg_test_funding", "Funding", "skos:Concept", ["money", "support"]),
    )

    embedding_client.generate_embeddings_batch(["Funding"], "RETRIEVAL_DOCUMENT")

    candidates = agent._retrieve_candidate_nodes("funding", {}, top_k=5)

    assert isinstance(candidates, list)
    assert len(candidates) >= 1

    funding_nodes = [c for c in candidates if "fund" in c["label"].lower()]
    assert len(funding_nodes) > 0

    postgres_client.execute_update(
        "DELETE FROM kg_nodes WHERE id = %s", ("kg_test_funding",)
    )
    postgres_client.execute_update(
        "DELETE FROM paragraphs WHERE id = %s", ("test_para_1",)
    )


def test_retrieve_sentences_for_utterances(
    postgres_client: PostgresClient,
    agent: KGChatAgent,
):
    """Test sentence retrieval from utterance IDs."""
    postgres_client.execute_update(
        """
            INSERT INTO sentences (id, text, seconds_since_start, timestamp_str,
                              youtube_video_id, speaker_id, paragraph_id, sentence_order)
            VALUES (%s, %s, %s, %s, %s, %s),
            """,
        (
            "test_utt_2",
            "This is a test sentence about funding.",
            2345,
            "00:20:35",
            "Syxyah7QIaM",
            "s_test_speaker_1",
            "test_para_2",
            1,
        ),
    )

    sentences = agent._retrieve_sentences_for_utterances(["test_utt_2"])

    assert len(sentences) == 1
    assert sentences[0]["id"] == "test_utt_2"
    assert "funding" in sentences[0]["text"].lower()
    postgres_client.execute_update(
        "DELETE FROM sentences WHERE id = %s", ("test_utt_2",)
    )


def test_normalize_label():
    """Test label normalization."""
    assert normalize_label("Funding for Schools") == "funding for schools"
    assert normalize_label("  SPORT  ") == "sport"
    assert normalize_label("multiple   spaces") == "multiple spaces"
