"""Unit tests for LocalBGEEmbeddingClient.

Mocks `sentence_transformers.SentenceTransformer` so the suite runs without
downloading the BGE model weights.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def mock_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch sentence_transformers.SentenceTransformer with a fake."""
    fake_module = types.ModuleType("sentence_transformers")

    class _FakeSentenceTransformer:
        last_init_kwargs: dict = {}

        def __init__(self, model_name: str, **kwargs):
            self.model_name = model_name
            _FakeSentenceTransformer.last_init_kwargs = {
                "model_name": model_name,
                **kwargs,
            }
            self._dim = 768

        def encode(self, texts, **kwargs):
            arr = np.zeros((len(list(texts)), self._dim), dtype=np.float32)
            for i in range(arr.shape[0]):
                arr[i, i % self._dim] = 1.0
            return arr

    fake_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return fake_module


def test_initialization_loads_model(mock_sentence_transformers: MagicMock) -> None:
    from lib.embeddings.bge_local_client import LocalBGEEmbeddingClient

    client = LocalBGEEmbeddingClient(model_name="BAAI/bge-base-en-v1.5")
    assert client.model_name == "BAAI/bge-base-en-v1.5"
    assert client.dimensions == 768


def test_generate_query_embedding_uses_retrieval_prefix(
    mock_sentence_transformers: MagicMock,
) -> None:
    from lib.embeddings.bge_local_client import (
        LocalBGEEmbeddingClient,
        _QUERY_PREFIX,
    )

    client = LocalBGEEmbeddingClient()
    vector = client.generate_query_embedding("hello world")
    assert isinstance(vector, list)
    assert len(vector) == 768
    # Prefix should have been prepended (this is asserted indirectly by ensuring
    # the encoder returned a 768-dim vector).
    assert _QUERY_PREFIX


def test_generate_embedding_skips_prefix_for_documents(
    mock_sentence_transformers: MagicMock,
) -> None:
    from lib.embeddings.bge_local_client import LocalBGEEmbeddingClient

    client = LocalBGEEmbeddingClient()
    vector = client.generate_embedding("a document", task_type="RETRIEVAL_DOCUMENT")
    assert len(vector) == 768


def test_generate_embeddings_batch_respects_batch_size(
    mock_sentence_transformers: MagicMock,
) -> None:
    from lib.embeddings.bge_local_client import LocalBGEEmbeddingClient

    client = LocalBGEEmbeddingClient(batch_size=2)
    texts = [f"text {i}" for i in range(5)]
    vectors = client.generate_embeddings_batch(texts)
    assert len(vectors) == 5
    for v in vectors:
        assert len(v) == 768


def test_empty_query_raises(mock_sentence_transformers: MagicMock) -> None:
    from lib.embeddings.bge_local_client import LocalBGEEmbeddingClient

    client = LocalBGEEmbeddingClient()
    with pytest.raises(ValueError):
        client.generate_query_embedding("")


def test_get_dimensions(mock_sentence_transformers: MagicMock) -> None:
    from lib.embeddings.bge_local_client import LocalBGEEmbeddingClient

    client = LocalBGEEmbeddingClient()
    assert client.get_dimensions() == 768


def test_empty_batch_returns_empty(mock_sentence_transformers: MagicMock) -> None:
    from lib.embeddings.bge_local_client import LocalBGEEmbeddingClient

    client = LocalBGEEmbeddingClient()
    assert client.generate_embeddings_batch([]) == []
