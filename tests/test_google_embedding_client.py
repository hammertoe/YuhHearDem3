from __future__ import annotations

from types import SimpleNamespace

import pytest

from lib.utils.config import config


class _DummyModels:
    def __init__(self):
        self.calls: list[dict] = []

    def embed_content(self, *, model: str, contents: str, config: dict):
        self.calls.append({"model": model, "contents": contents, "config": config})
        dim = int(getattr(config, "output_dimensionality", None) or 0)
        if not dim:
            dim = 3072
        values = [0.0] * dim
        return SimpleNamespace(embeddings=[SimpleNamespace(values=values)])


class _DummyClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.models = _DummyModels()


def test_generate_embedding_passes_output_dimensionality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lib.embeddings import google_client as mod

    # Configure for test.
    config.embedding.provider = "google_ai"
    config.embedding.api_key = "test"
    config.embedding.model = "gemini-embedding-001"
    config.embedding.dimensions = 768

    monkeypatch.setattr(mod.genai, "Client", lambda **kwargs: _DummyClient(**kwargs))

    client = mod.GoogleEmbeddingClient()
    vec = client.generate_embedding("hello", task_type="RETRIEVAL_DOCUMENT")

    assert len(vec) == 768
    cfg = client.client.models.calls[0]["config"]
    assert getattr(cfg, "task_type") == "RETRIEVAL_DOCUMENT"
    assert getattr(cfg, "output_dimensionality") == 768
