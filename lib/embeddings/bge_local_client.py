"""Local BGE embedding client using sentence-transformers.

Provides an offline embedding service backed by BAAI/bge-base-en-v1.5
(768 dimensions, cosine-similarity friendly). Drop-in replacement for
`GoogleEmbeddingClient` exposing the same surface used by the app
(`generate_query_embedding`, `generate_embeddings_batch`, `get_dimensions`).
"""

from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalBGEEmbeddingClient:
    """Local sentence-transformers backed embedding client."""

    DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
    DEFAULT_DIMENSIONS = 768
    DEFAULT_BATCH_SIZE = 32

    def __init__(
        self,
        model_name: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
        cache_dir: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        self.model_name = model_name or os.getenv("LOCAL_EMBEDDING_MODEL", self.DEFAULT_MODEL)
        self.dimensions = dimensions or int(
            os.getenv("EMBEDDING_DIMENSIONS", str(self.DEFAULT_DIMENSIONS))
        )
        self.batch_size = batch_size or int(
            os.getenv("EMBEDDING_BATCH_SIZE", str(self.DEFAULT_BATCH_SIZE))
        )

        load_kwargs: dict[str, Any] = {}
        if cache_dir:
            load_kwargs["cache_folder"] = cache_dir

        logger.info(
            "Loading local embedding model %s (this may take a moment on first run)",
            self.model_name,
        )
        self._model = SentenceTransformer(self.model_name, **load_kwargs)
        self._lock = Lock()

    def _encode(self, texts: list[str], *, normalize: bool) -> list[list[float]]:
        with self._lock:
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return [v.tolist() for v in vectors]

    def generate_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for a search query (applies BGE retrieval prefix)."""
        if not query:
            raise ValueError("query must be a non-empty string")
        prefixed = _QUERY_PREFIX + query if not query.startswith(_QUERY_PREFIX) else query
        vectors = self._encode([prefixed], normalize=True)
        return vectors[0]

    def generate_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        """Generate embedding for a single document (no retrieval prefix)."""
        if not text:
            raise ValueError("text must be a non-empty string")
        vectors = self._encode([text], normalize=True)
        return vectors[0]

    def generate_embeddings_batch(
        self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        """Generate embeddings for multiple documents in batches."""
        if not texts:
            return []
        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            all_vectors.extend(self._encode(batch, normalize=True))
        return all_vectors

    def get_dimensions(self) -> int:
        """Return the embedding dimensionality produced by this client."""
        return self.dimensions
