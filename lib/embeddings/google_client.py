"""Google embedding service for generating 768-dim vectors."""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types
from google.genai import errors
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from lib.utils.config import config


class GoogleEmbeddingClient:
    """Google embedding client with retry logic."""

    def __init__(self):
        provider = (config.embedding.provider or "google_ai").strip().lower()
        if provider not in {"google_ai", "vertex_ai"}:
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER={config.embedding.provider!r}; use google_ai or vertex_ai"
            )

        if provider == "google_ai":
            if not config.embedding.api_key:
                raise ValueError("GOOGLE_API_KEY not set in environment")
            self.client = genai.Client(api_key=config.embedding.api_key)
        else:
            if (
                not config.embedding.vertex_project
                or not config.embedding.vertex_location
            ):
                raise ValueError(
                    "VERTEX_PROJECT and VERTEX_LOCATION must be set when EMBEDDING_PROVIDER=vertex_ai"
                )
            self.client = genai.Client(
                vertexai=True,
                project=config.embedding.vertex_project,
                location=config.embedding.vertex_location,
            )

        configured = (config.embedding.model or "").strip()
        # Be flexible: the GenAI SDK has used both "text-embedding-004" and
        # "models/text-embedding-004" style identifiers over time.
        candidates: list[str] = []
        if configured:
            candidates.append(configured)
            if configured.startswith("models/"):
                candidates.append(configured.removeprefix("models/"))
            else:
                candidates.append("models/" + configured)

        # Known defaults. Availability depends on provider/account.
        candidates.extend(
            [
                "gemini-embedding-001",
                "models/gemini-embedding-001",
                "text-embedding-004",
                "models/text-embedding-004",
            ]
        )

        # De-dupe while preserving order.
        seen: set[str] = set()
        self._model_candidates = [
            m for m in candidates if not (m in seen or seen.add(m))
        ]
        self.model = self._model_candidates[0]
        self.dimensions = config.embedding.dimensions
        self.batch_size = config.embedding.batch_size

    def _embed(self, *, text: str, task_type: str) -> Any:
        # The Gemini Embeddings API supports requesting a specific vector size.
        # This must match the pgvector dimension in Postgres (e.g. vector(768)).
        cfg = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=int(self.dimensions) if self.dimensions else None,
        )

        return self.client.models.embed_content(
            model=self.model, contents=text, config=cfg
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate_embedding(
        self, text: str, task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[float]:
        """Generate embedding for a single text."""

        last_err: Exception | None = None
        tried: list[str] = []
        for candidate in self._model_candidates:
            self.model = candidate
            tried.append(candidate)
            try:
                result = self._embed(text=text, task_type=task_type)
                embeddings = getattr(result, "embeddings", None)
                if not embeddings:
                    raise RuntimeError("Embedding response missing embeddings")
                values = getattr(embeddings[0], "values", None)
                if values is None:
                    raise RuntimeError("Embedding response missing values")
                return list(values)
            except errors.ClientError as e:
                last_err = e
                # Common failure mode: configured model doesn't exist / not supported.
                msg = str(getattr(e, "message", "")) or str(e)
                if "NOT_FOUND" in msg or "not found" in msg or "ListModels" in msg:
                    continue
                raise
            except Exception as e:
                last_err = e
                raise

        hint = (
            "No embedding model worked. Run `python scripts/list_genai_models.py` to see which models "
            "support embedContent for your API key, then set EMBEDDING_MODEL accordingly. "
            f"Tried: {tried}"
        )
        raise RuntimeError(hint) from last_err

    def generate_embeddings_batch(
        self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts in batches."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            print(
                f"Generating embeddings for batch {i // self.batch_size + 1} "
                f"({len(batch)} texts)..."
            )

            batch_embeddings = [
                self.generate_embedding(text, task_type) for text in batch
            ]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def generate_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for search query."""
        return self.generate_embedding(query, "RETRIEVAL_QUERY")

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.dimensions
