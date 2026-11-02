"""Embedding providers.

``huggingface`` uses sentence-transformers (``all-MiniLM-L6-v2`` by default) and
is what production runs. ``hash`` is a deterministic bag-of-words projection that
requires no model download, which keeps the test suite hermetic and fast while
still producing usable cosine similarities for assertions.
"""

from __future__ import annotations

import abc
import hashlib
import math
import re
import threading
from collections.abc import Sequence
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


class EmbeddingProvider(abc.ABC):
    """Contract for text -> vector conversion."""

    model_name: str = "unknown"
    dimension: int = 384

    @abc.abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents (one vector each)."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_documents([text])[0]

    def health(self) -> dict[str, Any]:
        return {
            "provider": type(self).__name__,
            "model": self.model_name,
            "dimension": self.dimension,
            "status": "ok",
        }


class HashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dependency-free embedding (hashing trick + L2 norm).

    Not semantically strong, but it is *stable*, offline and good enough to
    verify that retrieval plumbing, ranking and thresholds behave correctly.
    """

    def __init__(self, *, dimension: int = 384, normalize: bool = True) -> None:
        self.dimension = max(int(dimension), 8)
        self.model_name = f"hash-embedding-{self.dimension}"
        self._normalize = normalize

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _TOKEN.findall(text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        if self._normalize:
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text or "") for text in texts]


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers embeddings, loaded lazily and cached per process."""

    _models: dict[str, Any] = {}
    _lock = threading.Lock()

    def __init__(self, config: Settings | None = None) -> None:
        self._config = config or get_settings()
        self.model_name = self._config.embedding_model
        self.dimension = self._config.embedding_dimension
        self._model: Any | None = None

    def _ensure_loaded(self) -> Any:  # noqa: ANN401 - SentenceTransformer
        if self._model is not None:
            return self._model

        with self._lock:
            cached = self._models.get(self.model_name)
            if cached is not None:
                self._model = cached
                return cached

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - dependency missing
                raise EmbeddingError(
                    "sentence-transformers is not installed. "
                    "Set EMBEDDING_PROVIDER=hash or install it.",
                    code="sentence_transformers_missing",
                ) from exc

            logger.info("Loading embedding model '%s'", self.model_name)
            try:
                model = SentenceTransformer(
                    self.model_name, cache_folder=self._config.hf_home or None
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to load embedding model '{self.model_name}': {exc}",
                    code="model_load_failed",
                ) from exc

            self._models[self.model_name] = model
            self._model = model
            logger.info("Embedding model '%s' ready.", self.model_name)
            return model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_loaded()
        try:
            vectors = model.encode(
                list(texts),
                batch_size=self._config.embedding_batch_size,
                normalize_embeddings=self._config.embedding_normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
        return [[float(value) for value in vector] for vector in vectors]

    def health(self) -> dict[str, Any]:
        return {
            "provider": "huggingface",
            "model": self.model_name,
            "dimension": self.dimension,
            "loaded": self._model is not None,
            "status": "ok",
        }


# --------------------------------------------------------------------------- #
# Process-wide singleton
# --------------------------------------------------------------------------- #
_provider_lock = threading.Lock()
_provider: EmbeddingProvider | None = None


def build_embedding_provider(config: Settings | None = None) -> EmbeddingProvider:
    """Construct an embedding provider from settings (no caching)."""
    config = config or get_settings()
    if config.embedding_provider == "hash":
        logger.info("Embedding provider: hash (%s dims)", config.embedding_dimension)
        return HashingEmbeddingProvider(
            dimension=config.embedding_dimension, normalize=config.embedding_normalize
        )
    logger.info("Embedding provider: huggingface (%s)", config.embedding_model)
    return HuggingFaceEmbeddingProvider(config=config)


def get_embedding_provider() -> EmbeddingProvider:
    """Return the lazily created process-wide embedding provider."""
    global _provider  # noqa: PLW0603
    if _provider is None:
        with _provider_lock:
            if _provider is None:
                _provider = build_embedding_provider()
    return _provider


def reset_embedding_provider(provider: EmbeddingProvider | None = None) -> None:
    """Replace (or clear) the singleton - used by tests."""
    global _provider  # noqa: PLW0603
    _provider = provider
