"""Embedded & persistent ChromaDB vector store wrapper.

ChromaDB runs *inside* the Python process and persists to the local disk
(``CHROMA_PERSIST_DIR``) - no separate Chroma server or Docker container is
required. Chroma is imported lazily so the rest of the application (and the
unit tests) can run without the dependency installed.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Upper bound on rows fetched by the ANN-index drift probe. Diagnostics must
#: stay cheap, so the probe is bounded rather than exhaustive.
DRIFT_PROBE_LIMIT = 1_000

# --------------------------------------------------------------------------- #
# Chroma product telemetry
#
# Chroma routes every telemetry event through ``posthog.capture``, whose
# signature changed between posthog v2/v3 (positional
# ``capture(distinct_id, event, properties)``) and the modern releases
# (``capture(event, **kwargs)``):
#
#     # chromadb/telemetry/product/posthog.py (0.6.3)
#     posthog.capture(self.user_id, event.name, {...})
#
#     # posthog >= 4
#     def capture(event: str, **kwargs) -> Optional[str]: ...
#
# ``chromadb`` declares only ``posthog>=2.4.0`` with *no* upper bound, so a
# current posthog gets installed and every event logs::
#
#     Failed to send telemetry event ClientStartEvent:
#     capture() takes 1 positional argument but 3 were given
#
# ``anonymized_telemetry=False`` does NOT prevent this - it only sets
# ``posthog.disabled = True``, which skips the network send while still running
# the mismatched call. Chroma resolves its telemetry client by name from the
# ``chroma_product_telemetry_impl`` setting, so pointing that at the no-op class
# below removes the call entirely. That is version-independent and the honest
# behaviour for a self-hosted deployment that never intended to phone home.
#
# Subclassing ``ProductTelemetryClient`` is required, not cosmetic: Chroma's
# ``System`` treats it as a ``Component`` and calls ``dependencies()``,
# ``start()``, ``stop()`` and ``reset_state()`` on it. Inheriting supplies that
# machinery, and its ``EnforceOverrides`` metaclass then requires ``@override``.
#
# Guarded so this module keeps importing when chromadb is absent (the rest of
# the app, and the unit tests, do not need it at import time).
# --------------------------------------------------------------------------- #
try:
    from chromadb.telemetry.product import (
        ProductTelemetryClient,
        ProductTelemetryEvent,
    )
    from overrides import override

    class NoopProductTelemetryClient(ProductTelemetryClient):
        """Discards every telemetry event instead of sending it to posthog."""

        @override
        def capture(self, event: ProductTelemetryEvent) -> None:
            """Accept and discard a telemetry event."""
            return None

except ImportError:  # pragma: no cover - chromadb not installed
    NoopProductTelemetryClient = None  # type: ignore[assignment,misc]


def _build_chroma_settings() -> Any:  # noqa: ANN401 - chroma Settings duck type
    """Build Chroma settings with product telemetry fully disabled.

    ``chroma_product_telemetry_impl`` swaps Chroma's posthog-backed client for
    the no-op class defined above - see the comment there for why muting it with
    ``anonymized_telemetry`` alone is not enough.
    """
    from chromadb.config import Settings as ChromaSettings

    return ChromaSettings(
        anonymized_telemetry=False,
        allow_reset=True,
        chroma_product_telemetry_impl=(
            "app.db.vector_store.NoopProductTelemetryClient"
        ),
    )


@dataclass(frozen=True, slots=True)
class VectorMatch:
    """A single similarity-search hit."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """Thin, testable façade over a Chroma collection.

    Parameters
    ----------
    collection:
        Pre-built Chroma collection (injected in tests). When omitted the
        collection is created from settings via a persistent client.
    """

    def __init__(
        self,
        collection: Any | None = None,
        *,
        config: Settings | None = None,
        persist_directory: str | Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._config = config or get_settings()
        self._lock = threading.RLock()

        if collection is not None:
            self._collection = collection
            self._persist_directory: Path | None = None
        else:
            resolved_dir = Path(persist_directory or self._config.chroma_path)
            resolved_name = collection_name or self._config.chroma_collection
            self._persist_directory = resolved_dir
            self._collection = self._build_collection(resolved_dir, resolved_name)

    # ------------------------------------------------------------- internals
    @staticmethod
    def _build_collection(directory: Path, name: str) -> Any:  # noqa: ANN401
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise VectorStoreError(
                "chromadb is not installed. Install it or inject a collection.",
                code="chromadb_missing",
            ) from exc

        directory.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(directory),
            settings=_build_chroma_settings(),
        )
        # Only the HNSW index option is set: Chroma validates collection metadata
        # keys strictly, so arbitrary descriptive keys are avoided.
        collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB embedded collection '%s' ready at %s (%s vectors)",
            name,
            directory,
            collection.count(),
        )
        return collection

    @staticmethod
    def _to_score(distance: float | None) -> float:
        """Convert a cosine distance into a 0..1 similarity score."""
        if distance is None:
            return 0.0
        # Chroma cosine distance is in [0, 2]; similarity = 1 - distance.
        return round(max(min(1.0 - float(distance), 1.0), 0.0), 6)

    # ---------------------------------------------------------------- public
    @property
    def collection(self) -> Any:  # noqa: ANN401 - chroma collection duck type
        return self._collection

    @property
    def persist_directory(self) -> Path | None:
        return self._persist_directory

    def count(self) -> int:
        """Number of vectors currently stored."""
        try:
            return int(self._collection.count())
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(f"Unable to count vectors: {exc}") from exc

    def add(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        """Upsert vectors and return the number of items written."""
        if not ids:
            return 0
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise VectorStoreError("Chroma payload lists must have equal lengths.")

        # Chroma rejects ``None`` values inside metadata: normalise to sentinels.
        clean_metadatas = [
            {key: ("" if value is None else value) for key, value in meta.items()}
            for meta in metadatas
        ]
        try:
            with self._lock:
                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=clean_metadatas,
                )
        except Exception as exc:
            raise VectorStoreError(f"Failed to upsert vectors: {exc}") from exc
        return len(ids)

    def query(
        self,
        embedding: list[float],
        *,
        top_k: int = 4,
        where: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorMatch]:
        """Nearest-neighbour search filtered by a metadata ``where`` clause.

        Returns an empty list when nothing matches the filter: an empty
        knowledge base is a normal state and the RAG chain relies on ``[]`` to
        produce its "not enough information" answer.

        A genuine *failure* (bad filter, dimension mismatch, corrupt index) is
        deliberately **not** collapsed into that same empty list silently - that
        is how an ungrounded answer with no citations gets produced with nothing
        in the logs to explain it. Failures are logged loudly and then reported
        as empty so a single bad query cannot take down the chat endpoint.
        """
        if not embedding:
            return []

        # ``count()`` is separated from the search so an index-level failure is
        # reported as such rather than being mistaken for "no matches".
        try:
            with self._lock:
                available = int(self._collection.count())
        except Exception as exc:  # noqa: BLE001 - reported, then degraded
            logger.error("Vector store count failed: %s", exc)
            return []
        if available == 0:
            return []

        try:
            with self._lock:
                # Never ask for more neighbours than the collection holds: some
                # Chroma versions raise (and the RAG chain would then report "no
                # information") when ``n_results`` exceeds the index size.
                raw = self._collection.query(
                    query_embeddings=[embedding],
                    n_results=min(max(top_k, 1), available),
                    where=where or None,
                    include=["documents", "metadatas", "distances"],
                )
            matches = self._parse_query_result(raw, score_threshold=score_threshold)
        except Exception as exc:  # noqa: BLE001 - reported, then degraded
            # Without this the operator sees an answer with no citations and no
            # explanation. ``where`` is included because a malformed/over-narrow
            # metadata filter is the usual cause.
            logger.error(
                "Vector query failed (collection holds %s vectors, where=%s): %s",
                available,
                where,
                exc,
            )
            matches = []

        if not matches:
            # Chroma keeps two stores: the durable rows that ``count()`` reports,
            # and a per-process in-memory HNSW index that answers ``query()``. A
            # second process writing to the same persist directory (the Celery
            # worker, by default) is invisible to this index, so the index holds
            # 0 elements while rows exist. Chroma then answers *every* query with
            # zero hits and the whole knowledge base looks empty. Detecting it
            # here turns a silent "no sources" into an explicit, actionable error.
            matching = self._count_matching(where)
            if matching:
                logger.error(
                    "ANN index is out of sync: %s row(s) match %s but the vector "
                    "index returned nothing (collection holds %s). The index is "
                    "per-process, so vectors written by another process (e.g. the "
                    "Celery worker) cannot be read here. Re-index the affected "
                    "documents, or run ingestion in the API process "
                    "(CELERY_TASK_ALWAYS_EAGER=true) so one process owns the "
                    "collection.",
                    matching,
                    where,
                    available,
                )
        return matches

    def _count_matching(self, where: dict[str, Any] | None) -> int:
        """Count stored rows matching ``where`` (metadata store, not the index).

        ``get()`` is served from the durable metadata store rather than the
        in-memory ANN index, so it stays correct even when the index is stale -
        which is exactly what makes it usable as a drift probe. Bounded so a
        large collection cannot be materialised by a diagnostic.
        """
        try:
            with self._lock:
                raw = self._collection.get(where=where or None, limit=DRIFT_PROBE_LIMIT)
        except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
            logger.debug("Drift probe failed: %s", exc)
            return 0
        return len(raw.get("ids") or [])

    @staticmethod
    def _parse_query_result(
        raw: dict[str, Any], *, score_threshold: float | None
    ) -> list[VectorMatch]:
        """Turn a Chroma ``query()`` payload into :class:`VectorMatch` objects."""
        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        matches: list[VectorMatch] = []
        for index, vector_id in enumerate(ids):
            score = VectorStore._to_score(
                distances[index] if index < len(distances) else None
            )
            if score_threshold is not None and score < score_threshold:
                continue
            matches.append(
                VectorMatch(
                    id=vector_id,
                    text=documents[index] if index < len(documents) else "",
                    score=score,
                    metadata=dict(metadatas[index]) if index < len(metadatas) else {},
                )
            )
        return matches

    def delete(
        self, *, where: dict[str, Any] | None = None, ids: list[str] | None = None
    ) -> None:
        """Delete vectors by metadata filter and/or explicit ids.

        Deleting something that does not exist is a no-op, not an error: the
        ingestion pipeline always clears a document before writing it, so a
        first-time upload legitimately has nothing to remove.
        """
        if not where and not ids:
            return
        try:
            with self._lock:
                if ids:
                    self._collection.delete(ids=ids)
                if where:
                    self._collection.delete(where=where)
        except Exception as exc:  # noqa: BLE001 - nothing to delete is acceptable
            logger.debug("Vector delete matched nothing (%s); ignoring.", exc)

    def delete_document(self, document_id: str) -> None:
        """Remove every chunk belonging to a document."""
        self.delete(where={"document_id": document_id})

    def delete_user(self, user_id: str) -> None:
        """Remove every chunk owned by a user (account deletion)."""
        self.delete(where={"user_id": user_id})

    def reset(self) -> None:
        """Drop all vectors (used by tests and admin tooling).

        Paginated so a large collection is cleared without materialising every
        id at once.
        """
        try:
            with self._lock:
                while True:
                    batch = self._collection.get(limit=5_000)
                    ids = list(batch.get("ids") or [])
                    if not ids:
                        break
                    self._collection.delete(ids=ids)
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(f"Vector reset failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Process-wide singleton
# --------------------------------------------------------------------------- #
_store_lock = threading.Lock()
_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Return the lazily created process-wide vector store."""
    global _store  # noqa: PLW0603
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = VectorStore()
    return _store


def reset_vector_store() -> None:
    """Drop the singleton (tests / shutdown)."""
    global _store  # noqa: PLW0603
    _store = None
