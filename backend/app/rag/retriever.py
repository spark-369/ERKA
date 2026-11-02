"""Retrieval: embed the query, search Chroma, filter by score.

The retriever is the read half of the RAG chain. It enforces per-user isolation
by translating the caller's document scope into a Chroma ``where`` clause, so a
user can never retrieve another tenant's chunks.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.vector_store import VectorMatch, VectorStore, get_vector_store
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider

logger = get_logger(__name__)


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk returned by the retriever, ready for prompt building."""

    chunk_id: str
    document_id: str
    document_name: str
    text: str
    score: float
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def snippet(self, limit: int = 300) -> str:
        """Shortened text for citation cards."""
        clean = " ".join(self.text.split())
        if len(clean) <= limit:
            return clean
        return clean[:limit].rsplit(" ", 1)[0] + "..."


class Retriever:
    """Similarity search over the embedded ChromaDB collection."""

    def __init__(
        self,
        *,
        store: VectorStore | None = None,
        embeddings: EmbeddingProvider | None = None,
        config: Settings | None = None,
    ) -> None:
        self._config = config or get_settings()
        self._store = store
        self._embeddings = embeddings

    # ----------------------------------------------------------- lazy wiring
    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = get_vector_store()
        return self._store

    @property
    def embeddings(self) -> EmbeddingProvider:
        if self._embeddings is None:
            self._embeddings = get_embedding_provider()
        return self._embeddings

    # --------------------------------------------------------------- helpers
    @staticmethod
    def build_where(
        *,
        user_id: str,
        document_ids: list[str] | None = None,
        collection: str | None = None,
    ) -> dict[str, Any]:
        """Build a Chroma metadata filter enforcing tenant isolation."""
        clauses: list[dict[str, Any]] = [{"user_id": {"$eq": user_id}}]
        if document_ids:
            clauses.append({"document_id": {"$in": list(document_ids)}})
        if collection:
            clauses.append({"collection": {"$eq": collection}})
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _to_chunk(self, match: VectorMatch) -> RetrievedChunk:
        metadata = match.metadata or {}
        return RetrievedChunk(
            chunk_id=match.id,
            document_id=str(metadata.get("document_id", "")),
            document_name=str(
                metadata.get("document_name")
                or metadata.get("original_filename")
                or "document"
            ),
            text=match.text,
            score=match.score,
            chunk_index=int(metadata.get("chunk_index", 0) or 0),
            metadata=metadata,
        )

    # ---------------------------------------------------------------- public
    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        document_ids: list[str] | None = None,
        collection: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks for ``query``."""
        if not query or not query.strip():
            return []

        limit = top_k or self._config.retrieval_top_k
        threshold = (
            self._config.retrieval_score_threshold
            if score_threshold is None
            else score_threshold
        )
        embedding = self.embeddings.embed_query(query.strip())
        where = self.build_where(
            user_id=user_id, document_ids=document_ids, collection=collection
        )

        # Fetch a wider candidate pool than we return, then threshold + trim.
        # This stops a single document's high-scoring chunks from crowding every
        # other document out of the context.
        fetch_k = max(limit, self._config.retrieval_fetch_k)
        matches = self.store.query(
            embedding,
            top_k=fetch_k,
            where=where,
            score_threshold=threshold,
        )
        chunks = [self._to_chunk(match) for match in matches]

        if not chunks:
            # Distinguish "nothing in the knowledge base" from "the filter or
            # index returned nothing while vectors exist". The former is normal;
            # the latter is the failure mode that produces an ungrounded answer
            # with no citations and no explanation, so it is logged loudly.
            try:
                stored = self.store.count()
            except Exception:  # noqa: BLE001 - diagnostics must never raise
                stored = -1
            if stored != 0:
                logger.warning(
                    "Retrieval returned no chunks for user=%s although the vector "
                    "store holds %s vector(s) (where=%s, threshold=%.2f). "
                    "Check the document scope, the score threshold, and whether "
                    "the API process can see vectors written by the worker.",
                    user_id,
                    stored,
                    where,
                    threshold,
                )

        logger.debug(
            "Retrieved %s/%s chunks for user=%s (threshold=%.2f)",
            len(chunks),
            limit,
            user_id,
            threshold,
        )
        return chunks[:limit]

    def retrieve_as_langchain_documents(
        self,
        query: str,
        *,
        user_id: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[Any]:
        """Return results as LangChain ``Document`` objects.

        Useful when plugging the retriever into a LangChain chain that expects
        ``Document`` instances (e.g. ``create_stuff_documents_chain``).
        """
        try:
            from langchain_core.documents import Document
        except ImportError:  # pragma: no cover - langchain optional at runtime
            return []

        return [
            Document(
                page_content=chunk.text,
                metadata={
                    **chunk.metadata,
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "score": chunk.score,
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk in self.retrieve(
                query, user_id=user_id, document_ids=document_ids, top_k=top_k
            )
        ]


_retriever_lock = threading.Lock()
_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Return the lazily created process-wide retriever."""
    global _retriever  # noqa: PLW0603
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = Retriever()
    return _retriever


def reset_retriever(retriever: Retriever | None = None) -> None:
    """Replace (or clear) the singleton - used by tests."""
    global _retriever  # noqa: PLW0603
    _retriever = retriever
