"""Ingestion pipeline: load -> chunk -> embed -> persist (Chroma + Postgres).

This is the LangChain orchestration used by the Celery worker for heavy parsing
and by the synchronous ``reindex`` endpoint for small files.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.core.config import Settings, get_settings
from app.core.exceptions import DocumentProcessingError
from app.core.logging import get_logger
from app.db.vector_store import VectorStore, get_vector_store
from app.rag.chunking import TextChunk, TextChunker
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider
from app.rag.loaders import DocumentLoader, LoadedDocument

logger = get_logger(__name__)

ProgressCallback = Callable[[int, str], None]


@dataclass(slots=True)
class IngestionResult:
    """Summary of one document ingestion run."""

    document_id: str
    chunk_count: int
    token_count: int
    char_count: int
    vector_ids: list[str] = field(default_factory=list)
    #: Chunk ledger rows ready to be written to PostgreSQL by the caller.
    chunk_records: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class IngestionPipeline:
    """Turn a stored file into retrievable vectors + a chunk ledger."""

    def __init__(
        self,
        *,
        loader: DocumentLoader | None = None,
        chunker: TextChunker | None = None,
        embeddings: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
        config: Settings | None = None,
    ) -> None:
        self._config = config or get_settings()
        self._loader = loader
        self._chunker = chunker
        self._embeddings = embeddings
        self._store = store

    # ----------------------------------------------------------- lazy wiring
    @property
    def loader(self) -> DocumentLoader:
        if self._loader is None:
            self._loader = DocumentLoader(config=self._config)
        return self._loader

    @property
    def chunker(self) -> TextChunker:
        if self._chunker is None:
            self._chunker = TextChunker(config=self._config)
        return self._chunker

    @property
    def embeddings(self) -> EmbeddingProvider:
        if self._embeddings is None:
            self._embeddings = get_embedding_provider()
        return self._embeddings

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = get_vector_store()
        return self._store

    # ---------------------------------------------------------------- public
    def ingest_bytes(
        self,
        data: bytes,
        *,
        document_id: str,
        user_id: str,
        filename: str,
        collection: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        progress: ProgressCallback | None = None,
    ) -> IngestionResult:
        """Full pipeline for an in-memory file."""
        self._report(progress, 5, "Parsing document")
        loaded = self.loader.load_bytes(data, filename)
        return self._ingest_loaded(
            loaded,
            document_id=document_id,
            user_id=user_id,
            collection=collection or self._config.chroma_collection,
            extra_metadata=extra_metadata,
            progress=progress,
        )

    def ingest_path(
        self,
        path: str | Path,
        *,
        document_id: str,
        user_id: str,
        filename: str | None = None,
        collection: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        progress: ProgressCallback | None = None,
    ) -> IngestionResult:
        """Full pipeline for a file already stored on disk (Celery worker path)."""
        self._report(progress, 5, "Reading stored file")
        loaded = self.loader.load_path(path)
        if filename:
            loaded.source = filename
        return self._ingest_loaded(
            loaded,
            document_id=document_id,
            user_id=user_id,
            collection=collection or self._config.chroma_collection,
            extra_metadata=extra_metadata,
            progress=progress,
        )

    def delete_document(self, document_id: str) -> None:
        """Remove a document's vectors from Chroma."""
        self.store.delete_document(document_id)

    # ------------------------------------------------------------- internals
    def _ingest_loaded(
        self,
        loaded: LoadedDocument,
        *,
        document_id: str,
        user_id: str,
        collection: str,
        extra_metadata: dict[str, Any] | None,
        progress: ProgressCallback | None,
    ) -> IngestionResult:
        self._report(progress, 20, "Splitting text into chunks")
        base_metadata: dict[str, Any] = {
            "document_id": document_id,
            "user_id": user_id,
            "document_name": loaded.source,
            "collection": collection,
            "extension": loaded.extension,
            "page_count": loaded.page_count,
        }
        base_metadata.update(extra_metadata or {})

        chunks = self.chunker.split_text(loaded.text, metadata=base_metadata)
        if not chunks:
            raise DocumentProcessingError(
                "Chunking produced no content for this document.", code="empty_chunks"
            )

        self._report(progress, 45, f"Embedding {len(chunks)} chunks")
        vectors = self._embed_in_batches(chunks)

        self._report(progress, 80, "Writing vectors to ChromaDB")
        vector_ids = [f"{document_id}:{chunk.chunk_index}" for chunk in chunks]
        # Idempotency: drop any previous version of this document first.
        self.store.delete_document(document_id)
        self.store.add(
            ids=vector_ids,
            embeddings=vectors,
            documents=[chunk.content for chunk in chunks],
            metadatas=[
                {
                    **chunk.metadata,
                    "char_count": chunk.char_count,
                    "token_count": chunk.token_count,
                }
                for chunk in chunks
            ],
        )

        self._report(progress, 95, "Finalising")
        return IngestionResult(
            document_id=document_id,
            chunk_count=len(chunks),
            token_count=sum(chunk.token_count for chunk in chunks),
            char_count=sum(chunk.char_count for chunk in chunks),
            vector_ids=vector_ids,
            chunk_records=[
                {
                    "chunk_index": chunk.chunk_index,
                    "vector_id": vector_id,
                    "content": chunk.content,
                    "char_count": chunk.char_count,
                    "token_count": chunk.token_count,
                }
                for chunk, vector_id in zip(chunks, vector_ids, strict=False)
            ],
            metadata={
                "source": loaded.source,
                "extension": loaded.extension,
                "page_count": loaded.page_count,
                "collection": collection,
                **{k: v for k, v in loaded.metadata.items() if k != "pdf_metadata"},
            },
        )

    def _embed_in_batches(self, chunks: list[TextChunk]) -> list[list[float]]:
        batch_size = max(self._config.embedding_batch_size, 1)
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors.extend(self.embeddings.embed_documents([c.content for c in batch]))
        if len(vectors) != len(chunks):
            raise DocumentProcessingError(
                "Embedding count does not match chunk count.", code="embedding_mismatch"
            )
        return vectors

    @staticmethod
    def _report(progress: ProgressCallback | None, value: int, message: str) -> None:
        logger.debug("Ingestion progress %s%%: %s", value, message)
        if progress is not None:
            progress(value, message)


_pipeline_lock = threading.Lock()
_pipeline: IngestionPipeline | None = None


def get_ingestion_pipeline() -> IngestionPipeline:
    """Return the lazily created process-wide ingestion pipeline."""
    global _pipeline  # noqa: PLW0603
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = IngestionPipeline()
    return _pipeline


def reset_ingestion_pipeline(pipeline: IngestionPipeline | None = None) -> None:
    """Replace (or clear) the singleton - used by tests."""
    global _pipeline  # noqa: PLW0603
    _pipeline = pipeline
