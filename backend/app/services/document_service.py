"""Document service: upload validation, storage, ingestion dispatch, deletion.

The service never touches Celery or Chroma directly for heavy work - it hands
the stored path to a task (see :mod:`app.workers.tasks`) and records the task in
the durable ledger.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    ConflictError,
    FileValidationError,
    NotFoundError,
    ValidationError,
)
from app.core.logging import get_logger
from app.models.document import Document
from app.models.task import BackgroundTask
from app.repositories.document import DocumentChunkRepository, DocumentRepository
from app.repositories.task import BackgroundTaskRepository
from app.schemas.document import DocumentUpdate

logger = get_logger(__name__)

UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB read buffer


async def read_upload(file: Any, *, max_bytes: int, chunk_size: int = UPLOAD_CHUNK_SIZE) -> bytes:
    """Read an ``UploadFile`` incrementally, refusing anything over ``max_bytes``.

    Reading with ``await file.read()`` buffers the entire request body in memory
    before any policy check runs, so the configured upload ceiling could only be
    enforced *after* the damage was done. Streaming in bounded chunks lets the
    size check reject an oversized upload as soon as the limit is crossed.
    """
    buffer = bytearray()
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise FileValidationError(
                f"File exceeds the {max_bytes // (1024 * 1024)} MB limit.",
                code="file_too_large",
                details={"max_bytes": max_bytes},
            )
    return bytes(buffer)


class DocumentService:
    """File lifecycle: validate -> persist -> enqueue -> report -> delete."""

    def __init__(self, session: Session, config: Settings | None = None) -> None:
        self._session = session
        self._config = config or get_settings()
        self._documents = DocumentRepository(session)
        self._chunks = DocumentChunkRepository(session)
        self._tasks = BackgroundTaskRepository(session)

    # -------------------------------------------------------------- validation
    @property
    def max_upload_bytes(self) -> int:
        """Configured upload ceiling in bytes (single source of truth)."""
        return self._config.max_upload_size_bytes

    def validate_upload(self, *, filename: str, size_bytes: int, content_type: str | None) -> str:
        """Enforce extension/size policy; returns the normalised extension."""
        if not filename or "." not in filename:
            raise FileValidationError(
                "Uploaded files must have a file extension.", code="missing_extension"
            )
        extension = Path(filename).suffix.lower()
        if extension not in self._config.allowed_extensions:
            raise FileValidationError(
                f"Unsupported file type '{extension}'. Allowed: "
                f"{', '.join(self._config.allowed_extensions)}.",
                code="unsupported_extension",
                details={"allowed": self._config.allowed_extensions},
            )
        if size_bytes <= 0:
            raise FileValidationError("The uploaded file is empty.", code="empty_file")
        if size_bytes > self._config.max_upload_size_bytes:
            raise FileValidationError(
                f"File exceeds the {self._config.max_upload_size_mb} MB limit.",
                code="file_too_large",
                details={"max_bytes": self._config.max_upload_size_bytes},
            )
        return extension

    @staticmethod
    def compute_checksum(data: bytes) -> str:
        """SHA-256 of the raw payload (deduplication + integrity)."""
        return hashlib.sha256(data).hexdigest()

    # ----------------------------------------------------------------- storage
    def _store_file(self, data: bytes, owner_id: str, filename: str) -> tuple[str, Path]:
        """Write the upload to disk under a per-user directory."""
        suffix = Path(filename).suffix.lower()
        stored_name = f"{uuid.uuid4().hex}{suffix}"
        directory = self._config.upload_path / owner_id
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / stored_name
        target.write_bytes(data)
        logger.info("Stored upload %s (%s bytes) at %s", filename, len(data), target)
        return stored_name, target

    def create_document(
        self,
        *,
        owner_id: str,
        filename: str,
        data: bytes,
        content_type: str | None = None,
        collection: str | None = None,
        tags: str | None = None,
        deduplicate: bool = True,
    ) -> Document:
        """Validate, store and register an uploaded document."""
        extension = self.validate_upload(
            filename=filename, size_bytes=len(data), content_type=content_type
        )
        checksum = self.compute_checksum(data)

        if deduplicate:
            existing = self._documents.get_by_checksum(owner_id, checksum)
            if existing is not None and existing.status != "failed":
                raise ConflictError(
                    "This exact file has already been uploaded.",
                    code="duplicate_document",
                    details={"document_id": existing.id, "status": existing.status},
                )

        stored_name, path = self._store_file(data, owner_id, filename)
        document = self._documents.create(
            owner_id=owner_id,
            filename=stored_name,
            original_filename=filename,
            content_type=content_type or "application/octet-stream",
            extension=extension,
            size_bytes=len(data),
            checksum=checksum,
            storage_path=str(path),
            collection=collection or self._config.chroma_collection,
            tags=tags,
        )
        self._session.commit()
        self._session.refresh(document)
        return document

    # ------------------------------------------------------------------ reads
    def get_document(self, document_id: str, owner_id: str) -> Document:
        document = self._documents.get_for_owner(document_id, owner_id)
        if document is None:
            raise NotFoundError("Document not found.", code="document_not_found")
        return document

    def list_documents(
        self,
        owner_id: str,
        *,
        page: int = 1,
        page_size: int = 20,
        status: str | None = None,
        search: str | None = None,
        collection: str | None = None,
        extension: str | None = None,
    ) -> tuple[list[Document], int]:
        """Return ``(documents, total)`` for the paginated list endpoint."""
        offset = (page - 1) * page_size
        documents = self._documents.list_for_owner(
            owner_id,
            offset=offset,
            limit=page_size,
            status=status,
            search=search,
            collection=collection,
            extension=extension,
        )
        total = self._documents.count_for_owner(
            owner_id,
            status=status,
            search=search,
            collection=collection,
            extension=extension,
        )
        return documents, total

    def list_chunks(self, document_id: str, owner_id: str, *, limit: int = 100):
        """Return a document's chunks (ownership enforced first)."""
        self.get_document(document_id, owner_id)
        return self._chunks.list_for_document(document_id, limit=limit)

    # ----------------------------------------------------------------- updates
    def update_document(
        self, document_id: str, owner_id: str, payload: DocumentUpdate
    ) -> Document:
        document = self.get_document(document_id, owner_id)
        if payload.original_filename is not None:
            document.original_filename = payload.original_filename
        if payload.tags is not None:
            document.tags = payload.tags
        if payload.collection is not None:
            document.collection = payload.collection
        self._session.commit()
        self._session.refresh(document)
        return document

    # ------------------------------------------------------------------ ingest
    def mark_processing(self, document: Document) -> Document:
        self._documents.mark_processing(document)
        self._session.commit()
        return document

    def mark_ready(
        self, document: Document, *, chunk_count: int, token_count: int
    ) -> Document:
        self._documents.mark_ready(
            document, chunk_count=chunk_count, token_count=token_count
        )
        self._session.commit()
        return document

    def mark_failed(self, document: Document, error: str) -> Document:
        self._documents.mark_failed(document, error)
        self._session.commit()
        return document

    def record_chunks(self, document_id: str, chunks: list[dict]) -> None:
        """Replace the chunk ledger rows for a document."""
        self._chunks.replace_for_document(document_id, chunks)
        self._session.commit()

    def register_task(
        self,
        *,
        name: str,
        owner_id: str,
        celery_task_id: str | None,
        document_id: str,
        queue: str = "ingestion",
    ) -> BackgroundTask:
        """Create the durable audit row for a dispatched ingestion task."""
        task = self._tasks.create(
            name=name,
            owner_id=owner_id,
            celery_task_id=celery_task_id,
            queue=queue,
            resource_type="document",
            resource_id=document_id,
        )
        self._session.commit()
        self._session.refresh(task)
        return task

    def get_latest_task(self, document_id: str) -> BackgroundTask | None:
        tasks = self._tasks.list_by_resource("document", document_id)
        return tasks[0] if tasks else None

    # ----------------------------------------------------------------- deletes
    def delete_document(
        self, document_id: str, owner_id: str, *, remove_vectors: bool = True
    ) -> Document:
        """Delete a document row, its chunks, its file and its vectors."""
        document = self.get_document(document_id, owner_id)
        storage_path = Path(document.storage_path) if document.storage_path else None

        if remove_vectors:
            # Imported here so the service stays importable without chromadb.
            from app.rag.pipeline import get_ingestion_pipeline

            try:
                get_ingestion_pipeline().delete_document(document_id)
            except Exception as exc:  # noqa: BLE001 - DB row still removed
                logger.warning("Vector cleanup failed for %s: %s", document_id, exc)

        self._documents.delete(document)
        self._session.commit()

        if storage_path and storage_path.exists():
            try:
                storage_path.unlink()
            except OSError as exc:  # pragma: no cover - filesystem edge case
                logger.warning("Could not delete stored file %s: %s", storage_path, exc)

        logger.info("Deleted document %s (owner=%s)", document_id, owner_id)
        return document

    def stats(self, owner_id: str) -> dict[str, int]:
        """Aggregate counts used by the dashboard."""
        return {
            "total": self._documents.count_for_owner(owner_id),
            "ready": self._documents.count_for_owner(owner_id, status="ready"),
            "processing": self._documents.count_for_owner(owner_id, status="processing"),
            "pending": self._documents.count_for_owner(owner_id, status="pending"),
            "failed": self._documents.count_for_owner(owner_id, status="failed"),
        }

    def ensure_storage_path(self, document: Document) -> Path:
        """Return the on-disk path, raising if the file disappeared."""
        if not document.storage_path:
            raise ValidationError(
                "This document has no stored file.", code="missing_storage_path"
            )
        path = Path(document.storage_path)
        if not path.exists():
            raise NotFoundError(
                "The stored file for this document is missing.", code="file_missing"
            )
        return path
