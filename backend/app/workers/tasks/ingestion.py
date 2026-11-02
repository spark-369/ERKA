"""Ingestion tasks: heavy document parsing, chunking, embedding and indexing.

These tasks run inside the Celery worker (or inline when
``CELERY_TASK_ALWAYS_EAGER=true``). Each task owns its own DB session and keeps
the ``Document`` and ``BackgroundTask`` rows in sync as it progresses.
"""

from __future__ import annotations

from typing import Any

from celery import shared_task

from app.core.logging import get_logger
from app.db.session import session_scope
from app.rag.pipeline import get_ingestion_pipeline
from app.repositories.document import DocumentChunkRepository, DocumentRepository
from app.repositories.task import BackgroundTaskRepository

logger = get_logger(__name__)


@shared_task(
    bind=True,
    name="app.workers.tasks.ingestion.ingest_document",
    max_retries=3,
    default_retry_delay=15,
    acks_late=True,
)
def ingest_document(
    self, document_id: str, task_record_id: str | None = None
) -> dict[str, Any]:
    """Parse, chunk, embed and index one stored document.

    Parameters
    ----------
    document_id:
        Primary key of the ``Document`` row to process.
    task_record_id:
        Primary key of the durable ``BackgroundTask`` row to keep updated.
    """
    logger.info("Ingestion task started for document %s", document_id)

    # ------------------------------------------------------- load + mark state
    with session_scope() as session:
        documents = DocumentRepository(session)
        tasks = BackgroundTaskRepository(session)

        # The task record is resolved first so *every* failure path below can be
        # recorded on it. Leaving it untouched would strand the row in "queued"
        # and the document in "pending", with nothing to explain why.
        task_record = tasks.get(task_record_id) if task_record_id else None

        document = documents.get(document_id)
        if document is None:
            message = (
                f"Document {document_id} no longer exists; nothing to ingest."
            )
            logger.warning(message)
            if task_record is not None:
                tasks.mark_failed(task_record, message)
            return {"document_id": document_id, "status": "missing"}

        if task_record is not None:
            tasks.mark_running(task_record, progress=5)

        documents.mark_processing(document)
        storage_path = document.storage_path
        original_filename = document.original_filename
        owner_id = document.owner_id
        collection = document.collection

    def progress(value: int, _message: str) -> None:
        """Persist coarse progress so the UI can poll it."""
        if not task_record_id:
            return
        with session_scope() as session:
            tasks = BackgroundTaskRepository(session)
            task_record = tasks.get(task_record_id)
            if task_record is not None:
                tasks.update_progress(task_record, value)

    # ------------------------------------------------------------- run pipeline
    try:
        pipeline = get_ingestion_pipeline()
        result = pipeline.ingest_path(
            storage_path,
            document_id=document_id,
            user_id=owner_id,
            filename=original_filename,
            collection=collection,
            progress=progress,
        )
    except Exception as exc:  # noqa: BLE001 - retry or fail permanently
        logger.exception("Ingestion failed for document %s", document_id)
        with session_scope() as session:
            document = DocumentRepository(session).get(document_id)
            if document is not None:
                DocumentRepository(session).mark_failed(document, str(exc))
            if task_record_id:
                task_record = BackgroundTaskRepository(session).get(task_record_id)
                if task_record is not None:
                    BackgroundTaskRepository(session).mark_failed(task_record, str(exc))

        # Retry transient failures (model download hiccups, IO contention).
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        raise

    # ------------------------------------------------------------ persist state
    with session_scope() as session:
        documents = DocumentRepository(session)
        chunk_repo = DocumentChunkRepository(session)

        document = documents.get(document_id)
        if document is not None:
            chunk_repo.replace_for_document(document_id, result.chunk_records)
            documents.mark_ready(
                document,
                chunk_count=result.chunk_count,
                token_count=result.token_count,
            )

        if task_record_id:
            tasks = BackgroundTaskRepository(session)
            task_record = tasks.get(task_record_id)
            if task_record is not None:
                tasks.mark_succeeded(
                    task_record,
                    result={
                        "document_id": document_id,
                        "chunk_count": result.chunk_count,
                        "token_count": result.token_count,
                    },
                )

    logger.info(
        "Ingestion finished for document %s (%s chunks)", document_id, result.chunk_count
    )
    return {
        "document_id": document_id,
        "status": "ready",
        "chunk_count": result.chunk_count,
        "token_count": result.token_count,
        "vector_ids": result.vector_ids[:5],
    }
