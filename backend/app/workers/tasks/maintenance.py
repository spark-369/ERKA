"""Maintenance tasks: reindex, cleanup and health probes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from celery import shared_task
from sqlalchemy import select

from app.core.logging import get_logger
from app.db.session import session_scope
from app.models.document import Document
from app.repositories.task import BackgroundTaskRepository

logger = get_logger(__name__)


@shared_task(name="app.workers.tasks.maintenance.reindex_document")
def reindex_document(document_id: str, task_record_id: str | None = None) -> dict[str, Any]:
    """Re-run the ingestion pipeline for an existing document."""
    from app.workers.tasks.ingestion import ingest_document

    logger.info("Reindex requested for document %s", document_id)
    with session_scope() as session:
        document = session.get(Document, document_id)
        if document is None:
            return {"document_id": document_id, "status": "missing"}
        document.status = "pending"
        document.error_message = None

    return ingest_document.apply(args=(document_id, task_record_id)).get()


@shared_task(name="app.workers.tasks.maintenance.cleanup_stale_tasks")
def cleanup_stale_tasks(max_age_hours: int = 24) -> dict[str, Any]:
    """Fail task rows that have been queued far longer than expected."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    failed = 0

    with session_scope() as session:
        tasks = BackgroundTaskRepository(session)
        statement = select(tasks.model).where(
            tasks.model.status.in_(["queued", "running"]),
            tasks.model.created_at < cutoff,
        )
        for task in session.execute(statement).scalars().all():
            tasks.mark_failed(task, "Marked stale by the maintenance sweep.")
            failed += 1

    logger.info("Stale task sweep finished (%s rows marked failed)", failed)
    return {"marked_failed": failed, "cutoff": cutoff.isoformat()}


@shared_task(name="app.workers.tasks.maintenance.health_check")
def health_check() -> dict[str, Any]:
    """Trivial round-trip task used to verify the worker is alive."""
    from app.llm.factory import get_llm_provider
    from app.rag.embeddings import get_embedding_provider

    return {
        "status": "ok",
        "llm": get_llm_provider().health(),
        "embeddings": get_embedding_provider().health(),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
