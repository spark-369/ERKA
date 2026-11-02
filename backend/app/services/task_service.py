"""Task service: durable records for background (Celery) work.

Celery results live in Redis, which is a cache and not a system of record. This
service reconciles the two: the database row is authoritative for auditing, and
Celery state is used to refresh in-flight progress.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.models.task import BackgroundTask
from app.repositories.task import BackgroundTaskRepository

logger = get_logger(__name__)


class TaskService:
    """Read/refresh helpers around :class:`BackgroundTask`."""

    def __init__(self, session: Session, config: Settings | None = None) -> None:
        self._session = session
        self._config = config or get_settings()
        self._tasks = BackgroundTaskRepository(session)

    # ------------------------------------------------------------------ reads
    def get_task(self, identifier: str, *, owner_id: str | None = None) -> BackgroundTask:
        """Fetch a task by record id or Celery id, refreshing from Celery."""
        task = self._tasks.resolve(identifier)
        if task is None:
            raise NotFoundError("Task not found.", code="task_not_found")
        if owner_id is not None and task.owner_id not in (None, owner_id):
            raise NotFoundError("Task not found.", code="task_not_found")

        if not task.is_terminal:
            self._sync_from_celery(task)
        return task

    def list_tasks(
        self, owner_id: str, *, page: int = 1, page_size: int = 20
    ) -> tuple[list[BackgroundTask], int]:
        offset = (page - 1) * page_size
        tasks = self._tasks.list_for_owner(owner_id, offset=offset, limit=page_size)
        total = self._tasks.count(
            select(BackgroundTask).where(BackgroundTask.owner_id == owner_id)
        )
        return tasks, total

    def list_for_document(self, document_id: str) -> list[BackgroundTask]:
        return self._tasks.list_by_resource("document", document_id)

    # --------------------------------------------------------------- mutation
    def update_progress(self, task_id: str, progress: int, *, status: str | None = None):
        task = self._tasks.get(task_id)
        if task is None:
            raise NotFoundError("Task not found.", code="task_not_found")
        self._tasks.update_progress(task, progress)
        if status:
            task.status = status
        self._session.commit()
        return task

    def mark_succeeded(self, task_id: str, *, result: Any = None):
        task = self._tasks.get(task_id)
        if task is None:
            return None
        self._tasks.mark_succeeded(task, result=result)
        self._session.commit()
        return task

    def mark_failed(self, task_id: str, error: str):
        task = self._tasks.get(task_id)
        if task is None:
            return None
        self._tasks.mark_failed(task, error)
        self._session.commit()
        return task

    # -------------------------------------------------------------- internals
    def _sync_from_celery(self, task: BackgroundTask) -> None:
        """Pull the latest Celery state into the durable row."""
        if not task.celery_task_id:
            return
        try:
            from app.workers.celery_app import celery_app

            state = celery_app.AsyncResult(task.celery_task_id)
            status = (state.state or "").upper()
        except Exception as exc:  # pragma: no cover - broker unavailable
            logger.debug("Could not query Celery for task %s: %s", task.id, exc)
            return

        if status == "STARTED" and task.status == "queued":
            self._tasks.mark_running(task)
        elif status == "PROGRESS":
            meta = state.info if isinstance(state.info, dict) else {}
            self._tasks.update_progress(task, int(meta.get("progress", task.progress)))
        elif status == "SUCCESS":
            result = state.result if isinstance(state.result, dict) else None
            self._tasks.mark_succeeded(task, result=result)
        elif status == "FAILURE":
            self._tasks.mark_failed(task, str(state.result or "Task failed."))

        if self._session.is_modified(task):
            self._session.commit()
