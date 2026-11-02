"""Background task repository (durable Celery audit trail)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.models.task import BackgroundTask
from app.repositories.base import BaseRepository


class BackgroundTaskRepository(BaseRepository[BackgroundTask]):
    """Queries scoped to the ``background_tasks`` table."""

    model = BackgroundTask

    def create(
        self,
        *,
        name: str,
        owner_id: str | None = None,
        celery_task_id: str | None = None,
        queue: str = "default",
        resource_type: str | None = None,
        resource_id: str | None = None,
    ) -> BackgroundTask:
        task = BackgroundTask(
            name=name,
            owner_id=owner_id,
            celery_task_id=celery_task_id,
            queue=queue,
            resource_type=resource_type,
            resource_id=resource_id,
            status="queued",
        )
        return self.add(task)

    def get_by_celery_id(self, celery_task_id: str) -> BackgroundTask | None:
        statement = select(BackgroundTask).where(
            BackgroundTask.celery_task_id == celery_task_id
        )
        return self.first(statement)

    def resolve(self, identifier: str) -> BackgroundTask | None:
        """Look a task up by either its own id or the Celery task id."""
        return self.get(identifier) or self.get_by_celery_id(identifier)

    def list_for_owner(
        self, owner_id: str, *, offset: int = 0, limit: int = 20
    ) -> list[BackgroundTask]:
        statement = (
            select(BackgroundTask)
            .where(BackgroundTask.owner_id == owner_id)
            .order_by(BackgroundTask.created_at.desc())
        )
        return self.list(statement, offset=offset, limit=limit)

    def list_by_resource(
        self, resource_type: str, resource_id: str
    ) -> list[BackgroundTask]:
        statement = (
            select(BackgroundTask)
            .where(
                BackgroundTask.resource_type == resource_type,
                BackgroundTask.resource_id == resource_id,
            )
            .order_by(BackgroundTask.created_at.desc())
        )
        return self.list(statement, limit=50)

    # ------------------------------------------------------ state transitions
    def mark_running(self, task: BackgroundTask, *, progress: int = 0) -> BackgroundTask:
        task.status = "running"
        task.progress = progress
        task.started_at = datetime.now(timezone.utc)
        self.session.flush()
        return task

    def update_progress(self, task: BackgroundTask, progress: int) -> BackgroundTask:
        task.progress = max(0, min(int(progress), 100))
        self.session.flush()
        return task

    def mark_succeeded(
        self, task: BackgroundTask, *, result: Any = None, progress: int = 100
    ) -> BackgroundTask:
        task.status = "succeeded"
        task.progress = progress
        task.result_json = json.dumps(result, default=str) if result is not None else None
        task.error_message = None
        task.finished_at = datetime.now(timezone.utc)
        self.session.flush()
        return task

    def mark_failed(self, task: BackgroundTask, error: str) -> BackgroundTask:
        task.status = "failed"
        task.error_message = error[:2000]
        task.finished_at = datetime.now(timezone.utc)
        self.session.flush()
        return task
