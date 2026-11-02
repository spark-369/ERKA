"""Background task schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class TaskRead(ORMModel):
    """Durable task record returned by ``GET /tasks/{id}``."""

    id: str
    celery_task_id: str | None = None
    name: str
    queue: str
    status: str
    resource_type: str | None = None
    resource_id: str | None = None
    progress: int = Field(ge=0, le=100)
    error_message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime


class TaskSubmitResponse(BaseModel):
    """Acknowledgement of a task dispatch."""

    task_id: str = Field(description="Celery task id (poll ``/tasks/{id}``).")
    record_id: str = Field(description="Durable task record id.")
    status: str = "queued"
    resource_id: str | None = None
    detail: str = "Task accepted."
    extra: dict[str, Any] | None = None
