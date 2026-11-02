"""Background task ledger.

Celery stores results in Redis, but Redis is not a durable system of record for
an enterprise deployment. Every dispatched task is therefore also recorded here
so clients can poll ``/tasks/{id}`` and operators can audit failures.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

TASK_STATUSES = ("queued", "running", "succeeded", "failed", "revoked")


class BackgroundTask(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Durable record of a Celery task dispatch."""

    __tablename__ = "background_tasks"
    __table_args__ = (
        Index("ix_background_tasks_owner_created", "owner_id", "created_at"),
        {"comment": "Celery task audit trail"},
    )

    owner_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    celery_task_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    queue: Mapped[str] = mapped_column(String(64), default="default", nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="queued", nullable=False, index=True)
    resource_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    @property
    def is_terminal(self) -> bool:
        return self.status in ("succeeded", "failed", "revoked")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<BackgroundTask id={self.id} name={self.name} status={self.status}>"
