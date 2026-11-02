"""Background task endpoints: status polling and listing."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, pagination_params
from app.db.session import get_db
from app.models.user import User
from app.schemas.common import Page, PaginationParams
from app.schemas.task import TaskRead, TaskSubmitResponse
from app.services.task_service import TaskService

router = APIRouter()


@router.get(
    "",
    response_model=Page[TaskRead],
    summary="List background tasks for the current user",
)
def list_tasks(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
    pagination: Annotated[PaginationParams, Depends(pagination_params)],
) -> Page[TaskRead]:
    """Paginated audit trail of dispatched Celery tasks."""
    tasks, total = TaskService(session).list_tasks(
        current_user.id, page=pagination.page, page_size=pagination.page_size
    )
    return Page[TaskRead](
        items=[TaskRead.model_validate(task) for task in tasks],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
    )


@router.get(
    "/{task_id}",
    response_model=TaskRead,
    summary="Fetch a task by record id or Celery id",
)
def get_task(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> TaskRead:
    """Return the durable task row, refreshed from Celery when still running."""
    task = TaskService(session).get_task(task_id, owner_id=current_user.id)
    return TaskRead.model_validate(task)


@router.post(
    "/maintenance/health",
    response_model=TaskSubmitResponse,
    summary="Dispatch a worker health probe",
)
def dispatch_health_check(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> TaskSubmitResponse:
    """Queue a trivial task to confirm workers are consuming the queue."""
    from app.workers.tasks.maintenance import health_check

    result = health_check.apply_async()
    return TaskSubmitResponse(
        task_id=result.id,
        record_id=result.id,
        status="queued",
        detail="Worker health probe dispatched.",
    )
