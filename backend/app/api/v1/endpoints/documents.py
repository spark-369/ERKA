"""Document endpoints: upload, listing, status, chunks, reindex, delete."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, pagination_params
from app.api.rate_limit_dep import rate_limit_upload
from app.db.session import get_db
from app.models.user import User
from app.schemas.common import MessageResponse, Page, PaginationParams
from app.schemas.document import (
    DocumentChunkRead,
    DocumentRead,
    DocumentStatusRead,
    DocumentUpdate,
    DocumentUploadResponse,
)
from app.schemas.task import TaskSubmitResponse
from app.services.document_service import DocumentService, read_upload

router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for ingestion",
    dependencies=[Depends(rate_limit_upload)],
)
async def upload_document(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
    file: Annotated[UploadFile, File(description="PDF, TXT, MD, CSV, DOCX, JSON or HTML")],
    collection: Annotated[str | None, Form()] = None,
    tags: Annotated[str | None, Form()] = None,
    process_now: Annotated[bool, Form()] = True,
) -> DocumentUploadResponse:
    """Accept a file, store it and dispatch asynchronous ingestion.

    The heavy work (parsing, chunking, embedding) happens in a Celery worker;
    the response returns immediately with the task id to poll.
    """
    service = DocumentService(session)
    # Streamed in bounded chunks so the size ceiling is enforced *while* reading
    # rather than after the whole body has already been buffered in memory.
    data = await read_upload(file, max_bytes=service.max_upload_bytes)
    document = service.create_document(
        owner_id=current_user.id,
        filename=file.filename or "upload",
        data=data,
        content_type=file.content_type,
        collection=collection,
        tags=tags,
    )

    task_id: str | None = None
    if process_now:
        # Imported lazily so the API can boot without a broker configured.
        from app.workers.tasks.ingestion import ingest_document

        # The durable record is created first so the task can write progress to it.
        record = service.register_task(
            name="ingest_document",
            owner_id=current_user.id,
            celery_task_id=None,
            document_id=document.id,
        )
        async_result = ingest_document.apply_async(args=(document.id, record.id))
        task_id = async_result.id
        record.celery_task_id = task_id
        session.commit()

    return DocumentUploadResponse(
        document=DocumentRead.model_validate(document),
        task_id=task_id,
        message=(
            "Document accepted for processing."
            if process_now
            else "Document stored. Ingestion was skipped (process_now=false)."
        ),
    )


@router.get(
    "",
    response_model=Page[DocumentRead],
    summary="List documents owned by the current user",
)
def list_documents(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
    pagination: Annotated[PaginationParams, Depends(pagination_params)],
    status_filter: Annotated[
        str | None, Query(alias="status", description="pending|processing|ready|failed")
    ] = None,
    search: Annotated[str | None, Query(max_length=200)] = None,
    collection: Annotated[str | None, Query(max_length=128)] = None,
    extension: Annotated[str | None, Query(max_length=16)] = None,
) -> Page[DocumentRead]:
    """Paginated, filterable document listing."""
    documents, total = DocumentService(session).list_documents(
        current_user.id,
        page=pagination.page,
        page_size=pagination.page_size,
        status=status_filter,
        search=search,
        collection=collection,
        extension=extension,
    )
    return Page[DocumentRead](
        items=[DocumentRead.model_validate(doc) for doc in documents],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
    )


@router.get(
    "/stats",
    summary="Aggregate document counters for the dashboard",
)
def document_stats(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> dict[str, int]:
    """Counts per ingestion status."""
    return DocumentService(session).stats(current_user.id)


@router.get(
    "/{document_id}",
    response_model=DocumentRead,
    summary="Fetch one document",
)
def get_document(
    document_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> DocumentRead:
    """Return a single document owned by the caller."""
    document = DocumentService(session).get_document(document_id, current_user.id)
    return DocumentRead.model_validate(document)


@router.get(
    "/{document_id}/chunks",
    response_model=Page[DocumentChunkRead],
    summary="List the stored chunks of a document",
)
def list_chunks(
    document_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
    pagination: Annotated[PaginationParams, Depends(pagination_params)],
) -> Page[DocumentChunkRead]:
    """Inspect the chunks that were embedded for a document."""
    chunks = DocumentService(session).list_chunks(
        document_id, current_user.id, limit=pagination.page_size
    )
    return Page[DocumentChunkRead](
        items=[DocumentChunkRead.model_validate(chunk) for chunk in chunks],
        total=len(chunks),
        page=pagination.page,
        page_size=pagination.page_size,
    )


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusRead,
    summary="Poll ingestion progress",
)
def document_status(
    document_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> DocumentStatusRead:
    """Combine document state with the latest background task state."""
    service = DocumentService(session)
    document = service.get_document(document_id, current_user.id)
    task = service.get_latest_task(document_id)

    progress = 100 if document.status == "ready" else (task.progress if task else 0)
    if document.status == "failed":
        progress = 100

    return DocumentStatusRead(
        document_id=document.id,
        status=document.status,
        chunk_count=document.chunk_count,
        progress=progress,
        error_message=document.error_message,
        task_id=task.celery_task_id if task else None,
        task_status=task.status if task else None,
    )


@router.patch(
    "/{document_id}",
    response_model=DocumentRead,
    summary="Update document metadata",
)
def update_document(
    document_id: str,
    payload: DocumentUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> DocumentRead:
    """Rename a document or change its tags/collection."""
    document = DocumentService(session).update_document(
        document_id, current_user.id, payload
    )
    return DocumentRead.model_validate(document)


@router.post(
    "/{document_id}/reindex",
    response_model=TaskSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Re-run ingestion for a document",
    dependencies=[Depends(rate_limit_upload)],
)
def reindex_document(
    document_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> TaskSubmitResponse:
    """Dispatch a fresh ingestion run (useful after changing chunk settings)."""
    from app.workers.tasks.ingestion import ingest_document

    service = DocumentService(session)
    document = service.get_document(document_id, current_user.id)

    record = service.register_task(
        name="ingest_document",
        owner_id=current_user.id,
        celery_task_id=None,
        document_id=document.id,
    )
    async_result = ingest_document.apply_async(args=(document.id, record.id))
    record.celery_task_id = async_result.id
    session.commit()

    return TaskSubmitResponse(
        task_id=async_result.id,
        record_id=record.id,
        status="queued",
        resource_id=document.id,
        detail="Re-indexing started.",
    )


@router.delete(
    "/{document_id}",
    response_model=MessageResponse,
    summary="Delete a document and its vectors",
)
def delete_document(
    document_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    """Remove the row, the chunk ledger, the stored file and the vectors."""
    DocumentService(session).delete_document(document_id, current_user.id)
    return MessageResponse(detail="Document deleted.")
