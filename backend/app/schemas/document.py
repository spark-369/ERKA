"""Document upload / metadata schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class DocumentRead(ORMModel):
    """Full document metadata as returned by the API."""

    id: str
    owner_id: str
    filename: str
    original_filename: str
    content_type: str
    extension: str
    size_bytes: int
    checksum: str | None = None
    status: str
    error_message: str | None = None
    chunk_count: int
    token_count: int
    collection: str
    tags: str | None = None
    indexed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class DocumentChunkRead(ORMModel):
    """A stored chunk (useful for debugging retrieval quality)."""

    id: str
    document_id: str
    chunk_index: int
    vector_id: str
    content: str
    char_count: int
    token_count: int


class DocumentUploadResponse(BaseModel):
    """Accepted-upload envelope.

    Uploads are processed asynchronously, so the response returns immediately
    with the ``task_id`` the client should poll (or subscribe to via SSE).
    """

    document: DocumentRead
    task_id: str | None = Field(
        default=None, description="Celery task id for the ingestion pipeline."
    )
    message: str = "Document accepted for processing."


class DocumentStatusRead(BaseModel):
    """Lightweight polling payload for ingestion progress."""

    document_id: str
    status: str
    chunk_count: int = 0
    progress: int = Field(default=0, ge=0, le=100)
    error_message: str | None = None
    task_id: str | None = None
    task_status: str | None = None


class DocumentUpdate(BaseModel):
    """Mutable document metadata."""

    original_filename: str | None = Field(default=None, max_length=512)
    tags: str | None = Field(default=None, max_length=512)
    collection: str | None = Field(default=None, max_length=128)
