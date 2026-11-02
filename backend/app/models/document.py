"""Document + chunk models.

``Document`` holds the metadata that PostgreSQL is good at (ownership, status,
counters, checksums). ``DocumentChunk`` records the text chunks that were pushed
into ChromaDB so the pipeline can be audited, re-embedded or deleted cleanly.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, utcnow

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models.user import User

# Processing lifecycle shared with the Celery ingestion task.
DOCUMENT_STATUSES = ("pending", "processing", "ready", "failed")


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An uploaded source file and its ingestion state."""

    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_owner_status", "owner_id", "status"),
        {"comment": "Uploaded knowledge base documents"},
    )

    owner_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(160), default="application/octet-stream")
    extension: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    status: Mapped[str] = mapped_column(String(24), default="pending", nullable=False, index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    storage_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    indexed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # --- optional knowledge-base scoping -----------------------------------
    collection: Mapped[str] = mapped_column(String(128), default="knowledge_base")
    tags: Mapped[str | None] = mapped_column(String(512), nullable=True)

    owner: Mapped["User"] = relationship(back_populates="documents")
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )

    # ------------------------------------------------------------- helpers
    @property
    def is_ready(self) -> bool:
        return self.status == "ready"

    @property
    def tag_list(self) -> list[str]:
        if not self.tags:
            return []
        return [tag.strip() for tag in self.tags.split(",") if tag.strip()]

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Document id={self.id} name={self.original_filename!r} status={self.status}>"


class DocumentChunk(UUIDPrimaryKeyMixin, Base):
    """A single text chunk mirrored from the vector store."""

    __tablename__ = "document_chunks"
    __table_args__ = (
        Index("ix_document_chunks_document_index", "document_id", "chunk_index"),
        {"comment": "Chunk ledger for the embedded ChromaDB collection"},
    )

    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    vector_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<DocumentChunk doc={self.document_id} index={self.chunk_index}>"
