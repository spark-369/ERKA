"""Document + chunk repositories."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

from sqlalchemy import delete, func, or_, select

from app.models.document import Document, DocumentChunk
from app.repositories.base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    """Queries scoped to the ``documents`` table."""

    model = Document

    def create(
        self,
        *,
        owner_id: str,
        filename: str,
        original_filename: str,
        content_type: str,
        extension: str,
        size_bytes: int,
        checksum: str | None = None,
        storage_path: str | None = None,
        collection: str = "knowledge_base",
        tags: str | None = None,
    ) -> Document:
        document = Document(
            owner_id=owner_id,
            filename=filename,
            original_filename=original_filename,
            content_type=content_type,
            extension=extension,
            size_bytes=size_bytes,
            checksum=checksum,
            storage_path=storage_path,
            collection=collection,
            tags=tags,
            status="pending",
        )
        return self.add(document)

    def get_for_owner(self, document_id: str, owner_id: str) -> Document | None:
        statement = select(Document).where(
            Document.id == document_id, Document.owner_id == owner_id
        )
        return self.first(statement)

    def get_by_checksum(self, owner_id: str, checksum: str) -> Document | None:
        statement = select(Document).where(
            Document.owner_id == owner_id, Document.checksum == checksum
        )
        return self.first(statement)

    def list_for_owner(
        self,
        owner_id: str,
        *,
        offset: int = 0,
        limit: int = 20,
        status: str | None = None,
        search: str | None = None,
        collection: str | None = None,
        extension: str | None = None,
    ) -> list[Document]:
        statement = self._owner_query(
            owner_id,
            status=status,
            search=search,
            collection=collection,
            extension=extension,
        ).order_by(Document.created_at.desc())
        return self.list(statement, offset=offset, limit=limit)

    def count_for_owner(
        self,
        owner_id: str,
        *,
        status: str | None = None,
        search: str | None = None,
        collection: str | None = None,
        extension: str | None = None,
    ) -> int:
        statement = self._owner_query(
            owner_id,
            status=status,
            search=search,
            collection=collection,
            extension=extension,
        )
        return self.count(statement)

    def _owner_query(
        self,
        owner_id: str,
        *,
        status: str | None = None,
        search: str | None = None,
        collection: str | None = None,
        extension: str | None = None,
    ):
        statement = select(Document).where(Document.owner_id == owner_id)
        if status:
            statement = statement.where(Document.status == status)
        if collection:
            statement = statement.where(Document.collection == collection)
        if extension:
            statement = statement.where(Document.extension == extension)
        if search:
            pattern = f"%{search.strip().lower()}%"
            statement = statement.where(
                or_(
                    func.lower(Document.original_filename).like(pattern),
                    func.lower(Document.tags).like(pattern),
                )
            )
        return statement

    def owner_document_ids(self, owner_id: str, *, only_ready: bool = True) -> list[str]:
        statement = select(Document.id).where(Document.owner_id == owner_id)
        if only_ready:
            statement = statement.where(Document.status == "ready")
        return list(self.session.execute(statement).scalars().all())

    def filter_existing_ids(self, owner_id: str, document_ids: Sequence[str]) -> list[str]:
        """Return the subset of ``document_ids`` this owner still has.

        A conversation persists the retrieval scope it was created with, so a
        scope captured before documents were deleted (and re-uploaded under new
        ids) points at rows - and therefore vectors - that no longer exist.
        Filtering here stops a stale scope from silently turning every later
        question into "no sources".
        """
        if not document_ids:
            return []
        statement = select(Document.id).where(
            Document.owner_id == owner_id, Document.id.in_(list(document_ids))
        )
        return list(self.session.execute(statement).scalars().all())

    # ------------------------------------------------------ state transitions
    def mark_processing(self, document: Document) -> Document:
        document.status = "processing"
        document.error_message = None
        self.session.flush()
        return document

    def mark_ready(
        self, document: Document, *, chunk_count: int, token_count: int
    ) -> Document:
        document.status = "ready"
        document.chunk_count = chunk_count
        document.token_count = token_count
        document.error_message = None
        document.indexed_at = datetime.now(timezone.utc)
        self.session.flush()
        return document

    def mark_failed(self, document: Document, error: str) -> Document:
        document.status = "failed"
        document.error_message = error[:2000]
        self.session.flush()
        return document

    def reset_for_reindex(self, document: Document) -> Document:
        document.status = "pending"
        document.error_message = None
        document.chunk_count = 0
        document.token_count = 0
        document.indexed_at = None
        self.session.flush()
        return document


class DocumentChunkRepository(BaseRepository[DocumentChunk]):
    """Queries scoped to the ``document_chunks`` ledger."""

    model = DocumentChunk

    def replace_for_document(
        self, document_id: str, chunks: list[dict]
    ) -> list[DocumentChunk]:
        """Atomically swap a document's chunks (used on re-index)."""
        self.session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        instances = [
            DocumentChunk(
                document_id=document_id,
                chunk_index=chunk["chunk_index"],
                vector_id=chunk["vector_id"],
                content=chunk["content"],
                char_count=chunk.get("char_count", len(chunk["content"])),
                token_count=chunk.get("token_count", 0),
            )
            for chunk in chunks
        ]
        if instances:
            self.add_all(instances)
        return instances

    def list_for_document(
        self, document_id: str, *, limit: int = 100
    ) -> list[DocumentChunk]:
        statement = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index.asc())
        )
        return self.list(statement, limit=limit)

    def count_for_document(self, document_id: str) -> int:
        statement = select(DocumentChunk).where(DocumentChunk.document_id == document_id)
        return self.count(statement)

    def total_tokens_for_document(self, document_id: str) -> int:
        statement = select(func.coalesce(func.sum(DocumentChunk.token_count), 0)).where(
            DocumentChunk.document_id == document_id
        )
        return int(self.session.scalar(statement) or 0)
