"""Unit tests for the ingestion pipeline and document service."""

from __future__ import annotations

import pytest

from app.core.exceptions import (
    ConflictError,
    FileValidationError,
    NotFoundError,
)
from app.rag.embeddings import HashingEmbeddingProvider
from app.rag.pipeline import IngestionPipeline
from app.services.document_service import DocumentService

pytestmark = pytest.mark.unit


@pytest.fixture()
def pipeline(vector_store) -> IngestionPipeline:
    """Pipeline wired to the per-test Chroma collection and hash embeddings."""
    vector_store.reset()
    return IngestionPipeline(
        embeddings=HashingEmbeddingProvider(dimension=64),
        store=vector_store,
    )


class TestIngestionPipeline:
    def test_ingest_bytes_produces_chunks_and_vectors(self, pipeline, vector_store) -> None:
        text = "\n\n".join(f"Section {i}: " + ("content " * 40) for i in range(6))
        result = pipeline.ingest_bytes(
            text.encode(),
            document_id="doc-1",
            user_id="user-1",
            filename="guide.txt",
        )
        assert result.chunk_count > 1
        assert result.chunk_count == len(result.chunk_records)
        assert result.token_count > 0
        assert vector_store.count() == result.chunk_count

    def test_chunk_records_are_complete(self, pipeline) -> None:
        result = pipeline.ingest_bytes(
            b"Only one short paragraph here.",
            document_id="doc-2",
            user_id="user-1",
            filename="short.txt",
        )
        record = result.chunk_records[0]
        assert set(record) == {
            "chunk_index",
            "vector_id",
            "content",
            "char_count",
            "token_count",
        }
        assert record["vector_id"] == "doc-2:0"

    def test_vectors_are_retrievable_with_metadata(self, pipeline, vector_store) -> None:
        pipeline.ingest_bytes(
            b"Vacation policy: employees get twenty days of paid leave.",
            document_id="doc-3",
            user_id="user-1",
            filename="hr.txt",
        )
        matches = vector_store.query(
            HashingEmbeddingProvider(dimension=64).embed_query("vacation policy"),
            top_k=1,
            where={"user_id": "user-1"},
        )
        assert matches
        assert matches[0].metadata["document_id"] == "doc-3"
        assert matches[0].metadata["document_name"] == "hr.txt"

    def test_re_ingesting_replaces_previous_vectors(self, pipeline, vector_store) -> None:
        pipeline.ingest_bytes(
            b"version one " * 100, document_id="doc-4", user_id="u1", filename="v.txt"
        )
        first_count = vector_store.count()
        pipeline.ingest_bytes(
            b"version two", document_id="doc-4", user_id="u1", filename="v.txt"
        )
        assert vector_store.count() == 1
        assert first_count > 1

    def test_progress_callback_is_invoked(self, pipeline) -> None:
        seen: list[int] = []
        pipeline.ingest_bytes(
            b"progress test " * 50,
            document_id="doc-5",
            user_id="u1",
            filename="p.txt",
            progress=lambda value, _message: seen.append(value),
        )
        assert seen == sorted(seen)
        assert seen[-1] == 95

    def test_ingest_path_reads_from_disk(self, pipeline, tmp_path) -> None:
        path = tmp_path / "stored.txt"
        path.write_text("Stored file content for ingestion. " * 20)
        result = pipeline.ingest_path(
            path, document_id="doc-6", user_id="u1", filename="stored.txt"
        )
        assert result.chunk_count >= 1

    def test_delete_document_removes_vectors(self, pipeline, vector_store) -> None:
        pipeline.ingest_bytes(
            b"delete me please", document_id="doc-7", user_id="u1", filename="d.txt"
        )
        assert vector_store.count() > 0
        pipeline.delete_document("doc-7")
        assert vector_store.count() == 0

    def test_corrupt_pdf_raises_a_processing_error(self, pipeline) -> None:
        """Unparseable binaries must fail loudly rather than index garbage."""
        from app.core.exceptions import DocumentProcessingError

        with pytest.raises(DocumentProcessingError):
            pipeline.ingest_bytes(
                b"this is definitely not a valid PDF payload",
                document_id="doc-8",
                user_id="u1",
                filename="broken.pdf",
            )


class TestDocumentServiceValidation:
    def test_accepts_allowed_extension(self, session) -> None:
        service = DocumentService(session)
        assert service.validate_upload(
            filename="a.pdf", size_bytes=10, content_type=None
        ) == ".pdf"

    def test_rejects_unsupported_extension(self, session) -> None:
        with pytest.raises(FileValidationError) as excinfo:
            DocumentService(session).validate_upload(
                filename="malware.exe", size_bytes=10, content_type=None
            )
        assert excinfo.value.code == "unsupported_extension"

    def test_rejects_missing_extension(self, session) -> None:
        with pytest.raises(FileValidationError) as excinfo:
            DocumentService(session).validate_upload(
                filename="noextension", size_bytes=10, content_type=None
            )
        assert excinfo.value.code == "missing_extension"

    def test_rejects_empty_file(self, session) -> None:
        with pytest.raises(FileValidationError) as excinfo:
            DocumentService(session).validate_upload(
                filename="a.txt", size_bytes=0, content_type=None
            )
        assert excinfo.value.code == "empty_file"

    def test_rejects_oversized_file(self, session) -> None:
        service = DocumentService(session)
        too_big = service._config.max_upload_size_bytes + 1  # noqa: SLF001
        with pytest.raises(FileValidationError) as excinfo:
            service.validate_upload(
                filename="a.txt", size_bytes=too_big, content_type=None
            )
        assert excinfo.value.code == "file_too_large"

    def test_checksum_is_stable_and_content_sensitive(self) -> None:
        checksum = DocumentService.compute_checksum
        assert checksum(b"abc") == checksum(b"abc")
        assert checksum(b"abc") != checksum(b"abd")


class TestDocumentServiceLifecycle:
    @pytest.fixture()
    def owner(self, user_factory):
        return user_factory()

    def test_create_document_persists_metadata(self, session, owner) -> None:
        service = DocumentService(session)
        document = service.create_document(
            owner_id=owner.id, filename="notes.txt", data=b"hello world"
        )
        assert document.status == "pending"
        assert document.extension == ".txt"
        assert document.size_bytes == 11
        assert document.owner_id == owner.id

    def test_duplicate_upload_is_rejected(self, session, owner) -> None:
        service = DocumentService(session)
        service.create_document(owner_id=owner.id, filename="dup.txt", data=b"same bytes")
        with pytest.raises(ConflictError) as excinfo:
            service.create_document(
                owner_id=owner.id, filename="dup.txt", data=b"same bytes"
            )
        assert excinfo.value.code == "duplicate_document"

    def test_ownership_is_enforced_on_read(self, session, user_factory) -> None:
        owner = user_factory(email="owner@example.com")
        other = user_factory(email="other@example.com")
        service = DocumentService(session)
        document = service.create_document(
            owner_id=owner.id, filename="private.txt", data=b"secret"
        )
        with pytest.raises(NotFoundError):
            service.get_document(document.id, other.id)

    def test_status_transitions(self, session, owner) -> None:
        service = DocumentService(session)
        document = service.create_document(
            owner_id=owner.id, filename="flow.txt", data=b"content"
        )
        service.mark_processing(document)
        assert document.status == "processing"
        service.mark_ready(document, chunk_count=3, token_count=120)
        assert document.status == "ready"
        assert document.chunk_count == 3
        assert document.indexed_at is not None
        service.mark_failed(document, "boom")
        assert document.status == "failed"
        assert document.error_message == "boom"

    def test_record_chunks_replaces_rows(self, session, owner) -> None:
        service = DocumentService(session)
        document = service.create_document(
            owner_id=owner.id, filename="chunks.txt", data=b"content"
        )
        service.record_chunks(
            document.id,
            [{"chunk_index": 0, "vector_id": "v0", "content": "a", "token_count": 1}],
        )
        assert len(service.list_chunks(document.id, owner.id)) == 1
        service.record_chunks(
            document.id,
            [
                {"chunk_index": 0, "vector_id": "v0", "content": "a", "token_count": 1},
                {"chunk_index": 1, "vector_id": "v1", "content": "b", "token_count": 1},
            ],
        )
        assert len(service.list_chunks(document.id, owner.id)) == 2

    def test_stats_aggregate_by_status(self, session, owner) -> None:
        service = DocumentService(session)
        first = service.create_document(owner_id=owner.id, filename="a.txt", data=b"a")
        service.create_document(owner_id=owner.id, filename="b.txt", data=b"b")
        service.mark_ready(first, chunk_count=1, token_count=1)
        stats = service.stats(owner.id)
        assert stats["total"] == 2
        assert stats["ready"] == 1
        assert stats["pending"] == 1

    def test_delete_document_removes_row_and_file(self, session, owner) -> None:
        from pathlib import Path

        service = DocumentService(session)
        document = service.create_document(
            owner_id=owner.id, filename="gone.txt", data=b"bye"
        )
        stored_path = Path(document.storage_path)
        assert stored_path.exists()
        service.delete_document(document.id, owner.id)
        assert not stored_path.exists()
        with pytest.raises(NotFoundError):
            service.get_document(document.id, owner.id)
