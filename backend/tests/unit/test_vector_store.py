"""Unit tests for the embedded ChromaDB vector store and retriever."""

from __future__ import annotations

import uuid

import pytest

from app.core.config import Settings
from app.core.exceptions import VectorStoreError
from app.rag.embeddings import HashingEmbeddingProvider
from app.rag.retriever import Retriever

pytestmark = pytest.mark.unit


@pytest.fixture()
def store(tmp_path):
    """A dedicated, empty collection for each test.

    These tests use tiny hand-written vectors (2 dimensions) to keep the
    assertions readable. Chroma fixes a collection's dimensionality on its first
    write and then rejects anything else, so each test gets its own collection
    rather than sharing the 384/64-dimensional one used elsewhere.
    """
    from app.db.vector_store import VectorStore

    return VectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name=f"vectors_{uuid.uuid4().hex[:12]}",
    )


class TestVectorStore:
    def test_add_and_count(self, store) -> None:
        store.add(
            ids=["a", "b"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            documents=["alpha", "beta"],
            metadatas=[{"user_id": "u1"}, {"user_id": "u1"}],
        )
        assert store.count() == 2

    def test_upsert_is_idempotent_for_same_ids(self, store) -> None:
        payload = dict(
            ids=["a"],
            embeddings=[[1.0, 0.0]],
            documents=["alpha"],
            metadatas=[{"user_id": "u1"}],
        )
        store.add(**payload)
        store.add(**payload)
        assert store.count() == 1

    def test_query_returns_sorted_matches(self, store) -> None:
        store.add(
            ids=["near", "far"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            documents=["near doc", "far doc"],
            metadatas=[{"user_id": "u1"}, {"user_id": "u1"}],
        )
        matches = store.query([1.0, 0.0], top_k=2)
        assert [match.id for match in matches] == ["near", "far"]
        assert matches[0].score >= matches[1].score

    def test_query_filters_by_metadata(self, store) -> None:
        store.add(
            ids=["mine", "theirs"],
            embeddings=[[1.0, 0.0], [1.0, 0.0]],
            documents=["mine", "theirs"],
            metadatas=[{"user_id": "u1"}, {"user_id": "u2"}],
        )
        matches = store.query([1.0, 0.0], top_k=5, where={"user_id": "u1"})
        assert [match.id for match in matches] == ["mine"]

    def test_score_threshold_filters_weak_matches(self, store) -> None:
        store.add(
            ids=["orthogonal"],
            embeddings=[[0.0, 1.0]],
            documents=["unrelated"],
            metadatas=[{"user_id": "u1"}],
        )
        assert store.query([1.0, 0.0], top_k=5, score_threshold=0.5) == []

    def test_delete_document_removes_only_that_document(self, store) -> None:
        store.add(
            ids=["d1:0", "d2:0"],
            embeddings=[[1.0, 0.0], [1.0, 0.0]],
            documents=["one", "two"],
            metadatas=[{"document_id": "d1"}, {"document_id": "d2"}],
        )
        store.delete_document("d1")
        assert store.count() == 1

    def test_delete_user_removes_all_tenant_data(self, store) -> None:
        store.add(
            ids=["a", "b"],
            embeddings=[[1.0, 0.0], [1.0, 0.0]],
            documents=["a", "b"],
            metadatas=[{"user_id": "u1"}, {"user_id": "u2"}],
        )
        store.delete_user("u1")
        assert store.count() == 1

    def test_none_metadata_values_are_normalised(self, store) -> None:
        store.add(
            ids=["a"],
            embeddings=[[1.0, 0.0]],
            documents=["alpha"],
            metadatas=[{"user_id": "u1", "tags": None}],
        )
        assert store.count() == 1

    def test_mismatched_payload_lengths_raise(self, store) -> None:
        with pytest.raises(VectorStoreError):
            store.add(
                ids=["a", "b"],
                embeddings=[[1.0, 0.0]],
                documents=["alpha"],
                metadatas=[{"user_id": "u1"}],
            )

    def test_empty_add_is_a_noop(self, store) -> None:
        assert store.add(ids=[], embeddings=[], documents=[], metadatas=[]) == 0

    def test_empty_query_returns_nothing(self, store) -> None:
        assert store.query([]) == []

    def test_persistence_directory_is_recorded(self, vector_store) -> None:
        assert vector_store.persist_directory is not None


class TestRetrieverWhereClause:
    def test_single_user_clause(self) -> None:
        where = Retriever.build_where(user_id="u1")
        assert where == {"user_id": {"$eq": "u1"}}

    def test_document_scope_is_and_ed(self) -> None:
        where = Retriever.build_where(user_id="u1", document_ids=["d1", "d2"])
        assert where == {
            "$and": [
                {"user_id": {"$eq": "u1"}},
                {"document_id": {"$in": ["d1", "d2"]}},
            ]
        }

    def test_collection_filter(self) -> None:
        where = Retriever.build_where(user_id="u1", collection="kb2")
        assert {"collection": {"$eq": "kb2"}} in where["$and"]


class TestRetriever:
    def test_retrieves_only_the_owners_chunks(self, store) -> None:
        embeddings = HashingEmbeddingProvider(dimension=64)
        texts = ["refund policy is 30 days", "vacation policy is 20 days"]
        store.add(
            ids=["u1:0", "u2:0"],
            embeddings=embeddings.embed_documents(texts),
            documents=texts,
            metadatas=[
                {"user_id": "u1", "document_id": "d1", "document_name": "policy.txt"},
                {"user_id": "u2", "document_id": "d2", "document_name": "other.txt"},
            ],
        )
        retriever = Retriever(store=store, embeddings=embeddings)
        results = retriever.retrieve("refund policy", user_id="u1", score_threshold=0.0)
        assert len(results) == 1
        assert results[0].document_id == "d1"
        assert results[0].document_name == "policy.txt"

    def test_empty_query_returns_nothing(self, store) -> None:
        retriever = Retriever(
            store=store, embeddings=HashingEmbeddingProvider(dimension=32)
        )
        assert retriever.retrieve("  ", user_id="u1") == []

    def test_returns_chunks_from_every_document(self, store) -> None:
        """A single query should be able to surface multiple documents."""
        embeddings = HashingEmbeddingProvider(dimension=64)
        texts = [
            "refund policy window is 30 days",
            "vacation policy grants twenty paid days",
        ]
        store.add(
            ids=["d1:0", "d2:0"],
            embeddings=embeddings.embed_documents(texts),
            documents=texts,
            metadatas=[
                {"user_id": "u1", "document_id": "d1", "document_name": "refunds.txt"},
                {"user_id": "u1", "document_id": "d2", "document_name": "vacation.txt"},
            ],
        )
        retriever = Retriever(
            store=store,
            embeddings=embeddings,
            config=Settings(retrieval_top_k=5, retrieval_fetch_k=20),
        )
        results = retriever.retrieve("policy", user_id="u1", score_threshold=0.0)
        document_ids = {chunk.document_id for chunk in results}
        assert document_ids == {"d1", "d2"}

    def test_snippet_is_truncated(self, store) -> None:
        match = type(
            "M",
            (),
            {"id": "x", "text": "word " * 200, "score": 0.5, "metadata": {}},
        )()
        chunk = Retriever(store=store)._to_chunk(match)  # noqa: SLF001
        assert chunk.snippet(50).endswith("...")
        assert len(chunk.snippet(50)) <= 53
