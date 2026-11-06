"""Unit tests for embedding providers and the LLM stub."""

from __future__ import annotations

import math

import pytest

from app.core.config import Settings
from app.llm.base import Message
from app.llm.factory import build_llm_provider
from app.llm.stub import StubLLMProvider
from app.rag.embeddings import (
    HashingEmbeddingProvider,
    build_embedding_provider,
)

pytestmark = pytest.mark.unit


class TestHashingEmbeddings:
    def test_dimension_matches_configuration(self) -> None:
        provider = HashingEmbeddingProvider(dimension=32)
        vector = provider.embed_query("hello world")
        assert len(vector) == 32
        assert provider.dimension == 32

    def test_vectors_are_deterministic(self) -> None:
        provider = HashingEmbeddingProvider(dimension=64)
        assert provider.embed_query("same text") == provider.embed_query("same text")

    def test_different_text_produces_different_vectors(self) -> None:
        provider = HashingEmbeddingProvider(dimension=64)
        assert provider.embed_query("alpha") != provider.embed_query("beta")

    def test_vectors_are_normalised(self) -> None:
        provider = HashingEmbeddingProvider(dimension=64, normalize=True)
        vector = provider.embed_query("normalise me please")
        norm = math.sqrt(sum(value * value for value in vector))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_empty_text_yields_zero_vector(self) -> None:
        provider = HashingEmbeddingProvider(dimension=16)
        assert provider.embed_query("") == [0.0] * 16

    def test_embed_documents_preserves_order_and_count(self) -> None:
        provider = HashingEmbeddingProvider(dimension=32)
        vectors = provider.embed_documents(["one", "two", "three"])
        assert len(vectors) == 3
        assert vectors[0] == provider.embed_query("one")

    def test_empty_batch_returns_empty_list(self) -> None:
        assert HashingEmbeddingProvider().embed_documents([]) == []

    def test_minimum_dimension_is_enforced(self) -> None:
        assert HashingEmbeddingProvider(dimension=1).dimension == 8


class TestEmbeddingFactory:
    def test_hash_provider_from_settings(self) -> None:
        provider = build_embedding_provider(
            Settings(embedding_provider="hash", embedding_dimension=16)
        )
        assert isinstance(provider, HashingEmbeddingProvider)
        assert provider.dimension == 16

    def test_huggingface_provider_from_settings(self) -> None:
        provider = build_embedding_provider(
            Settings(embedding_provider="huggingface", embedding_model="custom-model")
        )
        assert provider.model_name == "custom-model"

    def test_health_payload(self) -> None:
        health = HashingEmbeddingProvider(dimension=32).health()
        assert health["status"] == "ok"
        assert health["dimension"] == 32


class TestStubLLM:
    def test_answers_extractively_from_context(self) -> None:
        prompt = (
            "Context from the knowledge base:\n\n"
            "[1] Source: handbook.txt\n"
            "The refund window is 30 days from purchase.\n\n"
            "Question: What is the refund window?"
        )
        result = StubLLMProvider().generate([Message(role="user", content=prompt)])
        assert "30 days" in result.text
        assert "[1]" in result.text
        assert result.finish_reason == "stop"
        assert result.completion_tokens > 0

    def test_answers_cover_every_retrieved_document(self) -> None:
        """Regression: the stub used to quote only the top-ranked block."""
        prompt = (
            "Context from the knowledge base:\n\n"
            "[1] Source: refunds.txt\n"
            "The refund window is 30 days from purchase.\n\n"
            "---\n\n"
            "[2] Source: vacation.txt\n"
            "Vacation policy grants twenty paid days.\n\n"
            "Question: Tell me about the policies?"
        )
        result = StubLLMProvider().generate([Message(role="user", content=prompt)])
        # Both documents must appear, not just the first one.
        assert "30 days" in result.text
        assert "twenty paid days" in result.text
        assert "refunds.txt" in result.text
        assert "vacation.txt" in result.text
        assert "[1]" in result.text and "[2]" in result.text

    def test_admits_when_no_context_is_available(self) -> None:
        result = StubLLMProvider().generate(
            [Message(role="user", content="Question: Anything at all?")]
        )
        assert "do not have enough information" in result.text.lower()

    def test_output_is_deterministic(self) -> None:
        provider = StubLLMProvider()
        prompt = "[1] Source: a.txt\nSome content here.\nQuestion: What?"
        first = provider.generate([Message(role="user", content=prompt)]).text
        second = provider.generate([Message(role="user", content=prompt)]).text
        assert first == second

    def test_health_reports_ok(self) -> None:
        assert StubLLMProvider().health()["status"] == "ok"

    def test_count_tokens_heuristic(self) -> None:
        assert StubLLMProvider().count_tokens("a" * 40) == 10


class TestLLMFactory:
    def test_stub_provider_selected_by_settings(self) -> None:
        assert isinstance(
            build_llm_provider(Settings(llm_provider="stub")), StubLLMProvider
        )

    def test_transformers_provider_selected_by_settings(self) -> None:
        provider = build_llm_provider(
            Settings(llm_provider="transformers", llm_model_id="Qwen/Qwen2.5-0.5B-Instruct")
        )
        assert provider.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
        # No weights are loaded until the first generation call.
        assert provider.health()["loaded"] is False
