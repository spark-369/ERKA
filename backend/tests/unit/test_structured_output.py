"""Unit tests for constrained (structured) generation with outlines/Qwen2.5."""

from __future__ import annotations

import pytest

from app.llm.base import LLMProvider, Message
from app.llm.structured import (
    AnswerWithCitations,
    parse_answer_payload,
    structured_json_schema,
)
from app.llm.stub import StubLLMProvider
from app.rag.chain import RAGChain
from app.rag.retriever import RetrievedChunk

pytestmark = pytest.mark.unit


def chunk(index: int, text: str, score: float = 0.9, name: str = "doc.txt") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c{index}",
        document_id="doc-1",
        document_name=name,
        text=text,
        score=score,
        chunk_index=index,
    )


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, query, *, user_id, **kwargs):  # noqa: ANN001, ANN003
        return self.chunks


class TestAnswerSchema:
    def test_schema_exposes_structured_fields(self) -> None:
        schema = structured_json_schema()
        assert set(schema["properties"]) >= {"answer", "key_points", "citations"}
        assert "answer" in schema["required"]

    def test_cited_indices_are_normalised(self) -> None:
        payload = AnswerWithCitations(answer="x", citations=["2", "[1]", "2", "bogus"])
        assert payload.cited_indices() == [1, 2]

    def test_points_are_cleaned(self) -> None:
        payload = AnswerWithCitations(answer="x", key_points=[" one ", "", "  "])
        assert payload.points() == ["one"]


class TestParseAnswerPayload:
    def test_bare_json_object(self) -> None:
        payload = parse_answer_payload('{"answer": "hello", "citations": ["1"]}')
        assert payload is not None
        assert payload.answer == "hello"
        assert payload.citations == ["1"]

    def test_fenced_json_block_with_surrounding_prose(self) -> None:
        raw = 'Sure!\n```json\n{"answer": "fenced", "citations": ["3"]}\n```\nDone.'
        payload = parse_answer_payload(raw)
        assert payload is not None
        assert payload.answer == "fenced"
        assert payload.citations == ["3"]

    def test_embedded_json_object(self) -> None:
        raw = 'Here you go: {"answer": "embedded", "citations": []} thanks'
        payload = parse_answer_payload(raw)
        assert payload is not None
        assert payload.answer == "embedded"

    def test_plain_text_is_treated_as_an_answer(self) -> None:
        payload = parse_answer_payload("Just some prose, no JSON here.")
        assert payload is not None
        assert payload.answer == "Just some prose, no JSON here."
        assert payload.citations == []

    def test_empty_input_returns_none(self) -> None:
        assert parse_answer_payload("   ") is None

    def test_json_without_answer_returns_none(self) -> None:
        assert parse_answer_payload('{"citations": ["1"]}') is None


class TestStructuredGeneration:
    def test_base_provider_does_not_support_structured_output(self) -> None:
        class Minimal(LLMProvider):
            def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
                raise NotImplementedError

            def stream(self, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
                raise NotImplementedError

        provider = Minimal()
        assert provider.supports_structured_output() is False
        assert provider.generate_structured([], AnswerWithCitations) is None

    def test_stub_returns_grounded_schema(self) -> None:
        provider = StubLLMProvider()
        assert provider.supports_structured_output() is True
        prompt = (
            "Context from the knowledge base:\n\n"
            "[1] Source: refunds.txt\nThe refund window is 30 days.\n\n"
            "---\n\n"
            "[2] Source: hr.txt\nVacation grants twenty paid days.\n\n"
            "Question: Tell me about the policies?"
        )
        payload = provider.generate_structured(
            [Message(role="user", content=prompt)], AnswerWithCitations
        )
        assert isinstance(payload, AnswerWithCitations)
        # The answer is prose; the raw passages live in the bullet points.
        assert "refunds.txt" in payload.answer or "documents" in payload.answer
        points = payload.points()
        assert len(points) == 2
        assert any("30 days" in point for point in points)
        assert any("twenty paid days" in point for point in points)
        assert payload.cited_indices() == [1, 2]

    def test_stub_structured_without_context(self) -> None:
        payload = StubLLMProvider().generate_structured(
            [Message(role="user", content="Question: Anything?")], AnswerWithCitations
        )
        assert isinstance(payload, AnswerWithCitations)
        assert "do not have enough information" in payload.answer.lower()
        assert payload.citations == []
        assert payload.points() == []

    def test_chain_uses_structured_path_when_supported(self) -> None:
        retriever = FakeRetriever([chunk(1, "The SLA is 99.9 percent uptime.", name="sla.txt")])
        chain = RAGChain(retriever=retriever, llm=StubLLMProvider())
        result = chain.answer("What is the SLA?", user_id="user-1")
        assert result.metadata.get("structured") is True
        assert result.metadata.get("citations") == [1]
        key_points = result.metadata.get("key_points")
        assert key_points
        assert any("99.9" in point for point in key_points)

    def test_chain_falls_back_to_plain_text(self) -> None:
        class NoStructured(StubLLMProvider):
            def supports_structured_output(self) -> bool:
                return False

        retriever = FakeRetriever([chunk(1, "The SLA is 99.9 percent uptime.", name="sla.txt")])
        chain = RAGChain(retriever=retriever, llm=NoStructured())
        result = chain.answer("What is the SLA?", user_id="user-1")
        assert result.metadata.get("structured") is None
        assert "99.9" in result.answer
