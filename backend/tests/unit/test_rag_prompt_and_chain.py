"""Unit tests for prompt construction and the RAG chain wiring."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.llm.prompt import NO_CONTEXT_PROMPT, PromptBuilder, build_rag_prompt
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
    """Retriever double that returns a fixed list of chunks."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, str]] = []

    def retrieve(self, query, *, user_id, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append((query, user_id))
        return self.chunks


class TestPromptBuilder:
    def test_system_prompt_is_first(self) -> None:
        messages = PromptBuilder().build("Q?", sources=[chunk(1, "Body")])
        assert messages[0].role == "system"
        assert "assistant" in messages[0].content.lower()
        # The prompt must ask for a thorough answer, not a terse one.
        assert "detailed" in messages[0].content.lower()

    def test_context_blocks_are_numbered_and_labelled(self) -> None:
        builder = PromptBuilder()
        messages = builder.build(
            "What is X?",
            sources=[chunk(1, "First body", name="alpha.txt"), chunk(2, "Second body", name="beta.txt")],
        )
        prompt = messages[-1].content
        assert "[1] Source: alpha.txt" in prompt
        assert "[2] Source: beta.txt" in prompt
        assert "Question: What is X?" in prompt

    def test_no_context_uses_the_refusal_instruction(self) -> None:
        messages = PromptBuilder().build("Q?", sources=[])
        assert NO_CONTEXT_PROMPT in messages[-1].content

    def test_history_is_inserted_before_the_question(self) -> None:
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        messages = PromptBuilder().build("Q?", sources=[], history=history)
        roles = [message.role for message in messages]
        assert roles == ["system", "user", "assistant", "user"]
        assert messages[3].content.endswith("Question: Q?")

    def test_history_limit_is_respected(self) -> None:
        history = [{"role": "user", "content": f"q{i}"} for i in range(20)]
        messages = PromptBuilder(max_history_turns=2).build("Q?", sources=[], history=history)
        assert len(messages) == 4  # system + 2 turns + question

    def test_context_is_truncated_to_the_char_budget(self) -> None:
        sources = [
            chunk(1, "x" * 500, name="alpha.txt"),
            chunk(2, "y" * 500, name="beta.txt"),
        ]
        messages = PromptBuilder(max_context_chars=600).build("Q?", sources=sources)
        prompt = messages[-1].content
        # The first block fits; the second is trimmed to the remaining budget.
        assert "[1] Source: alpha.txt" in prompt
        assert "y" * 500 not in prompt
        assert len(prompt) < 800

    def test_dict_sources_are_supported(self) -> None:
        messages = PromptBuilder().build(
            "Q?",
            sources=[{"text": "dict body", "metadata": {"document_name": "dict.txt"}}],
        )
        assert "dict.txt" in messages[-1].content
        assert "dict body" in messages[-1].content

    def test_build_rag_prompt_helper(self) -> None:
        messages = build_rag_prompt("Q?", sources=[chunk(1, "Body")])
        assert messages[-1].role == "user"


class TestRAGChain:
    def test_answer_uses_retrieved_context_and_returns_sources(self) -> None:
        from app.llm.stub import StubLLMProvider

        retriever = FakeRetriever([chunk(1, "The SLA is 99.9 percent uptime.")])
        chain = RAGChain(retriever=retriever, llm=StubLLMProvider())

        result = chain.answer("What is the SLA?", user_id="user-1")

        assert retriever.calls == [("What is the SLA?", "user-1")]
        # Structured output keeps the prose short and puts the retrieved facts in
        # the key points, so check both surfaces for the grounding content.
        grounded = " ".join([result.answer, *result.metadata.get("key_points", [])])
        assert "99.9" in grounded
        assert len(result.sources) == 1
        assert result.latency_ms >= 0
        assert result.model_name == "stub-extractive-v1"

    def test_answer_without_sources_states_ignorance(self) -> None:
        from app.llm.stub import StubLLMProvider

        chain = RAGChain(retriever=FakeRetriever([]), llm=StubLLMProvider())
        result = chain.answer("Unknown topic?", user_id="user-1")
        assert "do not have enough information" in result.answer.lower()

    def test_stream_answer_yields_the_full_text(self) -> None:
        from app.llm.stub import StubLLMProvider

        chain = RAGChain(
            retriever=FakeRetriever([chunk(1, "Streaming works fine.")]),
            llm=StubLLMProvider(),
        )
        streamed = "".join(chain.stream_answer("Q?", user_id="u1"))
        assert "Streaming works fine." in streamed

    def test_build_messages_includes_retrieved_chunks(self) -> None:
        from app.llm.stub import StubLLMProvider

        chain = RAGChain(retriever=FakeRetriever([]), llm=StubLLMProvider())
        messages = chain.build_messages(
            "Q?", sources=[chunk(1, "Context body", name="guide.md")]
        )
        assert "guide.md" in messages[-1].content

    def test_config_is_honoured(self) -> None:
        chain = RAGChain(config=Settings(retrieval_top_k=7))
        assert chain._config.retrieval_top_k == 7  # noqa: SLF001 - deliberate check
