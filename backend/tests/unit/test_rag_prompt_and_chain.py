"""Unit tests for prompt construction and the RAG chain wiring."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.llm.base import Message
from app.llm.prompt import NO_CONTEXT_PROMPT, PromptBuilder, build_rag_prompt
from app.rag.chain import RAGChain, clean_answer
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
        # The extractive stub quotes the retrieved fact straight into the answer.
        assert "99.9" in result.answer
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
            retriever=FakeRetriever([chunk(1, "Answer works fine.")]),
            llm=StubLLMProvider(),
        )
        streamed = "".join(chain.stream_answer("Q?", user_id="u1"))
        assert "Answer works fine." in streamed

    def test_stream_tokens_orchestrates_retrieval_then_generation(self) -> None:
        from app.llm.stub import StubLLMProvider

        retriever = FakeRetriever([chunk(1, "Streamed context body.")])
        chain = RAGChain(retriever=retriever, llm=StubLLMProvider())

        streamed = "".join(chain.stream_tokens("Q?", user_id="user-1"))

        # Retrieval ran (once) and its context reached the model.
        assert retriever.calls == [("Q?", "user-1")]
        assert "Streamed context body." in streamed

    def test_build_stream_tokens_reuses_supplied_sources(self) -> None:
        from app.llm.stub import StubLLMProvider

        retriever = FakeRetriever([])
        chain = RAGChain(retriever=retriever, llm=StubLLMProvider())

        streamed = "".join(
            chain.build_stream_tokens(
                "Q?", sources=[chunk(1, "Reused body", name="reuse.txt")]
            )
        )

        assert "Reused body" in streamed
        # Using pre-retrieved sources must not trigger a second retrieval.
        assert retriever.calls == []

    def test_base_provider_stream_falls_back_to_generate(self) -> None:
        """A provider without real streaming still streams its full text once."""
        from app.llm.stub import StubLLMProvider

        provider = StubLLMProvider()
        prompt = "[1] Source: a.txt\nSome content here.\nQuestion: What?"
        chunks = list(provider.generate_stream([Message(role="user", content=prompt)]))
        assert len(chunks) == 1
        assert "Some content here." in chunks[0]

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


class TestCleanAnswer:
    def test_numbered_list_becomes_one_paragraph(self) -> None:
        raw = (
            "1. **Next.js**: a React framework.\n"
            "2. **PostgreSQL**: a relational database.\n"
            "3. **Prisma ORM**: a database toolkit.\n"
        )
        cleaned = clean_answer(raw)
        assert "\n" not in cleaned
        assert "1." not in cleaned and "**" not in cleaned
        assert cleaned.endswith(".")
        # All three facts survive, in order.
        assert cleaned.index("Next.js") < cleaned.index("PostgreSQL") < cleaned.index("Prisma")

    def test_duplicate_sentences_are_removed(self) -> None:
        raw = (
            "Google Gemini API is used for summaries. "
            "Google Gemini API is used for summaries. "
            "Prisma ORM manages the database."
        )
        cleaned = clean_answer(raw)
        assert cleaned.count("used for summaries") == 1
        assert "Prisma ORM manages the database" in cleaned

    def test_leading_meta_opener_is_dropped(self) -> None:
        cleaned = clean_answer(
            "Based on the provided information, the SLA is 99.9 percent uptime."
        )
        assert cleaned.startswith("The SLA is 99.9")

    def test_sources_listing_and_urls_are_dropped(self) -> None:
        raw = (
            "The refund window is thirty days.\n\n"
            "Sources used: [1], [2].\n"
            "https://example.com/docs\n"
            "References: handbook.pdf\n"
        )
        cleaned = clean_answer(raw)
        assert cleaned == "The refund window is thirty days."
        assert "http" not in cleaned and "References" not in cleaned

    def test_truncated_tail_is_trimmed_to_a_complete_sentence(self) -> None:
        raw = "The first point is complete. The second point is also complete. The third is cut off mid"
        cleaned = clean_answer(raw)
        assert cleaned.endswith("complete.")
        assert "cut off" not in cleaned

    def test_empty_input_returns_empty(self) -> None:
        assert clean_answer("") == ""
        assert clean_answer("   \n  ") == ""

    def test_chain_cleans_generated_answer(self) -> None:
        from app.llm.base import GenerationResult, LLMProvider

        class ListyLLM(LLMProvider):
            model_name = "listy"

            def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
                return GenerationResult(
                    text=(
                        "Based on the information provided:\n"
                        "1. **Alpha**: first thing.\n"
                        "2. **Beta**: second thing.\n"
                        "2. **Beta**: second thing.\n"
                    ),
                    model_name=self.model_name,
                )

        chain = RAGChain(retriever=FakeRetriever([]), llm=ListyLLM())
        result = chain.answer("Q?", user_id="u1")
        assert result.answer.startswith("Alpha")
        assert "**" not in result.answer and "\n" not in result.answer
        assert result.answer.count("Beta") == 1
        assert result.answer.endswith(".")

    def test_paraphrased_restatement_is_dropped(self) -> None:
        """A re-worded repeat of an earlier fact must not survive (user report)."""
        raw = (
            "It integrates seamlessly with other cloud services like AWS, Azure, "
            "and Google Cloud Platform. "
            "Integration with Other Services: It supports various cloud platforms, "
            "making it easy to integrate with other services like AWS, Azure, "
            "and IBM Cloud."
        )
        cleaned = clean_answer(raw)
        # The second sentence restates the first; only one "AWS" sentence remains
        # and the paraphrase's unique marker ("IBM") is gone.
        assert cleaned.lower().count("aws") == 1
        assert "ibm" not in cleaned.lower()

    def test_distinct_sentences_are_kept(self) -> None:
        raw = (
            "Cloud IDE is a cloud-based development environment. "
            "It offers encryption and access controls for security. "
            "It includes automation through plugins."
        )
        cleaned = clean_answer(raw)
        assert "development environment" in cleaned
        assert "encryption" in cleaned
        assert "automation" in cleaned

    def test_inline_list_numbers_are_stripped(self) -> None:
        raw = (
            "Cloud IDE includes features such as: 1. Integration with other "
            "services. 2. Scalability under heavy load. 3. Security controls."
        )
        cleaned = clean_answer(raw)
        assert "1." not in cleaned and "2." not in cleaned and "3." not in cleaned
        assert "Integration with other services" in cleaned
        assert "Scalability under heavy load" in cleaned


class TestIncrementalCleanStream:
    def test_emits_each_finished_sentence_as_it_arrives(self) -> None:
        from app.rag.chain import clean_answer_stream

        # Simulate token fragments: each tuple is one provider chunk.
        fragments = ["1. **Next.js** ", "is a framework. ", "2. **Prisma** ", "is an ORM."]
        out = list(clean_answer_stream(iter(fragments)))
        # More than one yield => it streamed progressively, not one final blob.
        assert len(out) >= 2
        joined = "".join(out)
        assert "**" not in joined and "1." not in joined
        assert "Next.js" in joined and "Prisma" in joined

    def test_duplicates_are_suppressed_while_streaming(self) -> None:
        from app.rag.chain import clean_answer_stream

        fragments = [
            "The SLA is 99.9 percent. ",
            "The SLA is 99.9 percent. ",
            "Backups run nightly.",
        ]
        joined = "".join(clean_answer_stream(iter(fragments)))
        assert joined.count("99.9 percent") == 1

    def test_meta_opener_is_dropped_and_not_lost_when_all_meta(self) -> None:
        from app.rag.chain import clean_answer_stream

        joined = "".join(
            clean_answer_stream(
                iter(["Based on the information provided, ", "the answer is forty-two."])
            )
        )
        assert joined.startswith("The answer is forty-two")

    def test_streaming_chain_yields_progressively(self) -> None:
        from app.llm.base import LLMProvider

        class TokenLLM(LLMProvider):
            model_name = "token"

            def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
                raise AssertionError("streaming must not call generate()")

            def generate_stream(self, messages, **kwargs):  # noqa: ANN001, ANN003
                yield "First sentence here. "
                yield "Second sentence follows. "

        chain = RAGChain(retriever=FakeRetriever([]), llm=TokenLLM())
        out = list(chain.stream_tokens("Q?", user_id="u1"))
        assert len(out) >= 2
        assert "First sentence here." in "".join(out)


