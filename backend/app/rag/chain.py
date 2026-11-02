"""The RAG chain: retrieve context, build a prompt, generate an answer.

Kept framework-light on purpose: the LangChain pieces we rely on (splitters,
``Document`` objects) live in ``rag/`` and are used where they add value, while
the chain itself is a plain, testable function object.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.llm.base import GenerationResult, LLMProvider, Message
from app.llm.factory import get_llm_provider
from app.llm.prompt import PromptBuilder
from app.llm.structured import AnswerWithCitations, parse_answer_payload
from app.rag.retriever import RetrievedChunk, Retriever, get_retriever

logger = get_logger(__name__)


@dataclass(slots=True)
class RAGAnswer:
    """A completed RAG turn."""

    answer: str
    sources: list[RetrievedChunk]
    model_name: str
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGChain:
    """Composes retriever + prompt builder + LLM provider."""

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        llm: LLMProvider | None = None,
        prompt_builder: PromptBuilder | None = None,
        config: Settings | None = None,
    ) -> None:
        self._config = config or get_settings()
        self._retriever = retriever
        self._llm = llm
        self._prompt_builder = prompt_builder or PromptBuilder()

    # ----------------------------------------------------------- lazy wiring
    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = get_retriever()
        return self._retriever

    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = get_llm_provider()
        return self._llm

    # ---------------------------------------------------------------- public
    def retrieve(
        self,
        question: str,
        *,
        user_id: str,
        document_ids: list[str] | None = None,
        collection: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve grounding context for a question."""
        return self.retriever.retrieve(
            question,
            user_id=user_id,
            document_ids=document_ids,
            collection=collection,
            top_k=top_k,
        )

    def build_messages(
        self,
        question: str,
        *,
        sources: Sequence[RetrievedChunk],
        history: Sequence[Any] = (),
    ) -> list[Message]:
        """Render the final prompt sent to the model."""
        return self._prompt_builder.build(question, sources=sources, history=history)

    def answer(
        self,
        question: str,
        *,
        user_id: str,
        document_ids: list[str] | None = None,
        history: Sequence[Any] = (),
        top_k: int | None = None,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> RAGAnswer:
        """Retrieve + generate in one call (non-streaming path).

        When the provider supports it, generation is constrained to
        :class:`~app.llm.structured.AnswerWithCitations` so the answer arrives as
        a validated object (with explicit citations) instead of free-form text.
        Anything that goes wrong in the structured path degrades gracefully to
        plain-text generation, so the endpoint never fails because of it.
        """
        started = time.perf_counter()
        sources = self.retrieve(
            question, user_id=user_id, document_ids=document_ids, top_k=top_k
        )
        messages = self.build_messages(question, sources=sources, history=history)

        result = self._generate(
            messages,
            sources=sources,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "RAG answer generated in %sms (sources=%s, tokens=%s)",
            latency_ms,
            len(sources),
            result.completion_tokens,
        )
        return RAGAnswer(
            answer=result.text,
            sources=list(sources),
            model_name=result.model_name,
            latency_ms=latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            metadata=result.metadata,
        )

    def _generate(
        self,
        messages: list[Message],
        *,
        sources: Sequence[RetrievedChunk],
        temperature: float | None,
        max_new_tokens: int | None,
    ) -> GenerationResult:
        """Generate the answer, preferring the constrained-decoding path.

        Order of preference:

        1. ``outlines``-constrained generation (a validated object with explicit
           citations) when the provider and configuration allow it;
        2. plain-text generation, salvaging structured fields with
           :func:`~app.llm.structured.parse_answer_payload` if the model happened
           to emit JSON anyway.

        The plain-text path is always available, so a missing/failed structured
        backend never breaks the endpoint.
        """
        structured = self._structured_result(
            messages,
            sources=sources,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        if structured is not None:
            return structured

        result = self.llm.generate(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        # A model that emits JSON without constrained decoding still gets parsed,
        # so citations survive even when ``outlines`` is unavailable.
        parsed = parse_answer_payload(result.text)
        if parsed is not None and parsed.answer and parsed.answer != result.text:
            result.text = parsed.answer
            result.metadata = {
                **result.metadata,
                "structured": False,
                "parsed": True,
                "citations": parsed.cited_indices(),
                "source_count": len(sources),
            }
        return result

    def generate_structured(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> AnswerWithCitations | None:
        """Constrained generation to :class:`AnswerWithCitations`.

        Returns the validated payload, or ``None`` when structured output is
        disabled/unsupported or the call fails - callers then fall back to plain
        text. Exposed so both the non-streaming and SSE paths share one code
        path (and therefore one answer format).
        """
        if not self._config.llm_structured_output:
            return None
        if not self.llm.supports_structured_output():
            return None

        try:
            payload = self.llm.generate_structured(
                messages,
                AnswerWithCitations,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - never fail the request over this
            logger.warning("Structured generation errored, using plain text: %s", exc)
            return None

        if payload is None or not payload.answer.strip():
            return None
        return payload

    def _structured_result(
        self,
        messages: list[Message],
        *,
        sources: Sequence[RetrievedChunk],
        temperature: float | None,
        max_new_tokens: int | None,
    ) -> GenerationResult | None:
        """Wrap :meth:`generate_structured` in a :class:`GenerationResult`."""
        payload = self.generate_structured(
            messages, temperature=temperature, max_new_tokens=max_new_tokens
        )
        if payload is None:
            return None

        answer = payload.answer.strip()
        return GenerationResult(
            text=answer,
            model_name=self.llm.model_name,
            prompt_tokens=sum(self.llm.count_tokens(m.content) for m in messages),
            completion_tokens=self.llm.count_tokens(answer),
            finish_reason="stop",
            metadata={
                "structured": True,
                "citations": payload.cited_indices(),
                "key_points": payload.points(),
                "source_count": len(sources),
            },
        )

    def stream_answer(
        self,
        question: str,
        *,
        user_id: str,
        document_ids: list[str] | None = None,
        history: Sequence[Any] = (),
        top_k: int | None = None,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> Iterator[str]:
        """Retrieve once, then stream the model's tokens.

        Retrieval happens eagerly so the API can emit a ``sources`` SSE event
        before the first token arrives.
        """
        sources = self.retrieve(
            question, user_id=user_id, document_ids=document_ids, top_k=top_k
        )
        messages = self.build_messages(question, sources=sources, history=history)
        yield from self.llm.stream(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
