"""Provider-agnostic LLM interface.

The RAG pipeline depends only on :class:`LLMProvider`, never on a concrete
implementation. That keeps the Qwen/transformers dependency swappable (and lets
the test suite inject :class:`~app.llm.stub.StubLLMProvider`).
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

MessageRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    """A single chat message handed to the model."""

    role: MessageRole
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class GenerationResult:
    """Output of a completion call plus bookkeeping metadata."""

    text: str
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(abc.ABC):
    """Contract every inference backend must satisfy."""

    #: Human readable model identifier surfaced to clients.
    model_name: str = "unknown"

    @abc.abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Return a full, synchronous completion for ``messages``."""

    def generate_stream(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield the completion incrementally, piece by piece.

        The default implementation calls :meth:`generate` once and yields the
        whole text, so every provider supports streaming without extra work.
        Real backends override this to emit output *as it is produced*, which is
        what makes the SSE endpoint feel fast: the first words reach the client
        in a fraction of the time the full answer would take to generate.
        """
        result = self.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        if result.text:
            yield result.text

    def health(self) -> dict[str, Any]:
        """Optional readiness probe reported by ``/health``."""
        return {"provider": type(self).__name__, "model": self.model_name, "status": "ok"}

    def warmup(self) -> None:
        """Optionally pre-load weights (no-op by default)."""
        return None

    def count_tokens(self, text: str) -> int:
        """Rough token estimate; real providers override with a tokenizer."""
        if not text:
            return 0
        return max(len(text) // 4, 1)
