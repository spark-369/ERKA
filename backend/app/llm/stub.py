"""Deterministic offline LLM provider.

Used when ``LLM_PROVIDER=stub`` (tests, CI, laptops without model weights).
It does not "think": it answers extractively from the numbered context blocks so
the surrounding pipeline (retrieval -> prompt -> generate -> persistence) is fully
exercised without a 1 GB download.
"""

from __future__ import annotations

import re
import time
from typing import Any

from app.core.config import Settings, get_settings
from app.llm.base import GenerationResult, LLMProvider, Message

_CONTEXT_BLOCK = re.compile(r"^\[(\d+)\]\s*Source:\s*(.+)$", re.MULTILINE)

#: The trailing prompt marker (``PromptBuilder`` appends ``\n\nQuestion: ...``).
_QUESTION_MARKER = re.compile(r"\n+Question:")

#: Reply used when retrieval produced no usable context at all.
NO_CONTEXT_ANSWER = (
    "I do not have enough information in the knowledge base to answer "
    "that. Please upload a document that covers this topic and try again."
)


class StubLLMProvider(LLMProvider):
    """Extractive, dependency-free generator with a stable output format.

    It does not "think": it quotes back the numbered context blocks the retriever
    supplied. Every block is included (one line per source), so a knowledge base
    with several documents is reflected in the answer instead of just the
    top-ranked chunk.
    """

    #: Characters quoted from each source block before truncation.
    _MAX_BLOCK_CHARS = 320

    def __init__(
        self,
        *,
        config: Settings | None = None,
        model_name: str = "stub-extractive-v1",
        delay_seconds: float = 0.0,
    ) -> None:
        self._config = config or get_settings()
        self.model_name = model_name
        self._delay = max(delay_seconds, 0.0)

    # ------------------------------------------------------------- internals
    @staticmethod
    def _latest_user_message(messages: list[Message]) -> str:
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return ""

    def _context_sections(self, prompt: str) -> list[tuple[str, str, str]]:
        """Extract ``(index, document_name, body)`` for every context block."""
        headers = list(_CONTEXT_BLOCK.finditer(prompt))
        if not headers:
            return []

        # Anything after the trailing "Question:" marker is not context.
        question_marker = _QUESTION_MARKER.search(prompt)
        context_end = question_marker.start() if question_marker else len(prompt)

        sections: list[tuple[str, str, str]] = []
        for position, header in enumerate(headers):
            index = header.group(1)
            document_name = header.group(2).strip()
            body_end = (
                headers[position + 1].start()
                if position + 1 < len(headers)
                else context_end
            )
            body = prompt[header.end() : body_end]
            # Trim the "---" separator that joins adjacent blocks.
            body = re.sub(r"\n\n---\s*$", "", body)
            body = " ".join(body.split())
            if not body:
                continue
            if len(body) > self._MAX_BLOCK_CHARS:
                body = body[: self._MAX_BLOCK_CHARS].rsplit(" ", 1)[0] + "..."
            sections.append((index, document_name, body))
        return sections

    def _compose(self, messages: list[Message]) -> str:
        prompt = self._latest_user_message(messages)
        sections = self._context_sections(prompt)
        if not sections:
            return NO_CONTEXT_ANSWER

        # Quote every retrieved source so multi-document answers stay grounded.
        citations = ", ".join(f"[{index}]" for index, _, _ in sections)
        body = "\n\n".join(
            f"[{index}] {document_name}: {text}"
            for index, document_name, text in sections
        )
        return (
            f"Based on the knowledge base ({citations}), here is what I found:\n\n"
            f"{body}\n\n"
            f"Sources used: {citations}."
        )

    # ---------------------------------------------------------------- public
    def generate(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        if self._delay:
            time.sleep(self._delay)
        text = self._compose(messages)
        prompt_tokens = sum(self.count_tokens(m.content) for m in messages)
        return GenerationResult(
            text=text,
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=self.count_tokens(text),
            finish_reason="stop",
            metadata={"provider": "stub", "deterministic": True},
        )

    def health(self) -> dict[str, Any]:
        return {"provider": "stub", "model": self.model_name, "status": "ok"}
