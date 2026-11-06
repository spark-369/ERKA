"""Prompt construction for the RAG pipeline.

Kept deliberately separate from the model so prompt engineering can be unit
tested without loading Qwen2.5.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from app.llm.base import Message

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers the user's question using only "
    "the provided context. Give a detailed but clean answer as natural, flowing "
    "prose in your own words - plain sentences or short paragraphs. Do not use "
    "numbered lists, bullet points, or headings, and never repeat the same point "
    "twice. Ignore reference lists, bibliographies, citations, documentation "
    "titles, source file names, and URLs that appear in the context; never "
    "present them as part of the answer. Do not begin with phrases such as "
    "'Based on the information provided' or 'the text states', and do not quote "
    "the context verbatim. Synthesise the information into a clear explanation "
    "that directly answers the question."
)

NO_CONTEXT_PROMPT = (
    "No relevant context was found, so tell the user you cannot answer from the "
    "knowledge base and suggest uploading the relevant document."
)


class PromptBuilder:
    """Builds chat-style prompts from retrieved context and history."""

    def __init__(
        self,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        #: Generous by default: detailed, paragraph-length answers need plenty of
        #: grounding context. Roughly 8k tokens (~32k chars), which leaves the bulk
        #: of the 32k-token context window for the answer. The provider trims the
        #: generation budget automatically whenever the prompt leaves less room.
        max_context_chars: int = 32_000,
        max_history_turns: int = 6,
    ) -> None:
        self.system_prompt = system_prompt
        self.max_context_chars = max_context_chars
        self.max_history_turns = max_history_turns

    # ------------------------------------------------------------- context
    def format_context(self, sources: Sequence[Any]) -> str:
        """Render retrieved chunks as numbered, citable context blocks.

        ``sources`` accepts anything with ``text``/``content`` and optional
        ``metadata``/``document_name`` attributes (see ``RetrievedChunk``).
        """
        blocks: list[str] = []
        used_chars = 0
        for index, source in enumerate(sources, start=1):
            text = self._text_of(source)
            if not text:
                continue
            document_name = self._document_name(source)
            header = f"[{index}] Source: {document_name}"
            block = f"{header}\n{text.strip()}"
            if used_chars + len(block) > self.max_context_chars:
                remaining = self.max_context_chars - used_chars
                if remaining <= len(header) + 50:
                    break
                block = block[:remaining]
            blocks.append(block)
            used_chars += len(block)
        return "\n\n---\n\n".join(blocks)

    def build(
        self,
        question: str,
        *,
        sources: Sequence[Any] = (),
        history: Iterable[Any] = (),
    ) -> list[Message]:
        """Compose the final message list sent to the model."""
        context = self.format_context(sources)
        if context:
            context_section = f"Context from the knowledge base:\n\n{context}"
        else:
            context_section = NO_CONTEXT_PROMPT

        messages: list[Message] = [
            Message(role="system", content=self.system_prompt),
        ]

        for turn in self._history_turns(history):
            messages.append(turn)

        messages.append(
            Message(
                role="user",
                content=f"{context_section}\n\nQuestion: {question.strip()}",
            )
        )
        return messages

    # ------------------------------------------------------------- helpers
    def _history_turns(self, history: Iterable[Any]) -> list[Message]:
        turns: list[Message] = []
        for item in history:
            role = getattr(item, "role", None) or (
                item.get("role") if isinstance(item, dict) else None
            )
            content = getattr(item, "content", None) or (
                item.get("content") if isinstance(item, dict) else None
            )
            if role in ("user", "assistant") and content:
                turns.append(Message(role=role, content=str(content)))
        return turns[-self.max_history_turns :]

    @staticmethod
    def _text_of(source: Any) -> str:
        for attribute in ("text", "content", "page_content"):
            value = getattr(source, attribute, None)
            if value:
                return str(value)
        if isinstance(source, dict):
            for key in ("text", "content", "page_content"):
                if source.get(key):
                    return str(source[key])
        return ""

    @staticmethod
    def _document_name(source: Any) -> str:
        metadata = getattr(source, "metadata", None)
        if metadata is None and isinstance(source, dict):
            metadata = source.get("metadata")
        if isinstance(metadata, dict):
            for key in ("document_name", "original_filename", "source", "filename"):
                if metadata.get(key):
                    return str(metadata[key])
        name = getattr(source, "document_name", None)
        return str(name) if name else "knowledge base"


def build_rag_prompt(
    question: str,
    *,
    sources: Sequence[Any] = (),
    history: Iterable[Any] = (),
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[Message]:
    """Functional shortcut around :class:`PromptBuilder`."""
    return PromptBuilder(system_prompt=system_prompt).build(
        question, sources=sources, history=history
    )
