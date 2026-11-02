"""Text splitting (chunking) with LangChain's recursive splitter.

Chunking is intentionally isolated so the same splitter is used by the API, the
Celery worker and the tests. ``RecursiveCharacterTextSplitter`` respects
paragraph/sentence boundaries before falling back to hard character cuts, which
matters a lot for retrieval quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from app.core.config import Settings, get_settings
from app.core.exceptions import DocumentProcessingError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Ordered separators: prefer semantic boundaries, degrade gracefully.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def count_tokens(text: str) -> int:
    """Token estimate that does not require a tokenizer (~4 chars per token)."""
    if not text:
        return 0
    return max(len(text) // 4, 1)


@dataclass(slots=True)
class TextChunk:
    """A chunk of text plus its position and provenance metadata."""

    content: str
    chunk_index: int
    char_count: int
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


class TextChunker:
    """Wraps LangChain's recursive splitter with our own chunk envelope."""

    def __init__(
        self,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: Sequence[str] | None = None,
        config: Settings | None = None,
    ) -> None:
        self._config = config or get_settings()
        self.chunk_size = chunk_size or self._config.chunk_size
        self.chunk_overlap = chunk_overlap or self._config.chunk_overlap
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.separators = list(separators or DEFAULT_SEPARATORS)
        self._splitter = self._build_splitter()

    def _build_splitter(self) -> Any:  # noqa: ANN401 - langchain splitter
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
                length_function=len,
                keep_separator=True,
            )
        except ImportError:  # pragma: no cover - fallback keeps the app usable
            logger.warning(
                "langchain-text-splitters not installed; using built-in splitter."
            )
            return _SimpleSplitter(self.chunk_size, self.chunk_overlap, self.separators)

    def split_text(
        self, text: str, *, metadata: dict[str, Any] | None = None
    ) -> list[TextChunk]:
        """Split ``text`` into ordered, metadata-carrying chunks."""
        if text is None or not text.strip():
            return []

        try:
            raw_chunks = self._splitter.split_text(text)
        except Exception as exc:
            raise DocumentProcessingError(
                f"Chunking failed: {exc}", code="chunk_failed"
            ) from exc

        base_metadata = dict(metadata or {})
        chunks: list[TextChunk] = []
        for index, raw in enumerate(raw_chunks):
            content = raw.strip()
            if not content:
                continue
            chunks.append(
                TextChunk(
                    content=content,
                    chunk_index=index,
                    char_count=len(content),
                    token_count=count_tokens(content),
                    metadata={**base_metadata, "chunk_index": index},
                )
            )
        return chunks

    def split_documents(self, documents: Iterable[Any]) -> list[TextChunk]:
        """Split several loaded documents, preserving their metadata."""
        chunks: list[TextChunk] = []
        for document in documents:
            text = getattr(document, "text", None) or getattr(document, "page_content", "")
            metadata = dict(getattr(document, "metadata", {}) or {})
            source = getattr(document, "source", None)
            if source:
                metadata.setdefault("source", source)
            chunks.extend(self.split_text(text, metadata=metadata))
        return chunks


class _SimpleSplitter:
    """Dependency-free fallback with the same semantics as the recursive one."""

    def __init__(self, chunk_size: int, chunk_overlap: int, separators: list[str]) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split_text(self, text: str) -> list[str]:
        units = [text]
        for separator in self.separators:
            if separator == "":
                break
            expanded: list[str] = []
            for unit in units:
                if len(unit) <= self.chunk_size:
                    expanded.append(unit)
                    continue
                parts = unit.split(separator)
                expanded.extend(
                    part if part.endswith(separator) else f"{part}{separator}"
                    for part in parts[:-1]
                )
                if parts[-1]:
                    expanded.append(parts[-1])
            units = expanded
            if all(len(unit) <= self.chunk_size for unit in units):
                break

        chunks: list[str] = []
        buffer = ""
        for unit in units:
            if len(buffer) + len(unit) <= self.chunk_size:
                buffer += unit
                continue
            if buffer.strip():
                chunks.append(buffer.strip())
            tail = buffer[-self.chunk_overlap :] if self.chunk_overlap else ""
            buffer = f"{tail}{unit}"
            while len(buffer) > self.chunk_size:
                chunks.append(buffer[: self.chunk_size].strip())
                buffer = buffer[self.chunk_size - self.chunk_overlap :]
        if buffer.strip():
            chunks.append(buffer.strip())
        return chunks


def chunk_documents(
    documents: Iterable[Any],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    config: Settings | None = None,
) -> list[TextChunk]:
    """Functional shortcut around :class:`TextChunker`."""
    chunker = TextChunker(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, config=config
    )
    return chunker.split_documents(documents)
