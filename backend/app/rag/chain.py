"""The RAG chain: retrieve context, build a prompt, generate an answer.

Kept framework-light on purpose: the LangChain pieces we rely on (splitters,
``Document`` objects) live in ``rag/`` and are used where they add value, while
the chain itself is a plain, testable function object.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.llm.base import LLMProvider, Message
from app.llm.factory import get_llm_provider
from app.llm.prompt import PromptBuilder
from app.rag.retriever import RetrievedChunk, Retriever, get_retriever

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Answer post-processing
#
# Small instruction-tuned models (Qwen2.5-0.5B) routinely ignore formatting
# instructions: they emit numbered/bulleted lists, repeat whole lines (a decoding
# loop), echo reference/bibliography lines out of the retrieved context, and get
# cut off mid-sentence at the token budget. Prompting alone cannot fix that
# reliably, so these deterministic transforms run *after* generation and
# guarantee the answer is one clean, complete paragraph.
# --------------------------------------------------------------------------- #
#: Leading list markers: "12. ", "3) ", "- ", "* ", "• ".
_LIST_MARKER = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•·])\s+")
#: Inline enumeration right after a list intro colon ("features: 1. X 2. Y"),
#: which survives line/sentence splitting and otherwise leaks into the prose.
_INLINE_ENUM = re.compile(r"(?<=:)\s*\d{1,2}\s*[.)]\s+")
#: Markdown bold (unwrapped to plain text) and stray inline emphasis marks.
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_INLINE_MARK = re.compile(r"[*_`]+")
#: A leading meta opener such as "Based on the provided information, ..." or
#: "... information: ...". The clause must end at a comma or colon so the regex
#: cannot run on into the real answer (e.g. across the "." in "99.9").
_LEADING_META = re.compile(
    r"^\s*based on (?:the )?(?:provided )?(?:information|context|knowledge base)"
    r"[^,:\n]*?[,:]\s*",
    re.IGNORECASE,
)
#: Trailing "Sources used: ..." / "References: ..." listing lines.
_SOURCES_LINE = re.compile(
    r"^\s*(?:sources?\s+used|references?|bibliography)\s*[:.\-].*$",
    re.IGNORECASE | re.MULTILINE,
)
#: Lines that are nothing but a bare URL.
_URL_LINE = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$", re.IGNORECASE)
#: Line-leading citation noise ("[1] Source: doc.pdf", "[2] ...").
_SOURCE_HEADER = re.compile(r"^\s*\[\d+\]\s*(?:source\s*:)?\s*", re.IGNORECASE)
#: Characters that end a complete sentence.
_TERMINAL = ".!?\"')”"
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

#: Words ignored when comparing sentences for repetition - they inflate the
#: overlap of genuinely different sentences without carrying meaning.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
        "with", "as", "at", "by", "from", "into", "is", "are", "was", "were",
        "be", "been", "being", "it", "its", "this", "that", "these", "those",
        "can", "will", "would", "should", "may", "might", "must", "has", "have",
        "had", "do", "does", "did", "not", "also", "more", "most", "very",
        "such", "like", "which", "who", "whose", "their", "there", "here",
        "they", "them", "then", "than", "each", "all", "any", "some", "other",
        "others", "so", "if", "when", "while", "because", "about", "over",
        "under", "between", "across", "through", "during", "including", "etc",
    }
)
#: Words are compared by their first few characters: word-form differences
#: ("integrate"/"integrates"/"integration", "platform"/"platforms") then collapse
#: to the same term, which is what lets a re-worded restatement be detected.
_TOKEN_PREFIX = 6


def _stem(word: str) -> str:
    """Fold a word to a comparison key (a fixed-length prefix).

    A crude, dependency-free normaliser: taking the first ``_TOKEN_PREFIX``
    characters makes inflected forms of the same word identical, which is all the
    near-duplicate check needs. It is never shown to the user.
    """
    return word[:_TOKEN_PREFIX]


def _dedupe_key(sentence: str) -> str:
    """Format-insensitive key so "1. **Next.js**:" and "next.js" collapse."""
    return re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()


def _content_tokens(sentence: str) -> frozenset[str]:
    """Meaningful, normalised words of ``sentence`` (stop-words/numbers out).

    Dropping digits is deliberate: a repeated line whose only difference is its
    leading list number ("1. X" vs "2. X") must collapse to the same token set.
    """
    words = re.findall(r"[a-z]+", sentence.lower())
    return frozenset(
        _stem(word) for word in words if len(word) > 2 and word not in _STOPWORDS
    )


def _is_near_duplicate(
    tokens: frozenset[str],
    kept: list[frozenset[str]],
    *,
    threshold: float = 0.6,
    min_shared: int = 5,
) -> bool:
    """True when ``tokens`` largely restates a sentence already in ``kept``.

    Uses the *overlap coefficient* (shared tokens divided by the smaller set's
    size): a re-worded restatement reuses most of the shorter sentence's content
    words, so it scores well above ``threshold``, while genuinely new sentences
    - and short asides that merely share a couple of words - do not. A
    ``min_shared`` floor of shared tokens prevents a short sentence from being
    dropped just because a longer one happens to contain a few of its words.
    """
    if not tokens:
        return True
    for other in kept:
        if not other:
            continue
        shared = tokens & other
        if len(shared) < min_shared:
            continue
        smaller = min(len(tokens), len(other))
        if smaller and len(shared) / smaller >= threshold:
            return True
    return False




def clean_answer(text: str) -> str:
    """Return ``text`` as a single clean, complete paragraph.

    * strips list/bullet numbering and Markdown emphasis,
    * removes a leading "Based on the information…" opener,
    * drops trailing "Sources used:" / reference listings and bare URLs,
    * de-duplicates repeated sentences (kills decoding loops),
    * trims a truncated trailing fragment to the last complete sentence,
    * joins everything into one flowing paragraph ending in punctuation.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SOURCES_LINE.sub("", text)
    text = _LEADING_META.sub("", text, count=1)
    text = _INLINE_ENUM.sub(" ", text)

    # Flatten each line into candidate sentence text.
    candidates: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        line = _SOURCE_HEADER.sub("", line).strip()
        line = _LIST_MARKER.sub("", line).strip()
        line = _BOLD.sub(r"\1", line)
        line = _INLINE_MARK.sub("", line).strip()
        if not line or _URL_LINE.match(line):
            continue
        candidates.append(line)

    # Split embedded sentences and drop repeats. Both exact repeats and
    # *near* repeats (re-worded restatements, the small-model failure mode)
    # are removed, keyed on meaningful words so numbering/word-endings do not
    # mask a duplicate.
    sentences: list[str] = []
    seen: set[str] = set()
    kept_tokens: list[frozenset[str]] = []
    for chunk in candidates:
        for sentence in _SENTENCE_SPLIT.split(chunk):
            sentence = " ".join(sentence.split()).strip()
            sentence = _LIST_MARKER.sub("", sentence).strip()
            if not sentence:
                continue
            key = _dedupe_key(sentence)
            tokens = _content_tokens(sentence)
            if not key or key in seen or _is_near_duplicate(tokens, kept_tokens):
                continue
            seen.add(key)
            kept_tokens.append(tokens)
            sentences.append(sentence)

    if not sentences:
        return ""

    # A model that hit the token budget leaves an unterminated final fragment;
    # drop it so the answer ends on a complete sentence.
    if len(sentences) >= 2 and sentences[-1][-1] not in _TERMINAL:
        sentences = sentences[:-1]

    paragraph = re.sub(r"\s+", " ", " ".join(sentences)).strip()
    if not paragraph:
        return ""
    if paragraph[-1] not in _TERMINAL:
        paragraph += "."
    return paragraph[0].upper() + paragraph[1:]


def clean_answer_stream(fragments: Iterable[str]) -> Iterator[str]:
    """Clean model output **incrementally** so it streams while it is produced.

    :func:`clean_answer` needs the whole completion, which defeats live
    streaming (nothing is emitted until generation finishes). This variant
    consumes the provider fragments as they arrive, cleans each finished
    sentence the moment it is complete, and yields it immediately - so the
    client sees the answer build up in real time.

    The transform is kept to the operations that can be applied sentence by
    sentence (list/markdown/citation stripping, lead-opener removal and
    duplicate suppression); a leading meta opener such as "Based on the ..."
    is dropped by staying silent until the first real sentence is known.
    """
    buffer = ""
    seen: set[str] = set()
    kept_tokens: list[frozenset[str]] = []
    head = True  # still at the very start of the answer
    emitted = False
    raw_all: list[str] = []
    produced = False
    #: Emit a partial even without a sentence break if the buffer grows this
    #: large, so a long unpunctuated run can never stall the stream.
    soft_limit = 240

    def _emit(sentence: str) -> str | None:
        nonlocal head, emitted
        sentence = _clean_line(sentence)
        if head:
            sentence = _LEADING_META.sub("", sentence).strip()
            head = False
        # Drop fragments with no words (e.g. a bare "1." left by the split).
        if not sentence or not any(character.isalpha() for character in sentence):
            return None
        # Drop an exact repeat, then a re-worded restatement of an earlier line.
        key = _dedupe_key(sentence)
        tokens = _content_tokens(sentence)
        if not key or key in seen or _is_near_duplicate(tokens, kept_tokens):
            return None
        seen.add(key)
        kept_tokens.append(tokens)
        if not emitted:
            sentence = sentence[0].upper() + sentence[1:]
            emitted = True
        return sentence if sentence[-1] in _TERMINAL else sentence + "."

    for fragment in fragments:
        if not fragment:
            continue
        raw_all.append(fragment)
        buffer += fragment
        # Emit every *complete* sentence now; keep the trailing partial.
        parts = _SENTENCE_SPLIT.split(buffer)
        for sentence in parts[:-1]:
            cleaned = _emit(sentence)
            if cleaned:
                produced = True
                yield cleaned + " "
        buffer = parts[-1]
        # Guard against a long run with no sentence terminator stalling output.
        if len(buffer) >= soft_limit:
            cleaned = _emit(buffer)
            if cleaned:
                produced = True
                yield cleaned + " "
            buffer = ""

    tail = _emit(buffer)
    if tail:
        produced = True
        yield tail

    # Safety net: if cleaning filtered everything out but the model did produce
    # text, surface the raw text rather than delivering an empty answer.
    if not produced:
        fallback = " ".join("".join(raw_all).split())
        if fallback:
            yield fallback


def _clean_line(block: str) -> str:
    """Strip list markers, citation noise and Markdown from one fragment.

    Operates line-by-line (like :func:`clean_answer`) so line-anchored noise
    (a ``[n] Source:`` header or a ``Sources used:`` listing) is removed even
    when the fragment spans several lines, then re-joins the survivors.
    """
    pieces: list[str] = []
    for raw_line in block.splitlines() or [block]:
        line = raw_line.strip()
        if not line or _SOURCES_LINE.match(line):
            continue
        line = _SOURCE_HEADER.sub("", line).strip()
        line = _LIST_MARKER.sub("", line).strip()
        line = _INLINE_ENUM.sub(" ", line)
        line = _BOLD.sub(r"\1", line)
        line = _INLINE_MARK.sub("", line).strip()
        if line and not _URL_LINE.match(line):
            pieces.append(line)
    return " ".join(pieces)


def _chunk_text(text: str, *, size: int = 24) -> Iterator[str]:
    """Split ``text`` into progressive chunks without altering it.

    ``"".join(_chunk_text(t)) == t`` always holds, which keeps the SSE contract
    intact for clients that reconstruct the answer from the ``token`` events.
    """
    if not text:
        return
    remaining = text
    while remaining:
        if len(remaining) <= size:
            yield remaining
            return
        cut = remaining.rfind(" ", 0, size)
        if cut <= 0:
            cut = size
        yield remaining[:cut]
        remaining = remaining[cut:]


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
        """Retrieve + generate in one call (non-streaming path)."""
        started = time.perf_counter()
        sources = self.retrieve(
            question, user_id=user_id, document_ids=document_ids, top_k=top_k
        )
        messages = self.build_messages(question, sources=sources, history=history)

        result = self.llm.generate(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        answer = clean_answer(result.text) or result.text.strip()

        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "RAG answer generated in %sms (sources=%s, tokens=%s)",
            latency_ms,
            len(sources),
            result.completion_tokens,
        )
        return RAGAnswer(
            answer=answer,
            sources=list(sources),
            model_name=result.model_name,
            latency_ms=latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            metadata=result.metadata,
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
        """Retrieve once, then generate the answer and yield it in chunks.

        Retrieval happens eagerly so the API can emit a ``sources`` SSE event
        before the answer arrives. Generation itself is a single synchronous
        call; the completion is sliced here only so the client can render it
        progressively.
        """
        sources = self.retrieve(
            question, user_id=user_id, document_ids=document_ids, top_k=top_k
        )
        messages = self.build_messages(question, sources=sources, history=history)
        result = self.llm.generate(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        yield from _chunk_text(clean_answer(result.text) or result.text.strip())

    def stream_tokens(
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
        """Retrieve once, then stream the answer as it is generated.

        Delegates to :meth:`LLMProvider.generate_stream` and cleans each finished
        sentence the moment it arrives (see :func:`clean_answer_stream`), so the
        first words reach the client in ~a second instead of after the whole
        completion. Retrieval still runs eagerly so the API can emit a ``sources``
        event before the first token.
        """
        sources = self.retrieve(
            question, user_id=user_id, document_ids=document_ids, top_k=top_k
        )
        messages = self.build_messages(question, sources=sources, history=history)
        yield from clean_answer_stream(
            self.llm.generate_stream(
                messages,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        )

    def build_stream_tokens(
        self,
        question: str,
        *,
        sources: Sequence[RetrievedChunk],
        history: Sequence[Any] = (),
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> Iterator[str]:
        """Like :meth:`stream_tokens` but reuses already-retrieved ``sources``.

        Lets the caller run retrieval itself (for its own error handling and SSE
        framing) without paying for a second retrieval. Generation is streamed
        and cleaned incrementally, so output appears live.
        """
        messages = self.build_messages(question, sources=sources, history=history)
        yield from clean_answer_stream(
            self.llm.generate_stream(
                messages,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        )
