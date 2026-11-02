"""Structured output for the local Qwen2.5 model (constrained decoding).

The RAG chain wants a *grounded*, *well-formatted* answer plus the citations the
model actually used. Asking a 0.5B model for free-form text and then cleaning it
up afterwards is brittle; instead we constrain the decoder so the only thing it
can emit is a small JSON object matching :class:`AnswerWithCitations`.

``outlines`` turns a Pydantic model (or JSON schema) into a finite-state machine
and masks the logits at every step, which guarantees valid, structured output
even from a very small model. The provider wires that up in
:meth:`~app.llm.transformers_llm.TransformersLLMProvider.generate_structured`.

Because the *shape* is guaranteed by the decoder, the frontend renders real
structure (a summary paragraph, bullet points, citation chips) instead of trying
to re-parse prose. :func:`parse_answer_payload` is a tolerant, regex-free
fallback used only when constrained decoding is unavailable.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field


class AnswerWithCitations(BaseModel):
    """The structured shape the model is constrained to produce.

    Splitting the answer into ``answer`` + ``key_points`` (rather than one blob
    of text) is what lets the UI render a clean, ChatGPT-style response without
    any text parsing.
    """

    answer: str = Field(
        description=(
            "A thorough, self-contained answer to the user's question, grounded "
            "in the context. Open with a direct answer, then explain the relevant "
            "details and specifics. Write flowing prose (a few paragraphs), not a "
            "list of raw context, and do not include bracketed citation numbers."
        ),
    )
    key_points: list[str] = Field(
        default_factory=list,
        description=(
            "Supporting bullet points that expand on the answer with concrete "
            "facts, figures, steps or technologies from the sources. Include "
            "several when the context supports them."
        ),
    )
    citations: list[str] = Field(
        default_factory=list,
        description=(
            "Source numbers the answer relies on, from the bracketed context "
            "headers, e.g. ['1', '2']."
        ),
    )

    def cited_indices(self) -> list[int]:
        """Return ``citations`` as sorted, de-duplicated integers."""
        seen: list[int] = []
        for value in self.citations:
            digits = str(value).strip().strip("[]").strip()
            if digits.isdigit():
                index = int(digits)
                if index not in seen:
                    seen.append(index)
        return sorted(seen)

    def points(self) -> list[str]:
        """Return the cleaned, non-empty bullet points."""
        return [point.strip() for point in self.key_points if point and point.strip()]


def structured_json_schema() -> dict[str, Any]:
    """JSON schema for :class:`AnswerWithCitations` (used by outlines/others)."""
    return AnswerWithCitations.model_json_schema()


def parse_answer_payload(raw: str) -> AnswerWithCitations | None:
    """Best-effort, regex-free parse of ``raw`` into :class:`AnswerWithCitations`.

    Returns ``None`` when ``raw`` carries no recognisable answer, so the caller
    can fall back to treating it as plain text. Note that a JSON object which
    simply lacks an ``answer`` field is *not* prose - it is reported as ``None``
    rather than being echoed back verbatim.
    """
    if not raw or not raw.strip():
        return None

    saw_json = False
    for candidate in _json_candidates(raw):
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        saw_json = True
        parsed = _from_payload(payload)
        if parsed is not None:
            return parsed

    # Valid JSON was present but none of it carried a usable answer, so this is
    # not free-form prose: do not echo the raw JSON back as the answer.
    if saw_json:
        return None

    text = raw.strip()
    if text:
        return AnswerWithCitations(answer=text, citations=[])
    return None


def _json_candidates(raw: str) -> list[str]:
    """Ordered substrings worth trying to JSON-decode (no regex)."""
    candidates: list[str] = []

    # Content inside a ```json ... ``` fence.
    fence_start = raw.find("```")
    if fence_start != -1:
        content_start = raw.find("\n", fence_start)
        if content_start != -1:
            content_end = raw.find("```", content_start)
            if content_end != -1:
                candidates.append(raw[content_start:content_end].strip())

    # The outermost {...} span.
    brace_start = raw.find("{")
    brace_end = raw.rfind("}")
    if 0 <= brace_start < brace_end:
        candidates.append(raw[brace_start : brace_end + 1])

    candidates.append(raw.strip())

    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _from_payload(payload: Any) -> AnswerWithCitations | None:
    """Build an :class:`AnswerWithCitations` from a decoded JSON value."""
    if isinstance(payload, str):
        return AnswerWithCitations(answer=payload, citations=[])
    if not isinstance(payload, dict):
        return None

    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return None

    raw_points = payload.get("key_points")
    points = [str(item) for item in raw_points] if isinstance(raw_points, list) else []

    raw_citations = payload.get("citations")
    citations: list[str] = []
    if isinstance(raw_citations, list):
        citations = [str(item) for item in raw_citations]
    elif isinstance(raw_citations, str) and raw_citations.strip():
        citations = [part.strip() for part in raw_citations.split(",") if part.strip()]

    return AnswerWithCitations(
        answer=answer.strip(), key_points=points, citations=citations
    )
