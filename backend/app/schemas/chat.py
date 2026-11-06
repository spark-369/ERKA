"""Chat / RAG conversation schemas.

The SSE contract is defined by :class:`ChatStreamEvent`: every event the backend
pushes down the stream is one JSON object with a ``type`` discriminator so the
Next.js client can switch on it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.config import LLM_MAX_GENERATION_TOKENS
from app.schemas.common import ORMModel

StreamEventType = Literal[
    "start", "sources", "token", "done", "error", "heartbeat",
]


class SourceReference(BaseModel):
    """A retrieved chunk cited in an answer."""

    chunk_id: str
    document_id: str
    document_name: str | None = None
    score: float = Field(ge=0.0, le=1.0)
    snippet: str = Field(description="Truncated chunk text for the citation card.")
    chunk_index: int | None = None
    #: True when the model's answer actually relied on this source.
    cited: bool = False


class MessageRead(ORMModel):
    """A stored conversation turn."""

    id: str
    conversation_id: str
    role: str
    content: str
    sources: str | None = None
    token_count: int
    latency_ms: int | None = None
    model_name: str | None = None
    created_at: datetime


class ConversationCreate(BaseModel):
    """Create a new chat session."""

    title: str | None = Field(default=None, max_length=255)
    document_ids: list[str] | None = Field(
        default=None, description="Restrict retrieval to these documents."
    )


class ConversationUpdate(BaseModel):
    """Rename / archive a session."""

    title: str | None = Field(default=None, max_length=255)
    is_archived: bool | None = None


class ConversationRead(ORMModel):
    """Conversation header."""

    id: str
    owner_id: str
    title: str
    is_archived: int
    message_count: int
    last_message_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationRead):
    """Conversation plus its full message history."""

    messages: list[MessageRead] = Field(default_factory=list)


class ChatCompletionRequest(BaseModel):
    """A RAG query.

    ``conversation_id`` continues an existing thread; when omitted a new
    conversation is created so the client can keep history.
    """

    question: str = Field(min_length=1, max_length=8_000)
    conversation_id: str | None = None
    document_ids: list[str] | None = Field(
        default=None, description="Optional document scope for retrieval."
    )
    top_k: int | None = Field(default=None, ge=1, le=20)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    #: Upper bound mirrors the model's maximum generation length (see
    #: ``app.core.config.LLM_MAX_GENERATION_TOKENS``). Asking for more than this
    #: is rejected here rather than silently producing degenerate output.
    max_new_tokens: int | None = Field(
        default=None,
        ge=16,
        le=LLM_MAX_GENERATION_TOKENS,
        description=(
            f"Maximum tokens to generate (defaults to {LLM_MAX_GENERATION_TOKENS})."
        ),
    )
    stream: bool = Field(default=False, description="Ignored by SSE endpoints.")
    use_history: bool = Field(default=True, description="Include prior turns in the prompt.")


class ChatCompletionResponse(BaseModel):
    """Non-streaming RAG answer."""

    conversation_id: str
    message_id: str
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    #: 0-based indices into ``sources`` that the answer cited.
    citations: list[int] = Field(default_factory=list)
    model_name: str
    latency_ms: int
    token_count: int = 0
    created_at: datetime


class ChatStreamEvent(BaseModel):
    """One SSE frame payload."""

    type: StreamEventType
    data: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def start(cls, conversation_id: str, model_name: str) -> "ChatStreamEvent":
        return cls(
            type="start",
            data={"conversation_id": conversation_id, "model_name": model_name},
        )

    @classmethod
    def sources(cls, references: list[SourceReference]) -> "ChatStreamEvent":
        return cls(
            type="sources",
            data={"sources": [ref.model_dump() for ref in references]},
        )

    @classmethod
    def token(cls, text: str, index: int) -> "ChatStreamEvent":
        return cls(type="token", data={"text": text, "index": index})

    @classmethod
    def done(
        cls,
        *,
        message_id: str,
        conversation_id: str,
        answer: str,
        latency_ms: int,
        token_count: int = 0,
        citations: list[int] | None = None,
    ) -> "ChatStreamEvent":
        return cls(
            type="done",
            data={
                "message_id": message_id,
                "conversation_id": conversation_id,
                "answer": answer,
                "latency_ms": latency_ms,
                "token_count": token_count,
                "citations": citations or [],
            },
        )

    @classmethod
    def error(cls, message: str, code: str = "stream_error") -> "ChatStreamEvent":
        return cls(type="error", data={"message": message, "code": code})

    @classmethod
    def heartbeat(cls) -> "ChatStreamEvent":
        return cls(type="heartbeat", data={})
