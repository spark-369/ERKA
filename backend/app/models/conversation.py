"""Chat history models: conversations and their messages."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    new_uuid,
    utcnow,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models.user import User

MESSAGE_ROLES = ("system", "user", "assistant", "tool")


class Conversation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A threaded chat session between a user and the RAG assistant."""

    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_owner_updated", "owner_id", "updated_at"),
        {"comment": "RAG chat sessions"},
    )

    owner_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(255), default="New conversation", nullable=False)
    # Stored as an integer flag (0/1) rather than a boolean so the column is
    # identical across every PostgreSQL deployment and easy to filter on.
    is_archived: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Free-form per-conversation RAG overrides (top_k, model, document ids...).
    settings_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_message_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    owner: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Conversation id={self.id} title={self.title!r}>"


class Message(UUIDPrimaryKeyMixin, Base):
    """One turn in a conversation.

    ``sources`` stores the JSON-serialised retrieved chunks that grounded an
    assistant answer, which is what the UI renders as citations.
    """

    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
        {"comment": "Individual chat turns"},
    )

    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True, default=utcnow
    )

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Message id={self.id} role={self.role} len={len(self.content or '')}>"


def make_message(
    conversation_id: str,
    role: str,
    content: str,
    *,
    sources: list[dict[str, Any]] | None = None,
    token_count: int = 0,
    latency_ms: int | None = None,
    model_name: str | None = None,
) -> Message:
    """Factory that JSON-encodes ``sources`` for storage."""
    import json

    if role not in MESSAGE_ROLES:
        raise ValueError(f"Unsupported message role: {role!r}")
    return Message(
        id=new_uuid(),
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources=json.dumps(sources or []) if sources else None,
        token_count=token_count,
        latency_ms=latency_ms,
        model_name=model_name,
    )
