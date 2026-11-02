"""Conversation + message repositories (chat history persistence)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import delete, func, select

from app.models.conversation import Conversation, Message
from app.repositories.base import BaseRepository


class ConversationRepository(BaseRepository[Conversation]):
    """Queries scoped to the ``conversations`` table."""

    model = Conversation

    def create(
        self, *, owner_id: str, title: str | None = None, settings_json: str | None = None
    ) -> Conversation:
        conversation = Conversation(
            owner_id=owner_id,
            title=(title or "New conversation")[:255],
            settings_json=settings_json,
        )
        return self.add(conversation)

    def get_for_owner(self, conversation_id: str, owner_id: str) -> Conversation | None:
        statement = select(Conversation).where(
            Conversation.id == conversation_id, Conversation.owner_id == owner_id
        )
        return self.first(statement)

    def list_for_owner(
        self,
        owner_id: str,
        *,
        offset: int = 0,
        limit: int = 20,
        include_archived: bool = False,
    ) -> list[Conversation]:
        statement = select(Conversation).where(Conversation.owner_id == owner_id)
        if not include_archived:
            statement = statement.where(Conversation.is_archived == 0)
        statement = statement.order_by(
            func.coalesce(Conversation.last_message_at, Conversation.created_at).desc()
        )
        return self.list(statement, offset=offset, limit=limit)

    def count_for_owner(self, owner_id: str, *, include_archived: bool = False) -> int:
        statement = select(Conversation).where(Conversation.owner_id == owner_id)
        if not include_archived:
            statement = statement.where(Conversation.is_archived == 0)
        return self.count(statement)

    def touch(self, conversation: Conversation, *, title: str | None = None) -> Conversation:
        """Update recency metadata after a new turn."""
        conversation.last_message_at = datetime.now(timezone.utc)
        if title:
            conversation.title = title[:255]
        self.session.flush()
        return conversation


class MessageRepository(BaseRepository[Message]):
    """Queries scoped to the ``messages`` table."""

    model = Message

    def add_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        sources: str | None = None,
        token_count: int = 0,
        latency_ms: int | None = None,
        model_name: str | None = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources=sources,
            token_count=token_count,
            latency_ms=latency_ms,
            model_name=model_name,
        )
        self.add(message)

        # Keep the denormalised counter on the conversation row in sync.
        conversation = self.session.get(Conversation, conversation_id)
        if conversation is not None:
            conversation.message_count = int(conversation.message_count or 0) + 1
            conversation.last_message_at = message.created_at or datetime.now(timezone.utc)
        self.session.flush()
        return message

    def list_for_conversation(
        self, conversation_id: str, *, limit: int = 200, ascending: bool = True
    ) -> list[Message]:
        order = Message.created_at.asc() if ascending else Message.created_at.desc()
        statement = (
            select(Message).where(Message.conversation_id == conversation_id).order_by(order)
        )
        return self.list(statement, limit=limit)

    def recent_for_conversation(self, conversation_id: str, *, limit: int = 10) -> list[Message]:
        """Newest ``limit`` turns, returned oldest-first for prompt building."""
        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        rows = list(self.session.execute(statement).scalars().all())
        return list(reversed(rows))

    def count_for_conversation(self, conversation_id: str) -> int:
        statement = select(Message).where(Message.conversation_id == conversation_id)
        return self.count(statement)

    def delete_for_conversation(self, conversation_id: str) -> None:
        self.session.execute(delete(Message).where(Message.conversation_id == conversation_id))
        self.session.flush()
