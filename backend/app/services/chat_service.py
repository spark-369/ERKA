"""Chat service: conversations, RAG answers and SSE streaming.

This is where the RAG chain meets persistence: the user's turn and the
assistant's turn are both written to PostgreSQL, and citations are stored as
JSON so the UI can render them later.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.models.conversation import Conversation, Message
from app.rag.chain import RAGChain
from app.rag.retriever import RetrievedChunk
from app.repositories.conversation import ConversationRepository, MessageRepository
from app.repositories.document import DocumentRepository
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatStreamEvent,
    SourceReference,
)

logger = get_logger(__name__)

SNIPPET_LENGTH = 300
TITLE_LENGTH = 60


def _chunk_text(text: str, *, size: int = 24) -> Iterator[str]:
    """Split ``text`` into progressive chunks without altering it.

    Structured output is produced in one shot (no token stream), so it is sliced
    here purely so the UI still types itself out. ``"".join(_chunk_text(t)) == t``
    always holds, which keeps the SSE contract intact for clients that
    reconstruct the answer from the ``token`` events.
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


class ChatService:
    """Conversation history + RAG orchestration for a single request."""

    def __init__(
        self,
        session: Session,
        *,
        chain: RAGChain | None = None,
        config: Settings | None = None,
    ) -> None:
        self._session = session
        self._config = config or get_settings()
        self._conversations = ConversationRepository(session)
        self._messages = MessageRepository(session)
        self._documents = DocumentRepository(session)
        self._chain = chain

    @property
    def chain(self) -> RAGChain:
        if self._chain is None:
            self._chain = RAGChain(config=self._config)
        return self._chain

    # ----------------------------------------------------------- conversations
    def create_conversation(
        self, owner_id: str, *, title: str | None = None, document_ids: list[str] | None = None
    ) -> Conversation:
        settings_json = (
            json.dumps({"document_ids": document_ids}) if document_ids else None
        )
        conversation = self._conversations.create(
            owner_id=owner_id,
            title=title or "New conversation",
            settings_json=settings_json,
        )
        self._session.commit()
        self._session.refresh(conversation)
        return conversation

    def get_conversation(self, conversation_id: str, owner_id: str) -> Conversation:
        conversation = self._conversations.get_for_owner(conversation_id, owner_id)
        if conversation is None:
            raise NotFoundError("Conversation not found.", code="conversation_not_found")
        return conversation

    def list_conversations(
        self, owner_id: str, *, page: int = 1, page_size: int = 20, include_archived: bool = False
    ) -> tuple[list[Conversation], int]:
        offset = (page - 1) * page_size
        conversations = self._conversations.list_for_owner(
            owner_id, offset=offset, limit=page_size, include_archived=include_archived
        )
        total = self._conversations.count_for_owner(
            owner_id, include_archived=include_archived
        )
        return conversations, total

    def list_messages(self, conversation_id: str, owner_id: str, *, limit: int = 200):
        self.get_conversation(conversation_id, owner_id)
        return self._messages.list_for_conversation(conversation_id, limit=limit)

    def rename_conversation(
        self, conversation_id: str, owner_id: str, *, title: str | None, archived: bool | None
    ) -> Conversation:
        conversation = self.get_conversation(conversation_id, owner_id)
        if title is not None:
            conversation.title = title[:255]
        if archived is not None:
            conversation.is_archived = 1 if archived else 0
        self._session.commit()
        self._session.refresh(conversation)
        return conversation

    def delete_conversation(self, conversation_id: str, owner_id: str) -> None:
        conversation = self.get_conversation(conversation_id, owner_id)
        self._conversations.delete(conversation)
        self._session.commit()

    # ----------------------------------------------------------------- helpers
    def resolve_conversation(
        self, request: ChatCompletionRequest, owner_id: str
    ) -> Conversation:
        """Continue the given conversation or start a new one.

        When a new conversation is started from a scoped request, the scope is
        persisted so later turns inherit it (see :meth:`_scoped_document_ids`).
        """
        if request.conversation_id:
            return self.get_conversation(request.conversation_id, owner_id)
        return self.create_conversation(
            owner_id,
            title=self._derive_title(request.question),
            document_ids=request.document_ids,
        )

    @staticmethod
    def _derive_title(question: str) -> str:
        clean = " ".join(question.split())
        if len(clean) <= TITLE_LENGTH:
            return clean or "New conversation"
        return clean[:TITLE_LENGTH].rsplit(" ", 1)[0] + "..."

    @staticmethod
    def _conversation_document_ids(conversation: Conversation) -> list[str] | None:
        """Read the document scope stored on a conversation (``settings_json``)."""
        if not conversation.settings_json:
            return None
        try:
            settings = json.loads(conversation.settings_json)
        except (TypeError, ValueError):
            return None
        document_ids = settings.get("document_ids") if isinstance(settings, dict) else None
        return list(document_ids) if document_ids else None

    def _scoped_document_ids(
        self, request: ChatCompletionRequest, conversation: Conversation
    ) -> list[str] | None:
        """Resolve the retrieval scope: explicit request scope wins, then the
        conversation's persisted scope, otherwise ``None`` (search everything).

        Whatever wins is narrowed to documents the owner still has. A
        conversation persists the scope it was created with, so once those
        documents are deleted and re-uploaded under new ids the stored scope
        refers to nothing - which would silently turn every later question into
        "no sources". When nothing in the scope survives, fall back to ``None``
        (search the whole knowledge base) instead of an empty filter.
        """
        requested = request.document_ids or self._conversation_document_ids(conversation)
        if not requested:
            return None

        existing = self._documents.filter_existing_ids(conversation.owner_id, requested)
        if not existing:
            logger.info(
                "Conversation %s scope referenced no existing documents; "
                "searching the whole knowledge base instead.",
                conversation.id,
            )
            return None
        return existing

    def _history(self, conversation_id: str, *, enabled: bool) -> list[Message]:
        if not enabled:
            return []
        return self._messages.recent_for_conversation(conversation_id, limit=6)

    @staticmethod
    def to_source_references(
        chunks: list[RetrievedChunk], *, cited_indices: Sequence[int] | None = None
    ) -> list[SourceReference]:
        """Convert retrieved chunks into API citation objects.

        ``cited_indices`` are 1-based numbers as they appear in the prompt's
        bracketed headers (``[1]`` -> index 1). Sources whose number is in that
        set are flagged ``cited=True`` so the UI can distinguish "retrieved" from
        "actually used by the model".
        """
        cited = set(cited_indices or ())
        return [
            SourceReference(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                score=round(float(chunk.score), 6),
                snippet=chunk.snippet(SNIPPET_LENGTH),
                chunk_index=chunk.chunk_index,
                cited=(position in cited),
            )
            for position, chunk in enumerate(chunks, start=1)
        ]

    def persist_user_message(self, conversation_id: str, question: str) -> Message:
        """Store the user's turn before generation starts."""
        message = self._messages.add_message(
            conversation_id=conversation_id, role="user", content=question
        )
        self._session.commit()
        self._session.refresh(message)
        return message

    def persist_assistant_message(
        self,
        conversation_id: str,
        *,
        answer: str,
        sources: list[RetrievedChunk],
        model_name: str,
        latency_ms: int,
        token_count: int = 0,
        cited_indices: Sequence[int] | None = None,
        key_points: Sequence[str] | None = None,
        structured: bool = False,
    ) -> Message:
        """Store the assistant's turn together with its structured extras.

        Citations are normalised through :meth:`to_source_references` so the rows
        written here are byte-for-byte the same shape the API returns to the
        client. The structured extras (``key_points`` + the ``structured`` flag)
        ride along in the same JSON envelope so history can re-render exactly
        like a fresh answer.
        """
        envelope = {
            "sources": [
                reference.model_dump(mode="json")
                for reference in self.to_source_references(
                    sources, cited_indices=cited_indices
                )
            ],
            "key_points": list(key_points or []),
            "structured": bool(structured),
        }
        message = self._messages.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            sources=json.dumps(envelope),
            token_count=token_count,
            latency_ms=latency_ms,
            model_name=model_name,
        )
        self._session.commit()
        self._session.refresh(message)
        return message

    # -------------------------------------------------------------------- RAG
    def complete(self, request: ChatCompletionRequest, owner_id: str) -> dict[str, Any]:
        """Non-streaming RAG turn (persists both sides of the exchange)."""
        conversation = self.resolve_conversation(request, owner_id)
        history = self._history(conversation.id, enabled=request.use_history)
        self.persist_user_message(conversation.id, request.question)

        result = self.chain.answer(
            request.question,
            user_id=owner_id,
            document_ids=self._scoped_document_ids(request, conversation),
            history=history,
            top_k=request.top_k,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens,
        )

        # Citations + key points come from the structured-output path (1-based
        # prompt indices for citations).
        cited_indices = list(result.metadata.get("citations") or [])
        key_points = list(result.metadata.get("key_points") or [])
        structured = bool(result.metadata.get("structured"))

        message = self.persist_assistant_message(
            conversation.id,
            answer=result.answer,
            sources=result.sources,
            model_name=result.model_name,
            latency_ms=result.latency_ms,
            token_count=result.completion_tokens,
            cited_indices=cited_indices,
            key_points=key_points,
            structured=structured,
        )
        return {
            "conversation_id": conversation.id,
            "message_id": message.id,
            "answer": result.answer,
            "key_points": key_points,
            "sources": self.to_source_references(
                result.sources, cited_indices=cited_indices
            ),
            "citations": self._citation_offsets(cited_indices, len(result.sources)),
            "structured": structured,
            "model_name": result.model_name,
            "latency_ms": result.latency_ms,
            "token_count": result.completion_tokens,
            "created_at": message.created_at,
        }

    @staticmethod
    def _citation_offsets(
        cited_indices: Sequence[int], source_count: int
    ) -> list[int]:
        """Map 1-based prompt citation numbers to 0-based ``sources`` offsets."""
        return sorted(
            {index - 1 for index in cited_indices if 1 <= index <= source_count}
        )

    def stream(self, request: ChatCompletionRequest, owner_id: str) -> Iterator[ChatStreamEvent]:
        """Streaming RAG turn as a sequence of typed SSE events.

        Event order: ``start`` -> ``sources`` -> ``token``* -> ``done``.
        Failures emit ``error`` (the HTTP status is already 200 at that point).
        """
        conversation = self.resolve_conversation(request, owner_id)
        history = self._history(conversation.id, enabled=request.use_history)
        self.persist_user_message(conversation.id, request.question)

        started = time.perf_counter()
        model_name = self.chain.llm.model_name

        try:
            sources = self.chain.retrieve(
                request.question,
                user_id=owner_id,
                document_ids=self._scoped_document_ids(request, conversation),
                top_k=request.top_k,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            logger.exception("Retrieval failed for conversation %s", conversation.id)
            yield ChatStreamEvent.error(f"Retrieval failed: {exc}", code="retrieval_error")
            return

        yield ChatStreamEvent.start(conversation.id, model_name)
        yield ChatStreamEvent.sources(self.to_source_references(sources))

        messages = self.chain.build_messages(
            request.question, sources=sources, history=history
        )

        # Prefer schema-constrained generation so the answer reads like prose
        # (not a dump of raw chunks). The structured payload is complete, so it
        # is chunked into ``token`` events purely for a progressive UI; when
        # structured output is unavailable we fall back to true token streaming.
        payload = self.chain.generate_structured(
            messages,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens,
        )

        if payload is not None:
            answer = payload.answer.strip()
            for index, chunk in enumerate(_chunk_text(answer)):
                yield ChatStreamEvent.token(chunk, index)
            cited_indices = list(payload.cited_indices())
            key_points = payload.points()
            structured = True
        else:
            parts: list[str] = []
            index = 0
            try:
                for delta in self.chain.llm.stream(
                    messages,
                    temperature=request.temperature,
                    max_new_tokens=request.max_new_tokens,
                ):
                    if not delta:
                        continue
                    parts.append(delta)
                    yield ChatStreamEvent.token(delta, index)
                    index += 1
            except Exception as exc:  # noqa: BLE001 - surfaced to the client
                logger.exception("Streaming failed for conversation %s", conversation.id)
                yield ChatStreamEvent.error(
                    f"Generation failed: {exc}", code="generation_error"
                )
                return
            answer = "".join(parts).strip()
            # Plain-text fallback: citations/key points are only reliable from
            # the structured path, so none are claimed here.
            cited_indices = []
            key_points = []
            structured = False

        latency_ms = int((time.perf_counter() - started) * 1000)
        token_count = self.chain.llm.count_tokens(answer)

        try:
            message = self.persist_assistant_message(
                conversation.id,
                answer=answer,
                sources=sources,
                model_name=model_name,
                latency_ms=latency_ms,
                token_count=token_count,
                cited_indices=cited_indices,
                key_points=key_points,
                structured=structured,
            )
        except Exception as exc:  # noqa: BLE001 - answer already streamed
            logger.exception("Failed to persist assistant message")
            yield ChatStreamEvent.error(
                f"Could not save the answer: {exc}", code="persist_error"
            )
            return

        yield ChatStreamEvent.done(
            message_id=message.id,
            conversation_id=conversation.id,
            answer=answer,
            latency_ms=latency_ms,
            token_count=token_count,
            citations=self._citation_offsets(cited_indices, len(sources)),
            key_points=key_points,
            structured=structured,
        )
