"""Chat endpoints: RAG completion, SSE streaming and conversation history."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, pagination_params
from app.api.rate_limit_dep import rate_limit_chat
from app.core.logging import get_logger
from app.db.session import get_db
from app.models.user import User
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatStreamEvent,
    ConversationCreate,
    ConversationDetail,
    ConversationRead,
    ConversationUpdate,
    MessageRead,
)
from app.schemas.common import MessageResponse, Page, PaginationParams
from app.services.chat_service import ChatService

logger = get_logger(__name__)

router = APIRouter()

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    # Disable proxy buffering (nginx) so tokens arrive immediately.
    "X-Accel-Buffering": "no",
}


def _sse_frame(event: ChatStreamEvent) -> str:
    """Serialise one event as a named SSE frame."""
    payload = json.dumps(event.data, default=str)
    return f"event: {event.type}\ndata: {payload}\n\n"


async def _stream_body(payload: ChatCompletionRequest, user_id: str) -> AsyncIterator[str]:
    """Async generator bridging the blocking chat service to SSE frames.

    Generation is CPU-bound and synchronous (as is the SQLAlchemy session), so
    the whole service generator runs on a dedicated worker thread and its frames
    are marshalled back to the event loop through an :class:`asyncio.Queue`.
    That keeps ``await``-based streaming without ever blocking the event loop,
    and it keeps every database call on a single thread (SQLAlchemy sessions are
    not thread-safe).

    The stream owns its own database session: FastAPI tears down request-scoped
    ``yield`` dependencies *before* a ``StreamingResponse`` body is consumed, so
    relying on the request session here would be fragile.
    """
    from app.db.session import get_session_factory

    session = get_session_factory()()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[object] = asyncio.Queue()
    _SENTINEL = object()

    def _produce() -> None:
        def _emit(item: object) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:
                # The event loop closed (client disconnected); nothing to do.
                pass

        try:
            service = ChatService(session)
            for event in service.stream(payload, user_id):
                _emit(event)
        except Exception:  # noqa: BLE001 - never leak a broken stream
            logger.exception("Unhandled error while streaming a chat response")
            _emit(
                ChatStreamEvent.error(
                    "The stream ended unexpectedly.", code="stream_error"
                )
            )
        finally:
            _emit(_SENTINEL)

    worker = threading.Thread(target=_produce, name="sse-producer", daemon=True)
    worker.start()
    try:
        while True:
            event = await queue.get()
            if event is _SENTINEL:
                break
            yield _sse_frame(event)  # type: ignore[arg-type]
    finally:
        # Runs on normal completion, client disconnect (task cancellation) and
        # generator close: always release the session and let the worker wind
        # down. The worker is a daemon thread, so a slow drain cannot hang exit.
        session.close()
        worker.join(timeout=5.0)


@router.post(
    "/completions",
    response_model=ChatCompletionResponse,
    summary="Ask a question (non-streaming)",
    dependencies=[Depends(rate_limit_chat)],
)
def complete(
    payload: ChatCompletionRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> ChatCompletionResponse:
    """Retrieve context and generate a complete answer in one response."""
    result = ChatService(session).complete(payload, current_user.id)
    return ChatCompletionResponse(**result)


@router.post(
    "/stream",
    summary="Ask a question and stream the answer over SSE",
    dependencies=[Depends(rate_limit_chat)],
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "text/event-stream of ChatStreamEvent frames",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
)
def stream(
    payload: ChatCompletionRequest,
    current_user: Annotated[User, Depends(get_current_user)],
) -> StreamingResponse:
    """Server-Sent Events endpoint consumed by the Next.js chat UI.

    Frame sequence: ``start`` -> ``sources`` -> ``token``* -> ``done``
    (or a terminal ``error`` frame).

    The response body owns its own database session, so no request-scoped
    session is injected here.
    """
    return StreamingResponse(
        _stream_body(payload, current_user.id),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post(
    "/stream/{conversation_id}",
    summary="Continue a conversation and stream the answer over SSE",
    dependencies=[Depends(rate_limit_chat)],
    response_class=StreamingResponse,
)
def stream_in_conversation(
    conversation_id: str,
    payload: ChatCompletionRequest,
    current_user: Annotated[User, Depends(get_current_user)],
) -> StreamingResponse:
    """Convenience route so clients can keep the thread id in the URL."""
    payload.conversation_id = conversation_id
    return StreamingResponse(
        _stream_body(payload, current_user.id),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


# --------------------------------------------------------------------------- #
# Conversation history
# --------------------------------------------------------------------------- #
@router.post(
    "/conversations",
    response_model=ConversationRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a conversation",
)
def create_conversation(
    payload: ConversationCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> ConversationRead:
    """Start an empty chat session (optionally scoped to documents)."""
    conversation = ChatService(session).create_conversation(
        current_user.id, title=payload.title, document_ids=payload.document_ids
    )
    return ConversationRead.model_validate(conversation)


@router.get(
    "/conversations",
    response_model=Page[ConversationRead],
    summary="List conversations",
)
def list_conversations(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
    pagination: Annotated[PaginationParams, Depends(pagination_params)],
    include_archived: Annotated[bool, Query()] = False,
) -> Page[ConversationRead]:
    """Paginated conversation listing for the sidebar."""
    conversations, total = ChatService(session).list_conversations(
        current_user.id,
        page=pagination.page,
        page_size=pagination.page_size,
        include_archived=include_archived,
    )
    return Page[ConversationRead](
        items=[ConversationRead.model_validate(item) for item in conversations],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
    summary="Fetch a conversation with its messages",
)
def get_conversation(
    conversation_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> ConversationDetail:
    """Full thread history, oldest first."""
    service = ChatService(session)
    conversation = service.get_conversation(conversation_id, current_user.id)
    messages = service.list_messages(conversation_id, current_user.id)
    return ConversationDetail(
        **ConversationRead.model_validate(conversation).model_dump(),
        messages=[MessageRead.model_validate(message) for message in messages],
    )


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=Page[MessageRead],
    summary="List the messages of a conversation",
)
def list_messages(
    conversation_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
    pagination: Annotated[PaginationParams, Depends(pagination_params)],
) -> Page[MessageRead]:
    """Messages of one conversation (ownership enforced)."""
    messages = ChatService(session).list_messages(conversation_id, current_user.id)
    return Page[MessageRead](
        items=[MessageRead.model_validate(message) for message in messages],
        total=len(messages),
        page=pagination.page,
        page_size=pagination.page_size,
    )


@router.patch(
    "/conversations/{conversation_id}",
    response_model=ConversationRead,
    summary="Rename or archive a conversation",
)
def update_conversation(
    conversation_id: str,
    payload: ConversationUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> ConversationRead:
    """Update the conversation title and/or archive flag."""
    conversation = ChatService(session).rename_conversation(
        conversation_id, current_user.id, title=payload.title, archived=payload.is_archived
    )
    return ConversationRead.model_validate(conversation)


@router.delete(
    "/conversations/{conversation_id}",
    response_model=MessageResponse,
    summary="Delete a conversation and its messages",
)
def delete_conversation(
    conversation_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    """Remove the thread and every message in it."""
    ChatService(session).delete_conversation(conversation_id, current_user.id)
    return MessageResponse(detail="Conversation deleted.")
