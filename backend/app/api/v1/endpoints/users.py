"""User endpoints: profile updates and account statistics."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import UserRead, UserUpdate
from app.schemas.common import MessageResponse
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService

router = APIRouter()


@router.patch(
    "/me",
    response_model=UserRead,
    summary="Update the current profile",
)
def update_profile(
    payload: UserUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> UserRead:
    """Change the display name and/or password."""
    user = AuthService(session).update_profile(current_user, payload)
    return UserRead.model_validate(user)


@router.get(
    "/me/stats",
    summary="Knowledge base statistics for the dashboard",
)
def my_stats(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    """Document counts plus conversation totals for the current user."""
    documents = DocumentService(session)
    chat = ChatService(session)
    _, conversation_total = chat.list_conversations(current_user.id, page_size=1)
    return {
        "documents": documents.stats(current_user.id),
        "conversations": conversation_total,
        "role": current_user.role,
    }


@router.delete(
    "/me",
    response_model=MessageResponse,
    summary="Deactivate the current account",
)
def deactivate_me(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    """Soft-delete the account (data is retained, access is revoked)."""
    AuthService(session).deactivate(current_user)
    return MessageResponse(detail="Account deactivated.")
