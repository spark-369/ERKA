"""Authentication endpoints: register, login, refresh, logout, me."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.api.rate_limit_dep import rate_limit_auth
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import (
    LoginRequest,
    RefreshRequest,
    TokenPair,
    UserCreate,
    UserRead,
)
from app.schemas.common import MessageResponse
from app.services.auth_service import AuthService

router = APIRouter()


@router.post(
    "/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account",
    dependencies=[Depends(rate_limit_auth)],
)
def register(
    payload: UserCreate, session: Annotated[Session, Depends(get_db)]
) -> UserRead:
    """Register a new user account."""
    user = AuthService(session).register(payload)
    return UserRead.model_validate(user)


@router.post(
    "/login",
    response_model=TokenPair,
    summary="Exchange credentials for JWT tokens",
    dependencies=[Depends(rate_limit_auth)],
)
def login(payload: LoginRequest, session: Annotated[Session, Depends(get_db)]) -> TokenPair:
    """Authenticate and return an access/refresh token pair."""
    return AuthService(session).login(payload)


@router.post(
    "/refresh",
    response_model=TokenPair,
    summary="Rotate a refresh token",
    dependencies=[Depends(rate_limit_auth)],
)
def refresh(
    payload: RefreshRequest, session: Annotated[Session, Depends(get_db)]
) -> TokenPair:
    """Exchange a valid refresh token for a new token pair."""
    return AuthService(session).refresh(payload.refresh_token)


@router.get("/me", response_model=UserRead, summary="Current user profile")
def me(current_user: Annotated[User, Depends(get_current_user)]) -> UserRead:
    """Return the authenticated user's profile."""
    return UserRead.model_validate(current_user)


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Client-side logout",
)
def logout(
    _current_user: Annotated[User, Depends(get_current_user)],
) -> MessageResponse:
    """Stateless logout.

    JWTs are not revocable without a denylist; the client simply discards both
    tokens. The endpoint exists so the frontend has one place to call.
    """
    return MessageResponse(detail="Signed out. Discard your local tokens.")
