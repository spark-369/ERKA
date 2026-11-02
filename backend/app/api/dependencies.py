"""Shared FastAPI dependencies (auth, sessions, rate limiting, pagination)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.exceptions import AuthenticationError, PermissionDeniedError
from app.core.security import decode_token
from app.db.session import get_db
from app.models.user import User
from app.repositories.user import UserRepository
from app.schemas.common import PaginationParams

# ``auto_error=False`` lets us raise our own uniform error envelope.
bearer_scheme = HTTPBearer(auto_error=False, description="JWT access token")


def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    session: Annotated[Session, Depends(get_db)],
) -> User:
    """Resolve the authenticated user from the ``Authorization`` header."""
    if credentials is None or not credentials.credentials:
        raise AuthenticationError(
            "Missing bearer token. Sign in and retry.", code="missing_token"
        )

    claims = decode_token(credentials.credentials, expected_type="access")
    user_id = str(claims.get("sub") or "")
    if not user_id:
        raise AuthenticationError("Token has no subject.", code="invalid_token")

    user = UserRepository(session).get(user_id)
    if user is None:
        raise AuthenticationError("Account no longer exists.", code="unknown_subject")
    if not user.is_active:
        raise PermissionDeniedError(
            "This account has been deactivated.", code="account_inactive"
        )

    # Make the identity available to logging / rate limiting.
    request.state.user_id = user.id
    request.state.user_email = user.email
    return user


def get_optional_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    session: Annotated[Session, Depends(get_db)],
) -> User | None:
    """Like :func:`get_current_user` but tolerates anonymous callers."""
    if credentials is None or not credentials.credentials:
        return None
    try:
        return get_current_user(request, credentials, session)
    except (AuthenticationError, PermissionDeniedError):
        return None


def require_admin(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Guard admin-only endpoints."""
    if not current_user.is_admin:
        raise PermissionDeniedError(
            "Administrator privileges are required.", code="admin_required"
        )
    return current_user


def pagination_params(
    page: Annotated[int, Query(ge=1, le=10_000, description="1-based page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
) -> PaginationParams:
    """Validated pagination query parameters."""
    return PaginationParams(page=page, page_size=page_size)
