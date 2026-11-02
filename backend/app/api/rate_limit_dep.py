"""Manual rate-limiting dependency.

Rate limiting is enforced in the API gateway layer, per endpoint scope:

* ``auth``   - login/register/refresh (brute-force protection)
* ``chat``   - RAG generation (the expensive path)
* ``upload`` - document uploads
* ``default``- everything else

Identity prefers the authenticated user id, falling back to the client IP so
anonymous endpoints (login) are still protected.

Implementation note: these are plain module-level functions rather than a
callable class instance. FastAPI introspects a dependency's signature to build
its parameter model, and a class instance used as a dependency is not reliably
introspected -- that produced spurious ``422 validation_error`` responses on
every rate-limited route. Plain functions are the idiomatic, dependable form.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.api.dependencies import get_optional_user
from app.core.exceptions import RateLimitError
from app.core.rate_limit import RateLimitResult, get_rate_limiter
from app.models.user import User


def client_identity(request: Request) -> str:
    """Best-effort client identity for anonymous callers."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _enforce(scope: str, request: Request, current_user: User | None) -> RateLimitResult:
    """Consume one unit of quota in ``scope`` and stash the result on the request."""
    identity = (
        f"user:{current_user.id}" if current_user is not None else f"ip:{client_identity(request)}"
    )

    result = get_rate_limiter().check(scope, identity, raise_on_limit=False)

    # Read by RateLimitHeadersMiddleware and copied onto the response.
    request.state.rate_limit = result

    if not result.allowed:
        raise RateLimitError(
            f"Rate limit exceeded for '{scope}'. Try again in {result.retry_after}s.",
            retry_after=result.retry_after,
            limit=result.limit,
            remaining=0,
        )
    return result


def rate_limit_default(
    request: Request,
    current_user: Annotated[User | None, Depends(get_optional_user)],
) -> RateLimitResult:
    """Apply the general-purpose quota."""
    return _enforce("default", request, current_user)


def rate_limit_auth(
    request: Request,
    current_user: Annotated[User | None, Depends(get_optional_user)],
) -> RateLimitResult:
    """Apply the stricter authentication quota (brute-force protection)."""
    return _enforce("auth", request, current_user)


def rate_limit_chat(
    request: Request,
    current_user: Annotated[User | None, Depends(get_optional_user)],
) -> RateLimitResult:
    """Apply the chat/generation quota (the expensive path)."""
    return _enforce("chat", request, current_user)


def rate_limit_upload(
    request: Request,
    current_user: Annotated[User | None, Depends(get_optional_user)],
) -> RateLimitResult:
    """Apply the upload quota."""
    return _enforce("upload", request, current_user)
