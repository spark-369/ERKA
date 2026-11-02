"""HTTP middleware: request correlation, timing and rate-limit headers."""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id, measure duration and log the access line."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id
        request.state.user_id = getattr(request.state, "user_id", None)

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "request failed method=%s path=%s duration_ms=%.2f request_id=%s",
                request.method,
                request.url.path,
                duration_ms,
                request_id,
            )
            raise

        duration_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

        logger.info(
            "request method=%s path=%s status=%s duration_ms=%.2f request_id=%s user=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
            getattr(request.state, "user_id", None) or "anonymous",
        )
        return response


class RateLimitHeadersMiddleware(BaseHTTPMiddleware):
    """Copy the rate-limit result computed by the dependency onto the response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        result = getattr(request.state, "rate_limit", None)
        if result is not None:
            for header, value in result.as_headers().items():
                response.headers.setdefault(header, value)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Conservative hardening headers for a JSON/SSE API."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response
