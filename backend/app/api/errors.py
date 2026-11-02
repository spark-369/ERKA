"""Central exception handlers producing the uniform error envelope."""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppError, RateLimitError
from app.core.logging import get_logger
from app.schemas.common import ErrorResponse

logger = get_logger(__name__)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _json_safe(details: Any) -> Any:  # noqa: ANN401 - arbitrary error detail
    """Coerce error details into plain JSON-safe Python values.

    Pydantic validation errors embed the originating exception in a ``ctx`` key
    (e.g. the ``ValueError`` raised by a custom field validator). That object has
    no JSON representation, so serialising the response would itself raise
    ``PydanticSerializationError`` and mask the real 422.

    Round-tripping through ``json`` with ``default=str`` guarantees the result
    contains only dicts/lists/str/numbers/bools/None, while preserving ``loc``,
    ``msg`` and ``type``. ``ctx`` survives as a readable string.
    """
    if details is None:
        return None
    try:
        return json.loads(json.dumps(details, default=str))
    except (TypeError, ValueError):
        # Circular references or similar: fall back to a plain string.
        return str(details)


def _json_error(
    *,
    status_code: int,
    code: str,
    message: str,
    request: Request,
    details: Any = None,  # noqa: ANN401 - arbitrary error detail
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        code=code,
        message=message,
        details=_json_safe(details),
        request_id=_request_id(request),
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every handler to the application."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        headers: dict[str, str] = {}
        if isinstance(exc, RateLimitError):
            headers["Retry-After"] = str(exc.retry_after)
            if exc.limit is not None:
                headers["X-RateLimit-Limit"] = str(exc.limit)
                headers["X-RateLimit-Remaining"] = "0"
        logger.warning(
            "Application error %s on %s %s: %s",
            exc.code,
            request.method,
            request.url.path,
            exc.message,
        )
        return _json_error(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            details=exc.details,
            request=request,
            headers=headers or None,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.info("Validation error on %s %s", request.method, request.url.path)
        return _json_error(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="validation_error",
            message="The submitted data is invalid.",
            details=exc.errors(),
            request=request,
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return _json_error(
            status_code=exc.status_code,
            code=f"http_{exc.status_code}",
            message=str(exc.detail),
            request=request,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(SQLAlchemyError)
    async def handle_db_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        logger.exception("Database error on %s %s", request.method, request.url.path)
        return _json_error(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="storage_error",
            message="A storage backend is currently unavailable.",
            request=request,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return _json_error(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="internal_error",
            message="An unexpected error occurred.",
            request=request,
        )
