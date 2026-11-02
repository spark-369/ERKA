"""Domain specific exceptions mapped to HTTP responses by the API layer."""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for every expected (non-bug) application error."""

    status_code: int = 500
    code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: Any = None,
    ) -> None:
        self.message = message or self.message
        self.code = code or self.code
        self.status_code = status_code or self.status_code
        self.details = details
        super().__init__(self.message)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            payload["details"] = self.details
        return payload


class ValidationError(AppError):
    """Request payload failed business validation."""

    status_code = 422
    code = "validation_error"
    message = "The submitted data is invalid."


class NotFoundError(AppError):
    """Requested resource does not exist (or is not visible to the caller)."""

    status_code = 404
    code = "not_found"
    message = "The requested resource was not found."


class ConflictError(AppError):
    """Request conflicts with existing state (e.g. duplicate email)."""

    status_code = 409
    code = "conflict"
    message = "The resource already exists."


class AuthenticationError(AppError):
    """Credentials are missing or invalid."""

    status_code = 401
    code = "authentication_error"
    message = "Could not validate credentials."


class PermissionDeniedError(AppError):
    """Authenticated but not allowed to perform the action."""

    status_code = 403
    code = "permission_denied"
    message = "You do not have permission to perform this action."


class RateLimitError(AppError):
    """Client exceeded the configured request quota."""

    status_code = 429
    code = "rate_limit_exceeded"
    message = "Too many requests. Please slow down."

    def __init__(
        self,
        message: str | None = None,
        *,
        retry_after: int = 60,
        limit: int | None = None,
        remaining: int = 0,
        details: Any = None,
    ) -> None:
        super().__init__(message, details=details)
        self.retry_after = max(int(retry_after), 1)
        self.limit = limit
        self.remaining = remaining


class StorageError(AppError):
    """Relational, cache or vector storage failure."""

    status_code = 503
    code = "storage_error"
    message = "A storage backend is currently unavailable."


class VectorStoreError(StorageError):
    """ChromaDB (embedded vector store) failure."""

    code = "vector_store_error"
    message = "The vector store is currently unavailable."


class EmbeddingError(AppError):
    """Embedding model failed to produce vectors."""

    status_code = 500
    code = "embedding_error"
    message = "Failed to embed the provided content."


class LLMError(AppError):
    """Local language model failed to generate a response."""

    status_code = 503
    code = "llm_error"
    message = "The language model is currently unavailable."


class DocumentProcessingError(AppError):
    """Document could not be parsed, chunked or indexed."""

    status_code = 422
    code = "document_processing_error"
    message = "The document could not be processed."


class FileValidationError(AppError):
    """Uploaded file rejected by policy (type, size, emptiness)."""

    status_code = 400
    code = "invalid_file"
    message = "The uploaded file is not acceptable."


class TaskError(AppError):
    """Background (Celery) task could not be completed."""

    status_code = 500
    code = "task_error"
    message = "A background task failed."
