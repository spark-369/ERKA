"""Shared / generic schemas (pagination, envelopes, health)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ORMModel(BaseModel):
    """Base for schemas populated directly from ORM instances."""

    model_config = ConfigDict(from_attributes=True)


class MessageResponse(BaseModel):
    """Simple human-readable acknowledgement."""

    detail: str
    success: bool = True


class ErrorResponse(BaseModel):
    """Uniform error envelope returned by every exception handler."""

    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human readable explanation.")
    details: Any | None = Field(default=None, description="Optional structured context.")
    request_id: str | None = Field(default=None, description="Correlation id for logs.")


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    status: str = "ok"
    app_name: str
    version: str
    environment: str
    timestamp: datetime
    dependencies: dict[str, str] = Field(
        default_factory=dict, description="Per-dependency status map."
    )


class PaginationParams(BaseModel):
    """Validated query parameters for list endpoints."""

    page: int = Field(default=1, ge=1, le=10_000, description="1-based page number.")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page.")

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        return self.page_size


class Page(BaseModel, Generic[T]):
    """Offset pagination envelope.

    ``total_pages`` / ``has_next`` are serialised into the response so clients
    do not have to recompute the arithmetic (they are exposed as real fields,
    not properties, because Pydantic only serialises declared fields).
    """

    items: list[T]
    total: int = Field(description="Total number of matching rows.")
    page: int
    page_size: int
    total_pages: int = Field(default=0, description="Number of pages available.")
    has_next: bool = Field(default=False, description="Whether a further page exists.")

    def model_post_init(self, _context: object) -> None:
        """Derive the pagination helpers from ``total`` / ``page_size``."""
        if self.page_size > 0:
            computed_pages = (self.total + self.page_size - 1) // self.page_size
        else:
            computed_pages = 0
        self.total_pages = computed_pages
        self.has_next = self.page * self.page_size < self.total
