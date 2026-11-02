"""Pydantic v2 schemas: the API request/response contracts."""

from app.schemas.auth import (
    LoginRequest,
    RefreshRequest,
    TokenPair,
    UserCreate,
    UserRead,
    UserUpdate,
)
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatStreamEvent,
    ConversationCreate,
    ConversationRead,
    MessageRead,
    SourceReference,
)
from app.schemas.common import (
    ErrorResponse,
    HealthResponse,
    MessageResponse,
    Page,
    PaginationParams,
)
from app.schemas.document import (
    DocumentChunkRead,
    DocumentRead,
    DocumentStatusRead,
    DocumentUploadResponse,
    DocumentUpdate,
)
from app.schemas.task import TaskRead, TaskSubmitResponse

__all__ = [
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatStreamEvent",
    "ConversationCreate",
    "ConversationRead",
    "DocumentChunkRead",
    "DocumentRead",
    "DocumentStatusRead",
    "DocumentUpdate",
    "DocumentUploadResponse",
    "ErrorResponse",
    "HealthResponse",
    "LoginRequest",
    "MessageRead",
    "MessageResponse",
    "Page",
    "PaginationParams",
    "RefreshRequest",
    "SourceReference",
    "TaskRead",
    "TaskSubmitResponse",
    "TokenPair",
    "UserCreate",
    "UserRead",
    "UserUpdate",
]
