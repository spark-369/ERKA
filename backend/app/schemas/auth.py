"""Authentication and user schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.schemas.common import ORMModel


class UserBase(BaseModel):
    """Fields shared by create/update/read payloads."""

    email: EmailStr
    full_name: str | None = Field(default=None, max_length=160)


class UserCreate(UserBase):
    """Registration payload."""

    password: str = Field(min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def _password_strength(cls, value: str) -> str:
        if value.isdigit() or value.isalpha():
            raise ValueError("Password must contain letters and numbers.")
        return value


class UserUpdate(BaseModel):
    """Self-service profile update."""

    full_name: str | None = Field(default=None, max_length=160)
    password: str | None = Field(default=None, min_length=8, max_length=128)


class UserRead(ORMModel):
    """Public user representation (never exposes the hash)."""

    id: str
    email: EmailStr
    full_name: str | None = None
    role: str
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None = None


class LoginRequest(BaseModel):
    """Credentials exchanged for a token pair."""

    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class RefreshRequest(BaseModel):
    """Refresh-token rotation request."""

    refresh_token: str = Field(min_length=10)


class TokenPair(BaseModel):
    """JWT access/refresh pair returned by the auth endpoints."""

    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(description="Access token lifetime in seconds.")
    user: UserRead | None = None
