"""Password hashing and JWT issuing/verification.

Uses ``bcrypt`` directly (no passlib) so the dependency surface stays small and
``PyJWT`` for tokens. Every function is pure with respect to the settings it is
given, which makes the unit tests trivial.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import bcrypt
import jwt

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError

TokenType = Literal["access", "refresh"]

# bcrypt only considers the first 72 bytes of the password.
_BCRYPT_MAX_BYTES = 72
_BCRYPT_ROUNDS = 12


# --------------------------------------------------------------------------- #
# Password hashing
# --------------------------------------------------------------------------- #
def _prepare_password(password: str) -> bytes:
    """Encode and pre-hash a password so long secrets stay safe.

    Passwords longer than 72 bytes are SHA-256 hashed first (base64 encoded),
    which is the documented workaround for bcrypt's input limit.
    """
    if not isinstance(password, str):
        raise AuthenticationError("Password must be a string.", code="invalid_password")
    raw = password.encode("utf-8")
    if len(raw) > _BCRYPT_MAX_BYTES:
        raw = hashlib.sha256(raw).hexdigest().encode("ascii")
    return raw


def hash_password(password: str) -> str:
    """Return a bcrypt hash for ``password``."""
    if not password or len(password) < 8:
        raise AuthenticationError(
            "Password must be at least 8 characters long.", code="weak_password"
        )
    return bcrypt.hashpw(_prepare_password(password), bcrypt.gensalt(_BCRYPT_ROUNDS)).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Constant-time verification of ``plain_password`` against a hash."""
    if not plain_password or not hashed_password:
        return False
    try:
        return bcrypt.checkpw(
            _prepare_password(plain_password), hashed_password.encode("utf-8")
        )
    except (ValueError, TypeError):
        return False


# --------------------------------------------------------------------------- #
# Token helpers
# --------------------------------------------------------------------------- #
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _expiry(token_type: TokenType, config: Settings) -> datetime:
    if token_type == "access":
        return _now() + timedelta(minutes=config.access_token_expire_minutes)
    return _now() + timedelta(days=config.refresh_token_expire_days)


def create_token(
    subject: str | int,
    token_type: TokenType = "access",
    *,
    scopes: list[str] | None = None,
    extra_claims: dict[str, Any] | None = None,
    config: Settings | None = None,
) -> str:
    """Create a signed JWT for ``subject`` (usually the user id)."""
    config = config or get_settings()
    issued_at = _now()
    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": token_type,
        "iss": config.jwt_issuer,
        "iat": int(issued_at.timestamp()),
        "nbf": int(issued_at.timestamp()),
        "exp": int(_expiry(token_type, config).timestamp()),
        "jti": uuid.uuid4().hex,
        "scopes": scopes or [],
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, config.secret_key, algorithm=config.jwt_algorithm)


def decode_token(
    token: str,
    *,
    expected_type: TokenType | None = None,
    config: Settings | None = None,
) -> dict[str, Any]:
    """Verify a JWT signature/expiry and return its claims."""
    config = config or get_settings()
    try:
        claims: dict[str, Any] = jwt.decode(
            token,
            config.secret_key,
            algorithms=[config.jwt_algorithm],
            issuer=config.jwt_issuer,
            options={"require": ["exp", "sub", "type"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Token has expired.", code="token_expired") from exc
    except jwt.InvalidIssuerError as exc:
        raise AuthenticationError("Token issuer is invalid.", code="invalid_issuer") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Token is invalid.", code="invalid_token") from exc

    if expected_type and claims.get("type") != expected_type:
        raise AuthenticationError(
            f"Expected a {expected_type} token.", code="invalid_token_type"
        )
    return claims


def create_access_token(
    subject: str | int, *, scopes: list[str] | None = None, config: Settings | None = None
) -> str:
    """Convenience wrapper for access tokens."""
    return create_token(subject, "access", scopes=scopes, config=config)


def create_refresh_token(
    subject: str | int, *, scopes: list[str] | None = None, config: Settings | None = None
) -> str:
    """Convenience wrapper for refresh tokens."""
    return create_token(subject, "refresh", scopes=scopes, config=config)


# --------------------------------------------------------------------------- #
# Misc helpers
# --------------------------------------------------------------------------- #
def generate_api_key(prefix: str = "rag") -> str:
    """Return a URL-safe random secret (used for API keys / celery tokens)."""
    return f"{prefix}_{secrets.token_urlsafe(32)}"


def secure_compare(left: str, right: str) -> bool:
    """Timing-safe string comparison."""
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))
