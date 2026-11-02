"""Unit tests for password hashing and JWT handling."""

from __future__ import annotations

import time
from datetime import timedelta

import pytest

from app.core.config import Settings
from app.core.exceptions import AuthenticationError
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    hash_password,
    secure_compare,
    verify_password,
)

pytestmark = pytest.mark.unit


class TestPasswordHashing:
    def test_hash_is_not_the_plaintext(self) -> None:
        hashed = hash_password("Password123")
        assert hashed != "Password123"
        assert hashed.startswith("$2b$")

    def test_verify_accepts_correct_password(self) -> None:
        hashed = hash_password("Password123")
        assert verify_password("Password123", hashed) is True

    def test_verify_rejects_wrong_password(self) -> None:
        hashed = hash_password("Password123")
        assert verify_password("password123", hashed) is False

    def test_same_password_produces_different_hashes(self) -> None:
        assert hash_password("Password123") != hash_password("Password123")

    def test_short_password_is_rejected(self) -> None:
        with pytest.raises(AuthenticationError) as excinfo:
            hash_password("short")
        assert excinfo.value.code == "weak_password"

    def test_long_password_is_supported(self) -> None:
        """Passwords beyond bcrypt's 72-byte limit must still verify."""
        long_password = "A1" * 100
        hashed = hash_password(long_password)
        assert verify_password(long_password, hashed) is True

    def test_verify_tolerates_garbage_hash(self) -> None:
        assert verify_password("Password123", "not-a-hash") is False


class TestJWT:
    def test_access_token_round_trip(self) -> None:
        token = create_access_token("user-123", scopes=["user"])
        claims = decode_token(token, expected_type="access")
        assert claims["sub"] == "user-123"
        assert claims["type"] == "access"
        assert claims["scopes"] == ["user"]
        assert "jti" in claims and "exp" in claims

    def test_refresh_token_round_trip(self) -> None:
        token = create_refresh_token("user-123")
        claims = decode_token(token, expected_type="refresh")
        assert claims["type"] == "refresh"

    def test_type_mismatch_is_rejected(self) -> None:
        token = create_access_token("user-123")
        with pytest.raises(AuthenticationError) as excinfo:
            decode_token(token, expected_type="refresh")
        assert excinfo.value.code == "invalid_token_type"

    def test_tampered_token_is_rejected(self) -> None:
        token = create_access_token("user-123")
        tampered = f"{token[:-4]}aaaa"
        with pytest.raises(AuthenticationError) as excinfo:
            decode_token(tampered)
        assert excinfo.value.code in {"invalid_token", "token_expired"}

    def test_token_from_other_secret_is_rejected(self) -> None:
        other = Settings(secret_key="a-completely-different-secret-key-value")
        token = create_access_token("user-123", config=other)
        with pytest.raises(AuthenticationError):
            decode_token(token)

    def test_expired_token_is_rejected(self) -> None:
        expired = Settings(
            secret_key="test-secret-key-for-unit-and-integration-tests-0123456789",
            access_token_expire_minutes=-1,
        )
        token = create_access_token("user-123", config=expired)
        time.sleep(0.01)
        with pytest.raises(AuthenticationError) as excinfo:
            decode_token(token)
        assert excinfo.value.code == "token_expired"

    def test_expiry_uses_configured_lifetime(self) -> None:
        config = Settings(access_token_expire_minutes=15)
        claims = decode_token(create_access_token("u", config=config))
        lifetime = claims["exp"] - claims["iat"]
        assert lifetime == int(timedelta(minutes=15).total_seconds())


class TestMiscSecurity:
    def test_generate_api_key_is_unique_and_prefixed(self) -> None:
        first, second = generate_api_key(), generate_api_key()
        assert first != second
        assert first.startswith("rag_")

    def test_secure_compare(self) -> None:
        assert secure_compare("abc", "abc") is True
        assert secure_compare("abc", "abd") is False
