"""Integration tests for the authentication and user endpoints."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

REGISTER = "/api/v1/auth/register"
LOGIN = "/api/v1/auth/login"
REFRESH = "/api/v1/auth/refresh"
ME = "/api/v1/auth/me"


class TestRegistration:
    def test_creates_an_account(self, client) -> None:
        response = client.post(
            REGISTER,
            json={
                "email": "New.User@Example.com",
                "password": "Password123",
                "full_name": "New User",
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["email"] == "new.user@example.com"  # normalised
        assert body["role"] == "user"
        assert body["is_active"] is True
        assert "hashed_password" not in body and "password" not in body

    def test_duplicate_email_is_rejected(self, client) -> None:
        payload = {"email": "dupe@example.com", "password": "Password123"}
        assert client.post(REGISTER, json=payload).status_code == 201
        conflict = client.post(REGISTER, json=payload)
        assert conflict.status_code == 409
        assert conflict.json()["code"] == "email_taken"

    def test_weak_password_is_rejected(self, client) -> None:
        response = client.post(
            REGISTER, json={"email": "weak@example.com", "password": "password"}
        )
        assert response.status_code == 422
        assert response.json()["code"] == "validation_error"

    def test_invalid_email_is_rejected(self, client) -> None:
        response = client.post(
            REGISTER, json={"email": "not-an-email", "password": "Password123"}
        )
        assert response.status_code == 422


class TestLogin:
    def test_returns_a_token_pair(self, client) -> None:
        client.post(REGISTER, json={"email": "login@example.com", "password": "Password123"})
        response = client.post(
            LOGIN, json={"email": "login@example.com", "password": "Password123"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["token_type"] == "bearer"
        assert body["access_token"] and body["refresh_token"]
        assert body["expires_in"] > 0
        assert body["user"]["email"] == "login@example.com"

    def test_wrong_password_is_401(self, client) -> None:
        client.post(REGISTER, json={"email": "wrong@example.com", "password": "Password123"})
        response = client.post(
            LOGIN, json={"email": "wrong@example.com", "password": "Password124"}
        )
        assert response.status_code == 401
        assert response.json()["code"] == "invalid_credentials"

    def test_unknown_email_gives_the_same_error(self, client) -> None:
        response = client.post(
            LOGIN, json={"email": "ghost@example.com", "password": "Password123"}
        )
        assert response.status_code == 401
        assert response.json()["code"] == "invalid_credentials"

    def test_deactivated_account_is_403(self, client, user_factory) -> None:
        user_factory(email="inactive@example.com", is_active=False)
        response = client.post(
            LOGIN, json={"email": "inactive@example.com", "password": "Password123"}
        )
        assert response.status_code == 403
        assert response.json()["code"] == "account_inactive"


class TestTokenRefresh:
    def test_rotates_the_pair(self, client) -> None:
        client.post(REGISTER, json={"email": "rotate@example.com", "password": "Password123"})
        tokens = client.post(
            LOGIN, json={"email": "rotate@example.com", "password": "Password123"}
        ).json()

        refreshed = client.post(REFRESH, json={"refresh_token": tokens["refresh_token"]})
        assert refreshed.status_code == 200
        assert refreshed.json()["access_token"]

    def test_access_token_is_not_accepted_as_refresh(self, client) -> None:
        client.post(REGISTER, json={"email": "swap@example.com", "password": "Password123"})
        tokens = client.post(
            LOGIN, json={"email": "swap@example.com", "password": "Password123"}
        ).json()

        response = client.post(REFRESH, json={"refresh_token": tokens["access_token"]})
        assert response.status_code == 401
        assert response.json()["code"] == "invalid_token_type"

    def test_garbage_token_is_rejected(self, client) -> None:
        response = client.post(REFRESH, json={"refresh_token": "not.a.jwt.value"})
        assert response.status_code == 401


class TestProtectedEndpoints:
    def test_me_requires_a_token(self, client) -> None:
        response = client.get(ME)
        assert response.status_code == 401
        assert response.json()["code"] == "missing_token"

    def test_me_returns_the_current_user(self, client, auth_headers, current_user) -> None:
        response = client.get(ME, headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["id"] == current_user.id

    def test_invalid_token_is_rejected(self, client) -> None:
        response = client.get(ME, headers={"Authorization": "Bearer garbage.token.here"})
        assert response.status_code == 401

    def test_logout_is_idempotent(self, client, auth_headers) -> None:
        assert client.post("/api/v1/auth/logout", headers=auth_headers).status_code == 200

    def test_profile_update(self, client, auth_headers) -> None:
        response = client.patch(
            "/api/v1/users/me", json={"full_name": "Updated Name"}, headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json()["full_name"] == "Updated Name"

    def test_stats_endpoint(self, client, auth_headers) -> None:
        response = client.get("/api/v1/users/me/stats", headers=auth_headers)
        assert response.status_code == 200
        body = response.json()
        assert body["documents"]["total"] == 0
        assert body["conversations"] == 0
