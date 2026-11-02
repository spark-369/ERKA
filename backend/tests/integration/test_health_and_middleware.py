"""Integration tests for health, root and cross-cutting HTTP behaviour."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestHealth:
    def test_liveness_probe(self, client) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["environment"] == "test"
        assert body["version"]

    def test_readiness_probe_reports_dependencies(self, client) -> None:
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200
        dependencies = response.json()["dependencies"]
        assert dependencies["database"] == "ok"
        assert "llm" in dependencies and "embeddings" in dependencies

    def test_root_endpoint(self, client) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["docs"] == "/docs"


class TestCrossCuttingBehaviour:
    def test_openapi_schema_is_served_with_bearer_auth(self, client) -> None:
        schema = client.get("/openapi.json").json()
        assert schema["info"]["title"]
        assert "BearerAuth" in schema["components"]["securitySchemes"]

    def test_request_id_header_is_returned(self, client) -> None:
        response = client.get("/api/v1/health")
        assert response.headers["X-Request-ID"]

    def test_supplied_request_id_is_echoed(self, client) -> None:
        response = client.get("/api/v1/health", headers={"X-Request-ID": "trace-me"})
        assert response.headers["X-Request-ID"] == "trace-me"

    def test_process_time_header_is_returned(self, client) -> None:
        assert float(client.get("/api/v1/health").headers["X-Process-Time-Ms"]) >= 0

    def test_security_headers_are_set(self, client) -> None:
        headers = client.get("/api/v1/health").headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"

    def test_rate_limit_headers_are_set(self, client) -> None:
        headers = client.get("/api/v1/health").headers
        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers

    def test_cors_preflight_is_allowed(self, client) -> None:
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

    def test_unknown_route_returns_error_envelope(self, client) -> None:
        response = client.get("/api/v1/does-not-exist")
        assert response.status_code == 404
        body = response.json()
        assert body["code"].startswith("http_")
        assert "message" in body

    def test_validation_error_uses_the_envelope(self, client) -> None:
        response = client.post("/api/v1/auth/login", json={"email": "bad"})
        assert response.status_code == 422
        body = response.json()
        assert body["code"] == "validation_error"
        assert body["details"]

    def test_unauthenticated_access_never_leaks_data(self, client) -> None:
        for path in ("/api/v1/documents", "/api/v1/chat/conversations", "/api/v1/tasks"):
            assert client.get(path).status_code == 401
