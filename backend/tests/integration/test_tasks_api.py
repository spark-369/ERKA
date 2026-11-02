"""Integration tests for the background task endpoints."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

TASKS = "/api/v1/tasks"


class TestTaskEndpoints:
    def test_listing_requires_authentication(self, client) -> None:
        assert client.get(TASKS).status_code == 401

    def test_listing_is_empty_for_a_new_user(self, client, auth_headers) -> None:
        response = client.get(TASKS, headers=auth_headers)
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 0
        assert body["items"] == []

    def test_upload_registers_a_task_record(self, client, auth_headers) -> None:
        client.post(
            "/api/v1/documents/upload",
            files={"file": ("task.txt", b"task body", "text/plain")},
            headers=auth_headers,
        )
        body = client.get(TASKS, headers=auth_headers).json()
        assert body["total"] == 1
        task = body["items"][0]
        assert task["name"] == "ingest_document"
        assert task["resource_type"] == "document"
        assert task["status"] in {"queued", "running", "succeeded", "failed"}

    def test_task_can_be_fetched_by_id(self, client, auth_headers) -> None:
        upload = client.post(
            "/api/v1/documents/upload",
            files={"file": ("fetch.txt", b"fetch body", "text/plain")},
            headers=auth_headers,
        ).json()
        task_id = upload["task_id"]

        response = client.get(f"{TASKS}/{task_id}", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["celery_task_id"] == task_id

    def test_unknown_task_is_404(self, client, auth_headers) -> None:
        response = client.get(f"{TASKS}/unknown-task-id", headers=auth_headers)
        assert response.status_code == 404
        assert response.json()["code"] == "task_not_found"

    def test_worker_health_probe_is_dispatched(self, client, auth_headers) -> None:
        response = client.post(f"{TASKS}/maintenance/health", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["status"] == "queued"

    def test_users_only_see_their_own_tasks(self, client, auth_headers, user_factory) -> None:
        from app.core.security import create_access_token

        client.post(
            "/api/v1/documents/upload",
            files={"file": ("mine.txt", b"private", "text/plain")},
            headers=auth_headers,
        )
        other = user_factory(email="nosy@example.com")
        other_headers = {
            "Authorization": f"Bearer {create_access_token(other.id, scopes=['user'])}"
        }
        assert client.get(TASKS, headers=other_headers).json()["total"] == 0
