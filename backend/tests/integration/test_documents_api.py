"""Integration tests for the document upload and ingestion endpoints."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

DOCUMENTS = "/api/v1/documents"


def upload(client, headers, *, name: str = "notes.txt", content: bytes = b"hello world"):
    return client.post(
        f"{DOCUMENTS}/upload",
        files={"file": (name, content, "text/plain")},
        headers=headers,
    )


class TestUpload:
    def test_upload_is_accepted_and_ingested(self, client, auth_headers, vector_store) -> None:
        response = upload(
            client,
            auth_headers,
            content=b"Refund policy: customers may request a refund within 30 days.",
        )
        assert response.status_code == 202
        body = response.json()
        assert body["document"]["status"] == "pending"
        assert body["task_id"]
        assert "accepted" in body["message"].lower()

    def test_upload_requires_authentication(self, client) -> None:
        response = client.post(
            f"{DOCUMENTS}/upload", files={"file": ("a.txt", b"x", "text/plain")}
        )
        assert response.status_code == 401

    def test_unsupported_extension_is_400(self, client, auth_headers) -> None:
        response = upload(client, auth_headers, name="virus.exe", content=b"MZ")
        assert response.status_code == 400
        assert response.json()["code"] == "unsupported_extension"

    def test_duplicate_upload_is_409(self, client, auth_headers) -> None:
        assert upload(client, auth_headers).status_code == 202
        duplicate = upload(client, auth_headers)
        assert duplicate.status_code == 409
        assert duplicate.json()["code"] == "duplicate_document"

    def test_empty_file_is_rejected(self, client, auth_headers) -> None:
        response = upload(client, auth_headers, content=b"")
        assert response.status_code == 400
        assert response.json()["code"] in {"empty_file", "invalid_file"}

    def test_process_now_false_skips_ingestion(self, client, auth_headers) -> None:
        response = client.post(
            f"{DOCUMENTS}/upload",
            files={"file": ("later.txt", b"store me only", "text/plain")},
            data={"process_now": "false"},
            headers=auth_headers,
        )
        assert response.status_code == 202
        assert response.json()["task_id"] is None


class TestDocumentListing:
    def test_list_is_paginated(self, client, auth_headers) -> None:
        upload(client, auth_headers, name="one.txt", content=b"first document")
        upload(client, auth_headers, name="two.txt", content=b"second document")

        response = client.get(f"{DOCUMENTS}?page=1&page_size=1", headers=auth_headers)
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert body["page"] == 1
        assert body["page_size"] == 1
        assert len(body["items"]) == 1

    def test_search_filter(self, client, auth_headers) -> None:
        upload(client, auth_headers, name="alpha.txt", content=b"alpha body")
        upload(client, auth_headers, name="beta.txt", content=b"beta body")

        response = client.get(f"{DOCUMENTS}?search=alpha", headers=auth_headers)
        assert response.status_code == 200
        names = [item["original_filename"] for item in response.json()["items"]]
        assert names == ["alpha.txt"]

    def test_stats_endpoint(self, client, auth_headers) -> None:
        upload(client, auth_headers)
        response = client.get(f"{DOCUMENTS}/stats", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["total"] == 1

    def test_tenants_cannot_see_each_others_documents(
        self, client, auth_headers, user_factory
    ) -> None:
        from app.core.security import create_access_token

        upload(client, auth_headers, name="mine.txt", content=b"private knowledge")

        other = user_factory(email="intruder@example.com")
        other_headers = {
            "Authorization": f"Bearer {create_access_token(other.id, scopes=['user'])}"
        }

        listing = client.get(DOCUMENTS, headers=other_headers).json()
        assert listing["total"] == 0


class TestDocumentDetail:
    def test_get_single_document(self, client, auth_headers) -> None:
        document_id = upload(client, auth_headers).json()["document"]["id"]
        response = client.get(f"{DOCUMENTS}/{document_id}", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["id"] == document_id

    def test_missing_document_is_404(self, client, auth_headers) -> None:
        response = client.get(f"{DOCUMENTS}/does-not-exist", headers=auth_headers)
        assert response.status_code == 404
        assert response.json()["code"] == "document_not_found"

    def test_status_endpoint_reports_ingestion_state(self, client, auth_headers) -> None:
        document_id = upload(client, auth_headers).json()["document"]["id"]
        response = client.get(f"{DOCUMENTS}/{document_id}/status", headers=auth_headers)
        assert response.status_code == 200
        body = response.json()
        assert body["document_id"] == document_id
        assert body["status"] in {"pending", "processing", "ready", "failed"}
        assert 0 <= body["progress"] <= 100

    def test_update_metadata(self, client, auth_headers) -> None:
        document_id = upload(client, auth_headers).json()["document"]["id"]
        response = client.patch(
            f"{DOCUMENTS}/{document_id}",
            json={"original_filename": "renamed.txt", "tags": "hr,policy"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["original_filename"] == "renamed.txt"
        assert response.json()["tags"] == "hr,policy"

    def test_delete_document(self, client, auth_headers) -> None:
        document_id = upload(client, auth_headers).json()["document"]["id"]
        deleted = client.delete(f"{DOCUMENTS}/{document_id}", headers=auth_headers)
        assert deleted.status_code == 200
        assert client.get(f"{DOCUMENTS}/{document_id}", headers=auth_headers).status_code == 404

    def test_chunks_endpoint(self, client, auth_headers) -> None:
        document_id = upload(client, auth_headers).json()["document"]["id"]
        response = client.get(f"{DOCUMENTS}/{document_id}/chunks", headers=auth_headers)
        assert response.status_code == 200
        assert "items" in response.json()

    def test_reindex_dispatches_a_task(self, client, auth_headers) -> None:
        document_id = upload(client, auth_headers).json()["document"]["id"]
        response = client.post(f"{DOCUMENTS}/{document_id}/reindex", headers=auth_headers)
        assert response.status_code == 202
        assert response.json()["task_id"]
        assert response.json()["resource_id"] == document_id
