"""Integration tests for the RAG chat endpoints (including SSE streaming)."""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.integration

CHAT = "/api/v1/chat"
COMPLETIONS = f"{CHAT}/completions"
STREAM = f"{CHAT}/stream"
CONVERSATIONS = f"{CHAT}/conversations"


def seed_knowledge(client, headers, *, text: str, name: str = "kb.txt"):
    """Upload + ingest a document so retrieval has something to find.

    Asserts the document reached ``ready``: ingestion runs inline (eager Celery)
    and silently leaves the document ``pending`` if it cannot see it, which would
    otherwise surface much later as a confusing "no citations" failure.
    """
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": (name, text.encode(), "text/plain")},
        headers=headers,
    )
    assert response.status_code == 202, response.text
    document_id = response.json()["document"]["id"]

    status = client.get(f"/api/v1/documents/{document_id}/status", headers=headers).json()
    assert status["status"] == "ready", f"ingestion did not complete: {status}"
    assert status["chunk_count"] > 0, f"ingestion produced no chunks: {status}"
    return document_id


def parse_sse(raw: str) -> list[tuple[str, dict]]:
    """Parse an SSE payload into ``(event_name, data)`` pairs."""
    events: list[tuple[str, dict]] = []
    for block in raw.strip().split("\n\n"):
        if not block.strip():
            continue
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if name:
            events.append((name, data or {}))
    return events


class TestCompletion:
    def test_answers_using_uploaded_knowledge(self, client, auth_headers) -> None:
        seed_knowledge(
            client,
            auth_headers,
            text="The refund window is 30 days from the date of purchase.",
            name="refunds.txt",
        )
        response = client.post(
            COMPLETIONS,
            json={"question": "What is the refund window?", "use_history": False},
            headers=auth_headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        # The answer must be grounded: a bare "I do not have enough information"
        # is also a non-empty string, so assert on the retrieved context too.
        assert body["sources"], "answer was not grounded in the knowledge base"
        # Structured output: the fact lands in a bullet point, not the prose.
        assert any("30 days" in point for point in body["key_points"])
        assert body["conversation_id"] and body["message_id"]
        assert body["latency_ms"] >= 0
        assert body["model_name"] == "stub-extractive-v1"

    def test_citations_are_returned(self, client, auth_headers) -> None:
        seed_knowledge(
            client, auth_headers, text="Vacation policy grants twenty paid days.", name="hr.txt"
        )
        body = client.post(
            COMPLETIONS,
            json={"question": "How many vacation days?", "use_history": False},
            headers=auth_headers,
        ).json()
        assert body["sources"], "expected at least one citation"
        source = body["sources"][0]
        assert {"chunk_id", "document_id", "document_name", "score", "snippet"} <= set(source)
        assert 0.0 <= source["score"] <= 1.0

    def test_answers_using_multiple_uploaded_documents(self, client, auth_headers) -> None:
        """A single question can be grounded in several uploaded files."""
        seed_knowledge(
            client,
            auth_headers,
            text="The refund window is 30 days from the date of purchase.",
            name="refunds.txt",
        )
        seed_knowledge(
            client,
            auth_headers,
            text="Vacation policy grants twenty paid days per year.",
            name="vacation.txt",
        )
        body = client.post(
            COMPLETIONS,
            json={"question": "refund window vacation policy paid days"},
            headers=auth_headers,
        ).json()
        assert body["sources"], "expected citations from the knowledge base"
        assert len(body["sources"]) <= 4  # standard RAG default (k=4)
        joined = " ".join(body["key_points"])
        assert "30 days" in joined
        assert "twenty paid days" in joined

    def test_structured_output_returns_citations(self, client, auth_headers) -> None:
        """The structured path surfaces citation offsets and a structured flag."""
        seed_knowledge(
            client, auth_headers, text="The refund window is 30 days.", name="refunds.txt"
        )
        body = client.post(
            COMPLETIONS,
            json={"question": "What is the refund window?"},
            headers=auth_headers,
        ).json()
        assert body["structured"] is True
        assert body["citations"], "expected at least one citation offset"
        assert all(0 <= offset < len(body["sources"]) for offset in body["citations"])
        assert any(source["cited"] for source in body["sources"])

    def test_document_scope_restricts_retrieval(self, client, auth_headers) -> None:
        """Passing ``document_ids`` limits retrieval to that document."""
        scoped = seed_knowledge(
            client, auth_headers, text="The refund window is 30 days.", name="refunds.txt"
        )
        seed_knowledge(
            client, auth_headers, text="Vacation policy grants twenty paid days.", name="hr.txt"
        )
        body = client.post(
            COMPLETIONS,
            json={"question": "What is the policy?", "document_ids": [scoped]},
            headers=auth_headers,
        ).json()
        assert body["sources"]
        assert {source["document_id"] for source in body["sources"]} == {scoped}

    def test_conversation_scope_is_honoured(self, client, auth_headers) -> None:
        """A conversation created with a scope keeps it on later turns."""
        scoped = seed_knowledge(
            client, auth_headers, text="The refund window is 30 days.", name="refunds.txt"
        )
        seed_knowledge(
            client, auth_headers, text="Vacation policy grants twenty paid days.", name="hr.txt"
        )
        conversation_id = client.post(
            CONVERSATIONS,
            json={"title": "Scoped", "document_ids": [scoped]},
            headers=auth_headers,
        ).json()["id"]
        body = client.post(
            COMPLETIONS,
            json={"question": "What is the policy?", "conversation_id": conversation_id},
            headers=auth_headers,
        ).json()
        assert body["sources"]
        assert {source["document_id"] for source in body["sources"]} == {scoped}

    def test_deleted_scope_falls_back_to_the_whole_knowledge_base(
        self, client, auth_headers
    ) -> None:
        """A scope pointing at deleted documents must not hide the corpus.

        A conversation persists the document ids it was created with. Delete
        those documents, upload a replacement (which gets a fresh id), and the
        stored scope refers to nothing - so retrieval used to return no sources
        and every later question looked like an empty knowledge base.
        """
        stale = seed_knowledge(
            client, auth_headers, text="The refund window is 30 days.", name="refunds.txt"
        )
        conversation_id = client.post(
            CONVERSATIONS,
            json={"title": "Scoped", "document_ids": [stale]},
            headers=auth_headers,
        ).json()["id"]

        assert client.delete(f"/api/v1/documents/{stale}", headers=auth_headers).status_code == 200
        seed_knowledge(
            client,
            auth_headers,
            text="Backups run nightly at 02:00 UTC.",
            name="ops.txt",
        )

        body = client.post(
            COMPLETIONS,
            json={"question": "When do backups run?", "conversation_id": conversation_id},
            headers=auth_headers,
        ).json()
        assert body["sources"], "stale scope hid the re-uploaded document"
        assert any(source["document_name"] == "ops.txt" for source in body["sources"])

    def test_unknown_topic_admits_ignorance(self, client, auth_headers) -> None:
        body = client.post(
            COMPLETIONS,
            json={"question": "What is the airspeed velocity of a swallow?"},
            headers=auth_headers,
        ).json()
        assert "enough information" in body["answer"].lower()

    def test_requires_authentication(self, client) -> None:
        assert client.post(COMPLETIONS, json={"question": "hi"}).status_code == 401

    def test_empty_question_is_rejected(self, client, auth_headers) -> None:
        response = client.post(COMPLETIONS, json={"question": ""}, headers=auth_headers)
        assert response.status_code == 422

    def test_conversation_is_created_and_reused(self, client, auth_headers) -> None:
        first = client.post(
            COMPLETIONS, json={"question": "First question?"}, headers=auth_headers
        ).json()
        second = client.post(
            COMPLETIONS,
            json={"question": "Follow up?", "conversation_id": first["conversation_id"]},
            headers=auth_headers,
        ).json()
        assert second["conversation_id"] == first["conversation_id"]

    def test_unknown_conversation_is_404(self, client, auth_headers) -> None:
        response = client.post(
            COMPLETIONS,
            json={"question": "hi", "conversation_id": "nope"},
            headers=auth_headers,
        )
        assert response.status_code == 404


class TestStreaming:
    def test_stream_emits_the_expected_event_sequence(self, client, auth_headers) -> None:
        seed_knowledge(
            client, auth_headers, text="The support SLA is 99.9 percent uptime.", name="sla.txt"
        )
        with client.stream(
            "POST", STREAM, json={"question": "What is the SLA?"}, headers=auth_headers
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            raw = "".join(response.iter_text())

        events = parse_sse(raw)
        names = [name for name, _ in events]
        assert names[0] == "start"
        assert "sources" in names
        assert "token" in names
        assert names[-1] == "done"

    def test_stream_tokens_reconstruct_the_answer(self, client, auth_headers) -> None:
        seed_knowledge(client, auth_headers, text="Alpha bravo charlie delta.", name="abc.txt")
        with client.stream(
            "POST", STREAM, json={"question": "Alpha?"}, headers=auth_headers
        ) as response:
            raw = "".join(response.iter_text())

        events = parse_sse(raw)
        tokens = "".join(data["text"] for name, data in events if name == "token")
        done = next(data for name, data in events if name == "done")
        assert tokens.strip() == done["answer"].strip()
        # The done frame always carries the structured-output contract fields.
        assert "citations" in done
        assert "structured" in done

    def test_stream_start_carries_the_model_name(self, client, auth_headers) -> None:
        with client.stream(
            "POST", STREAM, json={"question": "anything?"}, headers=auth_headers
        ) as response:
            raw = "".join(response.iter_text())
        start = next(data for name, data in parse_sse(raw) if name == "start")
        assert start["model_name"] == "stub-extractive-v1"
        assert start["conversation_id"]

    def test_stream_persists_both_turns(self, client, auth_headers) -> None:
        with client.stream(
            "POST", STREAM, json={"question": "Persist me?"}, headers=auth_headers
        ) as response:
            raw = "".join(response.iter_text())
        done = next(data for name, data in parse_sse(raw) if name == "done")

        detail = client.get(
            f"{CONVERSATIONS}/{done['conversation_id']}", headers=auth_headers
        ).json()
        roles = [message["role"] for message in detail["messages"]]
        assert roles == ["user", "assistant"]
        assert detail["messages"][1]["model_name"] == "stub-extractive-v1"

    def test_stream_requires_authentication(self, client) -> None:
        assert client.post(STREAM, json={"question": "hi"}).status_code == 401

    def test_stream_in_conversation_route(self, client, auth_headers) -> None:
        conversation_id = client.post(
            CONVERSATIONS, json={"title": "SSE thread"}, headers=auth_headers
        ).json()["id"]
        with client.stream(
            "POST",
            f"{STREAM}/{conversation_id}",
            json={"question": "Continue?"},
            headers=auth_headers,
        ) as response:
            assert response.status_code == 200
            raw = "".join(response.iter_text())
        done = next(data for name, data in parse_sse(raw) if name == "done")
        assert done["conversation_id"] == conversation_id


class TestConversationManagement:
    def test_create_and_list(self, client, auth_headers) -> None:
        created = client.post(CONVERSATIONS, json={"title": "My thread"}, headers=auth_headers)
        assert created.status_code == 201

        listing = client.get(CONVERSATIONS, headers=auth_headers).json()
        assert listing["total"] == 1
        assert listing["items"][0]["title"] == "My thread"

    def test_rename_and_archive(self, client, auth_headers) -> None:
        conversation_id = client.post(CONVERSATIONS, json={}, headers=auth_headers).json()["id"]

        renamed = client.patch(
            f"{CONVERSATIONS}/{conversation_id}",
            json={"title": "Renamed", "is_archived": True},
            headers=auth_headers,
        )
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "Renamed"
        assert renamed.json()["is_archived"] == 1

        default_list = client.get(CONVERSATIONS, headers=auth_headers).json()
        assert default_list["total"] == 0
        with_archived = client.get(
            f"{CONVERSATIONS}?include_archived=true", headers=auth_headers
        ).json()
        assert with_archived["total"] == 1

    def test_delete_conversation(self, client, auth_headers) -> None:
        conversation_id = client.post(CONVERSATIONS, json={}, headers=auth_headers).json()["id"]
        assert client.delete(f"{CONVERSATIONS}/{conversation_id}", headers=auth_headers).status_code == 200
        assert client.get(f"{CONVERSATIONS}/{conversation_id}", headers=auth_headers).status_code == 404

    def test_history_is_persisted_in_order(self, client, auth_headers) -> None:
        conversation_id = client.post(CONVERSATIONS, json={}, headers=auth_headers).json()["id"]
        for question in ("First?", "Second?", "Third?"):
            client.post(
                COMPLETIONS,
                json={"question": question, "conversation_id": conversation_id},
                headers=auth_headers,
            )

        detail = client.get(f"{CONVERSATIONS}/{conversation_id}", headers=auth_headers).json()
        assert detail["message_count"] == 6
        user_messages = [
            message["content"] for message in detail["messages"] if message["role"] == "user"
        ]
        assert user_messages == ["First?", "Second?", "Third?"]

    def test_citations_are_persisted_with_the_answer(self, client, auth_headers) -> None:
        seed_knowledge(client, auth_headers, text="Backups run nightly at 02:00 UTC.", name="ops.txt")
        conversation_id = client.post(CONVERSATIONS, json={}, headers=auth_headers).json()["id"]
        client.post(
            COMPLETIONS,
            json={"question": "When do backups run?", "conversation_id": conversation_id},
            headers=auth_headers,
        )

        detail = client.get(f"{CONVERSATIONS}/{conversation_id}", headers=auth_headers).json()
        assistant = detail["messages"][-1]
        assert assistant["sources"]
        assert "ops.txt" in assistant["sources"]

    def test_other_users_cannot_read_a_conversation(
        self, client, auth_headers, user_factory
    ) -> None:
        from app.core.security import create_access_token

        conversation_id = client.post(CONVERSATIONS, json={}, headers=auth_headers).json()["id"]
        other = user_factory(email="peeker@example.com")
        other_headers = {
            "Authorization": f"Bearer {create_access_token(other.id, scopes=['user'])}"
        }
        assert client.get(f"{CONVERSATIONS}/{conversation_id}", headers=other_headers).status_code == 404
