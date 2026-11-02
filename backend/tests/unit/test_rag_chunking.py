"""Unit tests for LangChain-powered loading and chunking."""

from __future__ import annotations

import json

import pytest

from app.core.exceptions import DocumentProcessingError
from app.rag.chunking import TextChunker, chunk_documents, count_tokens
from app.rag.loaders import DocumentLoader, LoadedDocument, load_document

pytestmark = pytest.mark.unit


class TestDocumentLoader:
    def test_loads_plain_text(self) -> None:
        loaded = DocumentLoader().load_bytes(b"Hello world.\nSecond line.", "notes.txt")
        assert isinstance(loaded, LoadedDocument)
        assert "Hello world." in loaded.text
        assert loaded.extension == ".txt"
        assert loaded.source == "notes.txt"

    def test_normalises_whitespace(self) -> None:
        loaded = DocumentLoader().load_bytes(b"a    b\n\n\n\nc", "x.txt")
        assert loaded.text == "a b\n\nc"

    def test_rejects_empty_payload(self) -> None:
        with pytest.raises(DocumentProcessingError) as excinfo:
            DocumentLoader().load_bytes(b"", "empty.txt")
        assert excinfo.value.code == "empty_file"

    def test_rejects_whitespace_only_document(self) -> None:
        with pytest.raises(DocumentProcessingError) as excinfo:
            DocumentLoader().load_bytes(b"   \n\t  ", "blank.txt")
        assert excinfo.value.code == "no_text"

    def test_loads_csv_as_pipe_separated_rows(self) -> None:
        data = b"name,role\nalice,admin\nbob,user\n"
        loaded = DocumentLoader().load_bytes(data, "people.csv")
        assert "name | role" in loaded.text
        assert "alice | admin" in loaded.text
        assert loaded.metadata["row_count"] == 2

    def test_loads_json_pretty_printed(self) -> None:
        payload = {"service": "rag", "version": 1}
        loaded = DocumentLoader().load_bytes(json.dumps(payload).encode(), "cfg.json")
        assert '"service": "rag"' in loaded.text
        assert loaded.metadata["valid_json"] is True

    def test_invalid_json_falls_back_to_raw_text(self) -> None:
        loaded = DocumentLoader().load_bytes(b"{not json", "broken.json")
        assert loaded.metadata["valid_json"] is False

    def test_loads_html_stripping_scripts(self) -> None:
        html = b"<html><head><title>Docs</title></head><body><p>Body text</p><script>bad()</script></body></html>"
        loaded = DocumentLoader().load_bytes(html, "page.html")
        assert "Body text" in loaded.text
        assert "bad()" not in loaded.text
        assert loaded.metadata["title"] == "Docs"

    def test_load_path_missing_file(self, tmp_path) -> None:
        with pytest.raises(DocumentProcessingError) as excinfo:
            DocumentLoader().load_path(tmp_path / "nope.txt")
        assert excinfo.value.code == "file_missing"

    def test_load_document_helper(self) -> None:
        assert "hello" in load_document(b"hello", "a.txt").text.lower()


class TestTextChunker:
    def test_returns_empty_for_blank_text(self) -> None:
        assert TextChunker(chunk_size=100, chunk_overlap=10).split_text("   ") == []

    def test_short_text_is_a_single_chunk(self) -> None:
        chunks = TextChunker(chunk_size=500, chunk_overlap=50).split_text("Short text.")
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].content == "Short text."

    def test_long_text_is_split_into_multiple_chunks(self) -> None:
        text = "\n\n".join(f"Paragraph {index}. " + ("word " * 60) for index in range(8))
        chunks = TextChunker(chunk_size=300, chunk_overlap=40).split_text(text)
        assert len(chunks) > 1
        assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))

    def test_chunks_carry_metadata(self) -> None:
        chunks = TextChunker(chunk_size=200, chunk_overlap=20).split_text(
            "alpha " * 100, metadata={"document_id": "doc-1"}
        )
        assert chunks[0].metadata["document_id"] == "doc-1"
        assert chunks[0].metadata["chunk_index"] == 0

    def test_char_and_token_counts_are_consistent(self) -> None:
        chunks = TextChunker(chunk_size=300, chunk_overlap=30).split_text("token " * 200)
        for chunk in chunks:
            assert chunk.char_count == len(chunk.content)
            assert chunk.token_count == count_tokens(chunk.content)

    def test_overlap_larger_than_size_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_split_documents_aggregates_metadata(self) -> None:
        documents = [
            LoadedDocument(text="one " * 50, source="a.txt", extension=".txt"),
            LoadedDocument(text="two " * 50, source="b.txt", extension=".txt"),
        ]
        chunks = TextChunker(chunk_size=200, chunk_overlap=20).split_documents(documents)
        sources = {chunk.metadata["source"] for chunk in chunks}
        assert sources == {"a.txt", "b.txt"}

    def test_chunk_documents_helper(self) -> None:
        documents = [LoadedDocument(text="x " * 200, source="a.txt", extension=".txt")]
        assert chunk_documents(documents, chunk_size=100, chunk_overlap=10)


class TestTokenCounting:
    def test_zero_for_empty(self) -> None:
        assert count_tokens("") == 0

    def test_approximates_four_chars_per_token(self) -> None:
        assert count_tokens("a" * 400) == 100
