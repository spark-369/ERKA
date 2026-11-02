"""Text loading: bytes -> plain text.

Parsing lives behind :class:`DocumentLoader` so the ingestion pipeline never
cares whether the source was a PDF, DOCX or CSV. LangChain-compatible loaders
are used where they exist; small formats are handled directly.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import DocumentProcessingError
from app.core.logging import get_logger

logger = get_logger(__name__)

TEXT_ENCODINGS = ("utf-8", "utf-16", "latin-1")


@dataclass(slots=True)
class LoadedDocument:
    """Result of parsing a file: normalised text plus provenance metadata."""

    text: str
    source: str
    extension: str
    metadata: dict[str, Any] = field(default_factory=dict)
    page_count: int = 0

    @property
    def char_count(self) -> int:
        return len(self.text)


class DocumentLoader:
    """Dispatches to the right parser based on the file extension."""

    def __init__(self, config: Settings | None = None) -> None:
        self._config = config or get_settings()

    # ------------------------------------------------------------ public API
    def load_bytes(self, data: bytes, filename: str) -> LoadedDocument:
        """Parse raw upload bytes."""
        if not data:
            raise DocumentProcessingError("The uploaded file is empty.", code="empty_file")

        extension = Path(filename).suffix.lower()
        try:
            if extension == ".pdf":
                text, metadata = self._load_pdf(data)
            elif extension == ".docx":
                text, metadata = self._load_docx(data)
            elif extension in {".html", ".htm"}:
                text, metadata = self._load_html(data)
            elif extension == ".csv":
                text, metadata = self._load_csv(data)
            elif extension == ".json":
                text, metadata = self._load_json(data)
            else:
                # Text-like or unknown-but-allowed extension: best effort decode.
                text, metadata = self._load_text(data)
        except DocumentProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise parser failures
            raise DocumentProcessingError(
                f"Failed to parse '{filename}': {exc}", code="parse_failed"
            ) from exc

        normalised = self._normalise(text)
        if not normalised.strip():
            raise DocumentProcessingError(
                "No extractable text was found in the document.", code="no_text"
            )

        return LoadedDocument(
            text=normalised,
            source=filename,
            extension=extension,
            metadata=metadata,
            page_count=int(metadata.get("page_count", 0) or 0),
        )

    def load_path(self, path: str | Path) -> LoadedDocument:
        """Parse a file from disk (used by Celery workers)."""
        file_path = Path(path)
        if not file_path.exists():
            raise DocumentProcessingError(
                f"Stored file not found: {file_path}", code="file_missing"
            )
        return self.load_bytes(file_path.read_bytes(), file_path.name)

    # -------------------------------------------------------------- parsers
    @staticmethod
    def _normalise(text: str) -> str:
        """Collapse whitespace noise introduced by PDF extraction."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _load_text(self, data: bytes) -> tuple[str, dict[str, Any]]:
        for encoding in TEXT_ENCODINGS:
            try:
                return data.decode(encoding), {"encoding": encoding}
            except UnicodeDecodeError:
                continue
        raise DocumentProcessingError("Unable to decode text file.", code="decode_failed")

    def _load_pdf(self, data: bytes) -> tuple[str, dict[str, Any]]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise DocumentProcessingError(
                "pypdf is not installed; PDF uploads are unavailable.",
                code="pypdf_missing",
            ) from exc

        reader = PdfReader(io.BytesIO(data))
        pages: list[str] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                content = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001 - keep going page by page
                logger.warning("Failed to extract PDF page %s: %s", index, exc)
                content = ""
            # The page number is *not* inlined into the text: it would leak into
            # retrieved context, citation snippets and the model's answer as
            # "[Page N]" noise. The page count stays in metadata instead.
            if content.strip():
                pages.append(content.strip())

        metadata: dict[str, Any] = {
            "page_count": len(reader.pages),
            "pdf_metadata": {
                str(key): str(value) for key, value in (reader.metadata or {}).items()
            },
        }
        return "\n\n".join(pages), metadata

    def _load_docx(self, data: bytes) -> tuple[str, dict[str, Any]]:
        try:
            import docx  # python-docx
        except ImportError as exc:  # pragma: no cover
            raise DocumentProcessingError(
                "python-docx is not installed; DOCX uploads are unavailable.",
                code="docx_missing",
            ) from exc

        document = docx.Document(io.BytesIO(data))
        paragraphs = [para.text.strip() for para in document.paragraphs if para.text.strip()]
        tables: list[str] = []
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    tables.append(" | ".join(cells))
        body = "\n".join(paragraphs)
        if tables:
            body = f"{body}\n\n[Tables]\n" + "\n".join(tables)
        return body, {"paragraph_count": len(paragraphs), "table_rows": len(tables)}

    def _load_html(self, data: bytes) -> tuple[str, dict[str, Any]]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:  # pragma: no cover - graceful degradation
            return self._load_text(data)

        soup = BeautifulSoup(data, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else ""
        text = soup.get_text(separator="\n")
        metadata = {"title": title} if title else {}
        return text, metadata

    def _load_csv(self, data: bytes) -> tuple[str, dict[str, Any]]:
        text, _ = self._load_text(data)
        reader = csv.reader(io.StringIO(text))
        rows = [row for row in reader if any(cell.strip() for cell in row)]
        if not rows:
            return text, {"row_count": 0}
        header, *body = rows
        lines = [" | ".join(header)]
        lines.extend(" | ".join(row) for row in body)
        return "\n".join(lines), {"row_count": len(body), "columns": len(header)}

    def _load_json(self, data: bytes) -> tuple[str, dict[str, Any]]:
        text, _ = self._load_text(data)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text, {"valid_json": False}
        pretty = json.dumps(payload, indent=2, ensure_ascii=False)
        keys = list(payload.keys())[:50] if isinstance(payload, dict) else []
        return pretty, {"valid_json": True, "top_level_keys": keys}


def load_document(data: bytes, filename: str, config: Settings | None = None) -> LoadedDocument:
    """Functional shortcut around :class:`DocumentLoader`."""
    return DocumentLoader(config=config).load_bytes(data, filename)
