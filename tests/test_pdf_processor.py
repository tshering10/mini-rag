from pathlib import Path

import pytest

from core import pdf_processor
from core.pdf_processor import PDFProcessingError, PDFProcessor


class FakePage:
    def __init__(self, text: str | None) -> None:
        self._text = text

    def extract_text(self) -> str | None:
        return self._text


class FakePDF:
    pages = [FakePage(" First page "), FakePage(None), FakePage("Second page")]
    metadata = {"Title": "Test document"}

    def __enter__(self) -> "FakePDF":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_extract_with_metadata_returns_pages_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"placeholder")
    monkeypatch.setattr(pdf_processor.pdfplumber, "open", lambda path: FakePDF())

    result = PDFProcessor().extract_with_metadata(pdf_path)

    assert result["text"] == "First page\n\nSecond page"
    assert result["pages"] == [
        {"page_number": 1, "text": "First page"},
        {"page_number": 2, "text": ""},
        {"page_number": 3, "text": "Second page"},
    ]
    assert result["metadata"] == {"Title": "Test document"}


def test_extract_text_returns_combined_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"placeholder")
    monkeypatch.setattr(pdf_processor.pdfplumber, "open", lambda path: FakePDF())

    assert PDFProcessor().extract_text(pdf_path) == "First page\n\nSecond page"


def test_missing_pdf_raises_processing_error(tmp_path: Path) -> None:
    with pytest.raises(PDFProcessingError, match="does not exist"):
        PDFProcessor().extract_text(tmp_path / "missing.pdf")
