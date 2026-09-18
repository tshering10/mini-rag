from pathlib import Path
from typing import Any

import pdfplumber
from pdfminer.pdfdocument import PDFSyntaxError


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be read or processed."""


class PDFProcessor:
    """Extract text and basic metadata from PDF files."""

    def extract_text(self, pdf_path: str | Path) -> str:
        """Return all extractable page text joined with blank lines."""
        document = self.extract_with_metadata(pdf_path)
        return document["text"]

    def extract_with_metadata(self, pdf_path: str | Path) -> dict[str, Any]:
        """Extract document text, page text, and PDF metadata."""
        path = Path(pdf_path)

        if not path.is_file():
            raise PDFProcessingError(f"PDF file does not exist: {path}")

        try:
            with pdfplumber.open(path) as pdf:
                pages: list[dict[str, Any]] = []

                for page_number, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""
                    pages.append(
                        {
                            "page_number": page_number,
                            "text": page_text.strip(),
                        }
                    )

                text = "\n\n".join(
                    page["text"] for page in pages if page["text"]
                )

                return {
                    "text": text,
                    "pages": pages,
                    "metadata": dict(pdf.metadata or {}),
                }
        except (OSError, PDFSyntaxError, ValueError) as error:
            raise PDFProcessingError(f"Unable to process PDF: {path}") from error