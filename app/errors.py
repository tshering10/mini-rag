import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from core.pdf_processor import PDFProcessingError

logger = logging.getLogger(__name__)


async def handle_pdf_processing_error(
    request: Request,
    error: PDFProcessingError,
) -> JSONResponse:
    """Return a safe response for invalid or unreadable PDF files."""
    logger.warning("PDF processing failed: %s", error)
    return JSONResponse(
        status_code=422,
        content={"detail": "Unable to process the uploaded PDF"},
    )


async def handle_value_error(
    request: Request,
    error: ValueError,
) -> JSONResponse:
    """Return a client error for invalid service input."""
    logger.warning("Invalid request value: %s", error)
    return JSONResponse(
        status_code=400,
        content={"detail": str(error)},
    )
