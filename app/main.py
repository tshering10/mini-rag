from fastapi import FastAPI

from app.config import get_settings
from app.errors import handle_pdf_processing_error, handle_value_error
from app.routers import documents, query
from core.pdf_processor import PDFProcessingError
from utils.logging_config import configure_logging

configure_logging()
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="A small retrieval-augmented generation API.",
)

app.add_exception_handler(
    PDFProcessingError,
    handle_pdf_processing_error,
)
app.add_exception_handler(ValueError, handle_value_error)

app.include_router(documents.router)
app.include_router(query.router)


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    return {"status": "ok", "environment": settings.environment}
