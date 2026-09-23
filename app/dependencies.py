from functools import lru_cache

from app.config import get_settings
from services.llm_service import LLMService
from services.rag_service import RAGService


@lru_cache
def get_rag_service() -> RAGService:
    """Return the shared RAG service used by the application."""
    settings = get_settings()
    llm_service = (
        LLMService(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model,
        )
        if settings.gemini_api_key
        else None
    )
    return RAGService(llm_service=llm_service)
