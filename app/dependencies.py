from functools import lru_cache

from services.rag_service import RAGService


@lru_cache
def get_rag_service() -> RAGService:
    """Return the shared RAG service used by the application."""
    return RAGService()
