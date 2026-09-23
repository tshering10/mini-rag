from fastapi import APIRouter, Depends

from app.dependencies import get_rag_service
from models.schemas import SearchRequest, SearchResponse
from services.rag_service import RAGService

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=SearchResponse)
async def query_documents(
    request: SearchRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> SearchResponse:
    """Return the most relevant indexed chunks for a question."""
    return await rag_service.query(request)
