from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_rag_service
from models.schemas import RAGResponse, SearchRequest, SearchResponse
from services.llm_service import LLMServiceError
from services.rag_service import RAGService

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=SearchResponse)
async def query_documents(
    request: SearchRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> SearchResponse:
    """Return the most relevant indexed chunks for a question."""
    return await rag_service.query(request)


@router.post("/answer", response_model=RAGResponse)
async def answer_question(
    request: SearchRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> RAGResponse:
    """Return a Gemini answer grounded in retrieved document chunks."""
    try:
        return await rag_service.answer(request)
    except LLMServiceError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
