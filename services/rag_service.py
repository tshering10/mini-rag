import logging
from time import perf_counter
from pathlib import Path

from core.embeddings import EmbeddingService
from core.indexer import Indexer
from core.vector_store import VectorStore
from models.schemas import (
    IndexingResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

logger = logging.getLogger(__name__)


class RAGService:
    """Coordinate document indexing and semantic retrieval."""

    def __init__(
        self,
        indexer: Indexer | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        if indexer is not None:
            self.indexer = indexer
            self.embedding_service = (
                embedding_service or indexer.embedding_service
            )
            self.vector_store = vector_store or indexer.vector_store
        else:
            self.embedding_service = embedding_service or EmbeddingService()
            self.vector_store = vector_store or VectorStore()
            self.indexer = Indexer(
                embedding_service=self.embedding_service,
                vector_store=self.vector_store,
            )

    async def ingest_document(
        self,
        file_path: str | Path,
        document_id: str,
    ) -> IndexingResponse:
        """Extract, chunk, embed, and store one PDF document."""
        result = await self.indexer.index_pdf(
            pdf_path=file_path,
            document_id=document_id,
        )
        logger.info(
            "Document ingested document_id=%s chunks=%d",
            result["document_id"],
            result["chunks_created"],
        )
        return IndexingResponse.model_validate(result)

    async def query(self, request: SearchRequest) -> SearchResponse:
        """Return the most relevant chunks for a user question."""
        question = request.question.strip()
        if not question:
            raise ValueError("question must not be empty")

        started_at = perf_counter()
        query_embedding = await self.embedding_service.embed_text(question)
        raw_results = await self.vector_store.search(
            query_embedding=query_embedding,
            k=request.k,
        )
        results = [
            SearchResult.model_validate(result)
            for result in raw_results
        ]

        response = SearchResponse(
            question=question,
            results=results,
        )
        logger.info(
            "Query completed results=%d k=%d duration_ms=%.2f",
            len(results),
            request.k,
            (perf_counter() - started_at) * 1000,
        )
        return response

    async def delete_document(self, document_id: str) -> None:
        """Remove all indexed chunks belonging to one document."""
        if not document_id.strip():
            raise ValueError("document_id must not be empty")

        await self.vector_store.delete_document(document_id)
