import asyncio
from pathlib import Path
from typing import Any

from core.indexer import Indexer
from models.schemas import SearchRequest
from services.rag_service import RAGService


def run(coroutine: object) -> object:
    return asyncio.run(coroutine)  # type: ignore[arg-type]


class FakePDFProcessor:
    def extract_with_metadata(self, pdf_path: str | Path) -> dict[str, Any]:
        return {
            "text": "unused",
            "pages": [{"page_number": 2, "text": "alpha beta"}],
            "metadata": {"Title": "Fake PDF"},
        }


class FakeEmbeddingService:
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1)] for index, _ in enumerate(texts)]

    async def embed_text(self, text: str) -> list[float]:
        return [1.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self.embeddings: list[list[float]] = []
        self.metadata: list[dict[str, Any]] = []
        self.deleted_document_id: str | None = None

    async def add_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> list[str]:
        self.embeddings = embeddings
        self.metadata = metadata
        return [f"embedding-{index}" for index in range(len(embeddings))]

    async def search(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[dict[str, Any]]:
        return [
            {
                "id": "embedding-0",
                "score": 1.0,
                "metadata": self.metadata[0],
            }
        ][:k] if self.metadata else []

    async def delete_document(self, document_id: str) -> None:
        self.deleted_document_id = document_id


def test_indexer_connects_all_indexing_steps() -> None:
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    indexer = Indexer(
        pdf_processor=FakePDFProcessor(),
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    result = run(indexer.index_pdf("document.pdf", "doc-1"))

    assert result["document_id"] == "doc-1"
    assert result["chunks_created"] == 1
    assert result["embedding_ids"] == ["embedding-0"]
    assert vector_store.metadata[0]["document_id"] == "doc-1"
    assert vector_store.metadata[0]["page_number"] == 2


def test_rag_service_queries_and_deletes_documents() -> None:
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    indexer = Indexer(
        pdf_processor=FakePDFProcessor(),
        embedding_service=embedding_service,
        vector_store=vector_store,
    )
    service = RAGService(indexer=indexer)
    run(service.indexer.index_pdf("document.pdf", "doc-1"))

    response = run(service.query(SearchRequest(question="alpha", k=1)))
    run(service.delete_document("doc-1"))

    assert response.question == "alpha"
    assert len(response.results) == 1
    assert response.results[0].metadata.document_id == "doc-1"
    assert vector_store.deleted_document_id == "doc-1"
