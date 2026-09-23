from typing import Any

from fastapi.testclient import TestClient

from app.dependencies import get_rag_service
from app.main import app
from models.schemas import IndexingResponse, SearchRequest, SearchResponse


class FakeRAGService:
    async def ingest_document(
        self,
        file_path: str,
        document_id: str,
    ) -> IndexingResponse:
        return IndexingResponse(
            document_id=document_id,
            chunks_created=1,
            embedding_ids=["embedding-1"],
        )

    async def query(self, request: SearchRequest) -> SearchResponse:
        return SearchResponse(question=request.question, results=[])

    async def delete_document(self, document_id: str) -> None:
        return None


def test_upload_query_and_delete_endpoints() -> None:
    app.dependency_overrides[get_rag_service] = FakeRAGService
    client = TestClient(app)

    try:
        upload_response = client.post(
            "/documents/upload",
            files={"file": ("document.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert upload_response.status_code == 201
        document_id = upload_response.json()["document_id"]

        query_response = client.post(
            "/query",
            json={"question": "What is this about?", "k": 3},
        )
        assert query_response.status_code == 200
        assert query_response.json() == {
            "question": "What is this about?",
            "results": [],
        }

        delete_response = client.delete(f"/documents/{document_id}")
        assert delete_response.status_code == 204
    finally:
        app.dependency_overrides.clear()


def test_upload_rejects_non_pdf_files() -> None:
    app.dependency_overrides[get_rag_service] = FakeRAGService
    client = TestClient(app)

    try:
        response = client.post(
            "/documents/upload",
            files={"file": ("document.txt", b"not a PDF", "text/plain")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 415
    assert response.json()["detail"] == "Only PDF files are supported"
