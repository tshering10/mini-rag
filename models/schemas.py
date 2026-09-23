from typing import Any

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata stored alongside an indexed text chunk."""

    chunk_id: str
    chunk_index: int = Field(ge=0)
    document_id: str
    text: str
    page_number: int | None = Field(default=None, ge=1)


class IndexingResponse(BaseModel):
    """Summary returned after a document has been indexed."""

    document_id: str
    chunks_created: int = Field(ge=0)
    embedding_ids: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    """Input accepted by the semantic-search endpoint."""

    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    """One vector-search result and its source chunk."""

    id: str
    score: float
    metadata: ChunkMetadata


class SearchResponse(BaseModel):
    """Results returned by semantic search."""

    question: str
    results: list[SearchResult]


class RAGResponse(BaseModel):
    """Future answer response combining an LLM answer and its sources."""

    answer: str
    source_chunks: list[SearchResult]
    confidence: float | None = Field(default=None, ge=0, le=1)
    model_used: str | None = None
    query_time_ms: float | None = Field(default=None, ge=0)
