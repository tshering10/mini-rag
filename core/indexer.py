import logging
from pathlib import Path
from typing import Any

from core.chunker import TextChunker
from core.embeddings import EmbeddingService
from core.pdf_processor import PDFProcessor
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Indexer:
    """Coordinate PDF extraction, chunking, embedding, and storage."""

    def __init__(
        self,
        pdf_processor: PDFProcessor | None = None,
        chunker: TextChunker | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.pdf_processor = pdf_processor or PDFProcessor()
        self.chunker = chunker or TextChunker()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()

    async def index_pdf(
        self,
        pdf_path: str | Path,
        document_id: str,
    ) -> dict[str, Any]:
        """Index a PDF and return a summary of the stored chunks."""
        if not document_id.strip():
            raise ValueError("document_id must not be empty")

        document = self.pdf_processor.extract_with_metadata(pdf_path)
        chunks: list[dict[str, Any]] = []

        for page in document["pages"]:
            page_chunks = self.chunker.chunk_text(
                page["text"],
                page_number=page["page_number"],
            )

            for chunk in page_chunks:
                chunk["document_id"] = document_id
                chunks.append(chunk)

        texts = [chunk["text"] for chunk in chunks]
        embeddings = await self.embedding_service.embed_batch(texts)
        embedding_ids = await self.vector_store.add_embeddings(
            embeddings=embeddings,
            metadata=chunks,
        )

        logger.info(
            "PDF indexed document_id=%s pages=%d chunks=%d",
            document_id,
            len(document["pages"]),
            len(chunks),
        )
        return {
            "document_id": document_id,
            "chunks_created": len(chunks),
            "embedding_ids": embedding_ids,
            "metadata": document["metadata"],
        }