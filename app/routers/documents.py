import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.dependencies import get_rag_service
from models.schemas import IndexingResponse
from services.rag_service import RAGService

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)
MAX_UPLOAD_SIZE = 50 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024


@router.post(
    "/upload",
    response_model=IndexingResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service),
) -> IndexingResponse:
    """Save, index, and remove an uploaded PDF."""
    filename = file.filename or ""
    logger.info("Document upload started filename=%s", filename)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported",
        )
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="The uploaded file must have content type application/pdf",
        )

    document_id = str(uuid4())
    temporary_path: Path | None = None
    total_bytes = 0

    try:
        with NamedTemporaryFile(
            suffix=".pdf",
            prefix="mini-rag-",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            while content := await file.read(UPLOAD_CHUNK_SIZE):
                total_bytes += len(content)
                if total_bytes > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="The PDF file must not exceed 50 MB",
                    )
                temporary_file.write(content)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded PDF must not be empty",
            )

        response = await rag_service.ingest_document(
            file_path=temporary_path,
            document_id=document_id,
        )
        logger.info(
            "Document indexing completed document_id=%s chunks=%d",
            document_id,
            response.chunks_created,
        )
        return response
    finally:
        await file.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    rag_service: RAGService = Depends(get_rag_service),
) -> None:
    """Delete all indexed chunks for a document."""
    await rag_service.delete_document(document_id)
    logger.info("Document deleted document_id=%s", document_id)
