from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.dependencies import get_rag_service
from models.schemas import IndexingResponse
from services.rag_service import RAGService

router = APIRouter(prefix="/documents", tags=["documents"])


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
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported",
        )

    document_id = str(uuid4())
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            suffix=".pdf",
            prefix="mini-rag-",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            while content := await file.read(1024 * 1024):
                temporary_file.write(content)

        return await rag_service.ingest_document(
            file_path=temporary_path,
            document_id=document_id,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
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
    try:
        await rag_service.delete_document(document_id)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
