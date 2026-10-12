import logging
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.core.exceptions import DocumentNotFoundError, FileTooLargeError
from app.dependencies import get_ingestion_service, get_qa_service, get_vector_store
from app.retrieval.vector_store import VectorStoreManager
from app.schemas.requests import AskRequest
from app.schemas.responses import (
    AskResponse,
    DocumentList,
    DocumentOut,
    ErrorResponse,
    HealthResponse,
    IngestFileResult,
    IngestResponse,
)
from app.services.ingestion_service import IngestionService
from app.services.qa_service import QAService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(settings: Settings = Depends(get_settings), store: VectorStoreManager = Depends(get_vector_store)):
    docs = store.list_documents()
    return HealthResponse(
        status="ok",
        documents=len(docs),
        chunks=sum(d.chunks for d in docs),
        llm_configured=bool(settings.groq_api_key),
        llm_model=settings.groq_model,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model if settings.embedding_provider == "fastembed" else "hash-bow",
    )


@router.post(
    "/documents",
    response_model=IngestResponse,
    tags=["documents"],
    summary="Upload and index PDF / TXT / Markdown files",
    responses={400: {"model": IngestResponse}},
)
def upload_documents(
    files: List[UploadFile] = File(..., description="One or more .pdf, .txt, .md files"),
    settings: Settings = Depends(get_settings),
    service: IngestionService = Depends(get_ingestion_service),
):
    """Each file is processed independently; per-file outcome is reported in ``results``.
    Returns 400 only when *every* file failed."""
    limit = settings.max_upload_mb * 1024 * 1024
    results: List[IngestFileResult] = []
    for upload in files:
        name = upload.filename or "unnamed"
        data = upload.file.read(limit + 1)  # never buffer more than limit+1 bytes
        if len(data) > limit:
            err = FileTooLargeError(f"'{name}' exceeds the {settings.max_upload_mb} MB limit.")
            results.append(IngestFileResult(file_name=name, status="failed", error=err.message, error_code=err.code))
            continue
        outcome = service.ingest_safely(name, data)
        results.append(IngestFileResult(**outcome.__dict__))

    body = IngestResponse(results=results)
    if all(r.status == "failed" for r in results):
        return JSONResponse(status_code=400, content=body.model_dump())
    return body


@router.get("/documents", response_model=DocumentList, tags=["documents"])
def list_documents(store: VectorStoreManager = Depends(get_vector_store)):
    docs = store.list_documents()
    return DocumentList(
        total_documents=len(docs),
        total_chunks=sum(d.chunks for d in docs),
        documents=[DocumentOut(file_name=d.file_name, chunks=d.chunks) for d in docs],
    )


@router.delete("/documents/{file_name}", tags=["documents"], responses={404: {"model": ErrorResponse}})
def delete_document(file_name: str, service: IngestionService = Depends(get_ingestion_service)):
    removed = service.delete(file_name)
    if removed == 0:
        raise DocumentNotFoundError(f"No document named '{file_name}' is indexed.")
    return {"file_name": file_name, "chunks_removed": removed}


@router.post(
    "/ask",
    response_model=AskResponse,
    tags=["qa"],
    summary="Ask a question; answers are grounded in the ingested documents",
    responses={502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def ask(
    request: AskRequest,
    settings: Settings = Depends(get_settings),
    service: QAService = Depends(get_qa_service),
):
    if len(request.question) > settings.max_question_chars:
        raise HTTPException(status_code=422, detail=f"question exceeds {settings.max_question_chars} characters")
    return service.answer(request.question, request.top_k)
