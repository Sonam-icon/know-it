"""FastAPI routes and translation between HTTP and application contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Protocol
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from google.genai.errors import APIError
from starlette.concurrency import run_in_threadpool

from app.documents import IndexedDocument, IngestionResult
from app.errors import (
    AnswerNotFoundError,
    CitationValidationError,
    ConfigurationError,
    DocumentLoadError,
    DocumentTooLargeError,
    EmptyDocumentError,
    IndexConfigurationError,
    QueryValidationError,
    StructuredOutputError,
    UnsafeUrlError,
    UnsupportedDocumentError,
)
from app.rag import QueryResult
from app.reliability import is_transient_error
from app.repository import SQLiteRepository
from app.schemas import (
    DocumentsResponse,
    DocumentSummary,
    FeedbackRequest,
    FeedbackResponse,
    IngestionStatus,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    QueryStatus,
    SourceReference,
)

router = APIRouter()


class QueryRunner(Protocol):
    def run(self, question: str) -> QueryResult: ...


class IngestionRunner(Protocol):
    def ingest_file(self, filename: str, data: bytes) -> IngestionResult: ...

    def ingest_url(self, url: str) -> IngestionResult: ...


class DocumentCatalog(Protocol):
    def list_documents(self) -> list[IndexedDocument]: ...


@dataclass(frozen=True, slots=True)
class ApplicationServices:
    query: QueryRunner | None
    ingestion: IngestionRunner | None
    documents: DocumentCatalog
    repository: SQLiteRepository


def get_services(request: Request) -> ApplicationServices:
    return request.app.state.services


Services = Annotated[ApplicationServices, Depends(get_services)]


@router.post("/query", response_model=QueryResponse, tags=["query"])
def query_documents(payload: QueryRequest, services: Services) -> QueryResponse:
    if services.query is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini is not configured; set GOOGLE_API_KEY",
        )
    try:
        result = services.query.run(payload.question)
        answer_id = services.repository.save_answer(payload.question, result)
    except Exception as exc:
        _raise_application_error(exc)
        raise
    return _query_response(answer_id, result)


@router.post("/ingest", response_model=IngestResponse, tags=["documents"])
async def ingest_document(
    request: Request,
    services: Services,
    file: Annotated[UploadFile | None, File()] = None,
    url: Annotated[str | None, Form()] = None,
) -> IngestResponse:
    if services.ingestion is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini is not configured; set GOOGLE_API_KEY",
        )
    normalized_url = url.strip() if url else None
    if (file is None) == (normalized_url is None):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Provide exactly one file or URL",
        )

    try:
        if file is not None:
            if not file.filename:
                raise DocumentLoadError("Uploaded file must include a filename")
            limit = request.app.state.settings.max_upload_bytes
            data = await file.read(limit + 1)
            if len(data) > limit:
                raise DocumentTooLargeError(f"File exceeds the {limit}-byte upload limit")
            result = await run_in_threadpool(
                services.ingestion.ingest_file,
                file.filename,
                data,
            )
        else:
            assert normalized_url is not None
            result = await run_in_threadpool(
                services.ingestion.ingest_url,
                normalized_url,
            )
    except Exception as exc:
        _raise_application_error(exc)
        raise
    finally:
        if file is not None:
            await file.close()
    return _ingest_response(result)


@router.get("/documents", response_model=DocumentsResponse, tags=["documents"])
def list_documents(services: Services) -> DocumentsResponse:
    try:
        documents = services.documents.list_documents()
    except Exception as exc:
        _raise_application_error(exc)
        raise
    summaries = [
        DocumentSummary(
            document_id=document.document_id,
            title=document.title,
            source=document.source,
            source_kind=document.source_kind.value,
            chunk_count=document.chunk_count,
            content_hash=document.content_hash,
        )
        for document in documents
    ]
    return DocumentsResponse(documents=summaries, total=len(summaries))


@router.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
def submit_feedback(payload: FeedbackRequest, services: Services) -> FeedbackResponse:
    try:
        feedback_id = services.repository.add_feedback(
            answer_id=payload.answer_id,
            rating=payload.rating,
            comment=payload.comment or None,
        )
    except Exception as exc:
        _raise_application_error(exc)
        raise
    return FeedbackResponse(
        feedback_id=feedback_id,
        answer_id=payload.answer_id,
    )


def _query_response(answer_id: UUID, result: QueryResult) -> QueryResponse:
    return QueryResponse(
        answer_id=answer_id,
        answer=result.answer,
        status=QueryStatus(result.status.value),
        sources=[
            SourceReference(
                citation_id=source.citation_id,
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                title=source.title,
                heading=source.heading,
                source=source.source,
                source_kind=source.source_kind.value,
                excerpt=source.excerpt,
            )
            for source in result.sources
        ],
        rewrite_count=result.rewrite_count,
    )


def _ingest_response(result: IngestionResult) -> IngestResponse:
    document = result.document
    return IngestResponse(
        document_id=document.document_id,
        title=document.title,
        source=document.source,
        source_kind=document.source_kind.value,
        chunk_count=document.chunk_count,
        status=IngestionStatus(result.status.value),
    )


def _raise_application_error(exc: Exception) -> None:
    if isinstance(exc, AnswerNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if isinstance(exc, DocumentTooLargeError):
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc),
        ) from exc
    if isinstance(exc, UnsupportedDocumentError):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    if isinstance(
        exc,
        (UnsafeUrlError, EmptyDocumentError, DocumentLoadError, QueryValidationError),
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    if isinstance(exc, (ConfigurationError, IndexConfigurationError)):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    if isinstance(exc, (StructuredOutputError, CitationValidationError)):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    if is_transient_error(exc):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="A temporary external service failure exhausted its retry limit",
        ) from exc
    if isinstance(exc, APIError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The model provider rejected the request",
        ) from exc
