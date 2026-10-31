"""Construct application services from validated settings."""

from dataclasses import dataclass

from app.chunking import TechnicalDocumentChunker
from app.config import Settings
from app.ingestion import IngestionService
from app.loaders import DocumentLoader
from app.providers import GeminiEmbeddingProvider, GeminiQueryModel
from app.query_workflow import QueryWorkflow, build_query_workflow
from app.reliability import build_retry_policy
from app.vector_store import ChromaDocumentStore
from app.web_search import TavilySearcher


@dataclass(frozen=True, slots=True)
class RuntimeServices:
    query: QueryWorkflow
    ingestion: IngestionService
    documents: ChromaDocumentStore


def build_document_loader(settings: Settings) -> DocumentLoader:
    return DocumentLoader(
        max_upload_bytes=settings.max_upload_bytes,
        max_fetch_bytes=settings.max_fetch_bytes,
        fetch_timeout_seconds=settings.fetch_timeout_seconds,
        max_fetch_redirects=settings.max_fetch_redirects,
    )


def build_document_chunker(settings: Settings) -> TechnicalDocumentChunker:
    return TechnicalDocumentChunker(
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )


def build_document_store(
    settings: Settings,
    *,
    embedding_model: str | None = None,
    embedding_dimension: int | None = None,
) -> ChromaDocumentStore:
    return ChromaDocumentStore(
        collection_name=settings.chroma_collection,
        embedding_model=embedding_model or settings.embedding_model,
        embedding_dimension=embedding_dimension or settings.embedding_dimension,
        persist_path=settings.chroma_path,
    )


def build_ingestion_service(settings: Settings) -> IngestionService:
    embeddings = GeminiEmbeddingProvider(settings)
    return IngestionService(
        loader=build_document_loader(settings),
        chunker=build_document_chunker(settings),
        embeddings=embeddings,
        store=build_document_store(
            settings,
            embedding_model=embeddings.model_name,
            embedding_dimension=embeddings.dimension,
        ),
    )


def build_query_service(settings: Settings) -> QueryWorkflow:
    embeddings = GeminiEmbeddingProvider(settings)
    return build_query_workflow(
        model=GeminiQueryModel(settings),
        embeddings=embeddings,
        retriever=build_document_store(
            settings,
            embedding_model=embeddings.model_name,
            embedding_dimension=embeddings.dimension,
        ),
        top_k=settings.retrieval_top_k,
        max_query_rewrites=settings.max_query_rewrites,
        max_context_characters=settings.max_context_characters,
        max_question_length=settings.max_question_length,
        max_answer_revisions=settings.max_answer_revisions,
        web_searcher=TavilySearcher(settings) if settings.web_fallback_enabled else None,
        web_top_k=settings.web_search_top_k,
        retry_policy=build_retry_policy(settings),
    )


def build_runtime_services(
    settings: Settings,
    *,
    store: ChromaDocumentStore | None = None,
) -> RuntimeServices:
    embeddings = GeminiEmbeddingProvider(settings)
    document_store = store or build_document_store(
        settings,
        embedding_model=embeddings.model_name,
        embedding_dimension=embeddings.dimension,
    )
    ingestion = IngestionService(
        loader=build_document_loader(settings),
        chunker=build_document_chunker(settings),
        embeddings=embeddings,
        store=document_store,
    )
    query = build_query_workflow(
        model=GeminiQueryModel(settings),
        embeddings=embeddings,
        retriever=document_store,
        top_k=settings.retrieval_top_k,
        max_query_rewrites=settings.max_query_rewrites,
        max_context_characters=settings.max_context_characters,
        max_question_length=settings.max_question_length,
        max_answer_revisions=settings.max_answer_revisions,
        web_searcher=TavilySearcher(settings) if settings.web_fallback_enabled else None,
        web_top_k=settings.web_search_top_k,
        retry_policy=build_retry_policy(settings),
    )
    return RuntimeServices(
        query=query,
        ingestion=ingestion,
        documents=document_store,
    )
