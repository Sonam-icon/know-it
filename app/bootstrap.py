"""Construct application services from validated settings."""

from app.chunking import TechnicalDocumentChunker
from app.config import Settings
from app.ingestion import IngestionService
from app.loaders import DocumentLoader
from app.providers import GeminiEmbeddingProvider
from app.vector_store import ChromaDocumentStore


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
