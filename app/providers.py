"""External model provider adapters."""

from typing import Protocol

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import Settings
from app.errors import ConfigurationError


class EmbeddingProvider(Protocol):
    model_name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbeddingProvider:
    """Gemini embeddings configured separately for documents and queries."""

    def __init__(self, settings: Settings) -> None:
        if not settings.provider_configured or settings.google_api_key is None:
            raise ConfigurationError("GOOGLE_API_KEY is required for embedding operations")
        self.model_name = settings.embedding_model
        self.dimension = settings.embedding_dimension
        common = {
            "model": settings.embedding_model,
            "api_key": settings.google_api_key,
            "output_dimensionality": settings.embedding_dimension,
        }
        self._documents = GoogleGenerativeAIEmbeddings(
            **common,
            task_type="RETRIEVAL_DOCUMENT",
        )
        self._queries = GoogleGenerativeAIEmbeddings(
            **common,
            task_type="RETRIEVAL_QUERY",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._documents.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._queries.embed_query(text)
