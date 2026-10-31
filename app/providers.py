"""External model provider adapters."""

from typing import Protocol, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.documents import RetrievedChunk
from app.errors import ConfigurationError, StructuredOutputError
from app.prompts import (
    ANALYZE_SYSTEM_PROMPT,
    GENERATE_SYSTEM_PROMPT,
    GRADE_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
)
from app.rag import (
    AnswerDraft,
    QueryAnalysis,
    QueryRewrite,
    RelevanceGrade,
)

StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


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


class GeminiQueryModel:
    """Structured Gemini calls for each reasoning responsibility in the graph."""

    def __init__(self, settings: Settings) -> None:
        if not settings.provider_configured or settings.google_api_key is None:
            raise ConfigurationError("GOOGLE_API_KEY is required for query operations")
        model = ChatGoogleGenerativeAI(
            model=settings.chat_model,
            api_key=settings.google_api_key,
            temperature=0,
            retries=0,
            request_timeout=settings.provider_timeout_seconds,
        )
        self._analyzer = model.with_structured_output(QueryAnalysis)
        self._grader = model.with_structured_output(RelevanceGrade)
        self._rewriter = model.with_structured_output(QueryRewrite)
        self._generator = model.with_structured_output(AnswerDraft)

    def analyze(self, question: str) -> QueryAnalysis:
        return self._invoke(
            self._analyzer,
            QueryAnalysis,
            ANALYZE_SYSTEM_PROMPT,
            f"Original question:\n{question}",
        )

    def grade(self, question: str, chunk: RetrievedChunk) -> RelevanceGrade:
        return self._invoke(
            self._grader,
            RelevanceGrade,
            GRADE_SYSTEM_PROMPT,
            (
                f"Original question:\n{question}\n\n"
                f"Chunk ID: {chunk.chunk_id}\n"
                f"Source title: {chunk.title}\n"
                f"Heading: {chunk.heading}\n"
                f"Chunk content:\n{chunk.text}"
            ),
        )

    def rewrite(
        self,
        question: str,
        current_query: str,
        attempted_queries: list[str],
    ) -> QueryRewrite:
        attempts = "\n".join(f"- {query}" for query in attempted_queries)
        return self._invoke(
            self._rewriter,
            QueryRewrite,
            REWRITE_SYSTEM_PROMPT,
            (
                f"Original question:\n{question}\n\n"
                f"Current failed query:\n{current_query}\n\n"
                f"Queries already attempted:\n{attempts}"
            ),
        )

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> AnswerDraft:
        context_parts: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            source = (
                f"SOURCE_ID: S{index}\n"
                f"Title: {chunk.title}\n"
                f"Heading: {chunk.heading}\n"
                f"Content:\n{chunk.text}"
            )
            context_parts.append(source)
        context = "\n\n---\n\n".join(context_parts)
        return self._invoke(
            self._generator,
            AnswerDraft,
            GENERATE_SYSTEM_PROMPT,
            f"Question:\n{question}\n\nApproved sources:\n{context}",
        )

    @staticmethod
    def _invoke(
        runnable: object,
        output_type: type[StructuredOutput],
        system_prompt: str,
        human_prompt: str,
    ) -> StructuredOutput:
        try:
            result = runnable.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt),
                ]
            )
        except (OutputParserException, ValidationError) as exc:
            raise StructuredOutputError(
                f"Gemini response did not match {output_type.__name__}"
            ) from exc
        if not isinstance(result, output_type):
            raise StructuredOutputError(
                f"Gemini returned {type(result).__name__}; expected {output_type.__name__}"
            )
        return result
