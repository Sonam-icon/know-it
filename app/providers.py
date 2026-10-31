"""External model provider adapters."""

import logging
from dataclasses import dataclass
from threading import Lock
from typing import Generic, Protocol, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, SecretStr, ValidationError

from app.config import Settings
from app.documents import RetrievedChunk
from app.errors import ConfigurationError, StructuredOutputError
from app.prompts import (
    ANALYZE_SYSTEM_PROMPT,
    GENERATE_SYSTEM_PROMPT,
    GRADE_SYSTEM_PROMPT,
    REVISE_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    VERIFY_SYSTEM_PROMPT,
)
from app.rag import (
    AnswerDraft,
    ConversationTurn,
    GroundednessVerdict,
    QueryAnalysis,
    QueryRewrite,
    RelevanceGrade,
)

StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)
PoolItem = TypeVar("PoolItem")
logger = logging.getLogger(__name__)


class RoundRobinPool(Generic[PoolItem]):
    """Select configured clients in order without holding a lock during I/O."""

    def __init__(self, items: list[PoolItem]) -> None:
        if not items:
            raise ValueError("RoundRobinPool requires at least one item")
        self._items = tuple(items)
        self._index = 0
        self._lock = Lock()

    @property
    def size(self) -> int:
        return len(self._items)

    def next(self) -> PoolItem:
        with self._lock:
            item = self._items[self._index]
            self._index = (self._index + 1) % len(self._items)
        return item


@dataclass(frozen=True, slots=True)
class _EmbeddingClients:
    documents: GoogleGenerativeAIEmbeddings
    queries: GoogleGenerativeAIEmbeddings


@dataclass(frozen=True, slots=True)
class _QueryClients:
    analyzer: object
    grader: object
    rewriter: object
    generator: object
    verifier: object
    reviser: object


class EmbeddingProvider(Protocol):
    model_name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbeddingProvider:
    """Gemini embeddings configured separately for documents and queries."""

    def __init__(self, settings: Settings) -> None:
        if not settings.provider_configured:
            raise ConfigurationError("GOOGLE_API_KEY is required for embedding operations")
        self.model_name = settings.embedding_model
        self.dimension = settings.embedding_dimension
        self._clients = RoundRobinPool(
            [
                _EmbeddingClients(
                    documents=GoogleGenerativeAIEmbeddings(
                        model=settings.embedding_model,
                        api_key=api_key,
                        output_dimensionality=settings.embedding_dimension,
                        task_type="RETRIEVAL_DOCUMENT",
                    ),
                    queries=GoogleGenerativeAIEmbeddings(
                        model=settings.embedding_model,
                        api_key=api_key,
                        output_dimensionality=settings.embedding_dimension,
                        task_type="RETRIEVAL_QUERY",
                    ),
                )
                for api_key in settings.google_api_keys
            ]
        )
        logger.info("gemini_embedding_pool_configured key_count=%s", self._clients.size)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._clients.next().documents.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._clients.next().queries.embed_query(text)


class GeminiQueryModel:
    """Structured Gemini calls for each reasoning responsibility in the graph."""

    def __init__(self, settings: Settings) -> None:
        if not settings.provider_configured:
            raise ConfigurationError("GOOGLE_API_KEY is required for query operations")
        self._clients = RoundRobinPool(
            [self._build_clients(settings, api_key) for api_key in settings.google_api_keys]
        )
        logger.info("gemini_chat_pool_configured key_count=%s", self._clients.size)

    @staticmethod
    def _build_clients(settings: Settings, api_key: SecretStr) -> _QueryClients:
        model = ChatGoogleGenerativeAI(
            model=settings.chat_model,
            api_key=api_key,
            retries=0,
            request_timeout=settings.provider_timeout_seconds,
        )
        return _QueryClients(
            analyzer=model.with_structured_output(QueryAnalysis),
            grader=model.with_structured_output(RelevanceGrade),
            rewriter=model.with_structured_output(QueryRewrite),
            generator=model.with_structured_output(AnswerDraft),
            verifier=model.with_structured_output(GroundednessVerdict),
            reviser=model.with_structured_output(AnswerDraft),
        )

    def analyze(
        self,
        question: str,
        history: tuple[ConversationTurn, ...],
    ) -> QueryAnalysis:
        history_text = "\n\n".join(
            f"User: {turn.question}\nAssistant: {turn.answer}" for turn in history
        )
        clients = self._clients.next()
        return self._invoke(
            clients.analyzer,
            QueryAnalysis,
            ANALYZE_SYSTEM_PROMPT,
            (f"Conversation history:\n{history_text or '(none)'}\n\nLatest question:\n{question}"),
        )

    def grade(self, question: str, chunk: RetrievedChunk) -> RelevanceGrade:
        clients = self._clients.next()
        return self._invoke(
            clients.grader,
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
        clients = self._clients.next()
        return self._invoke(
            clients.rewriter,
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
        validation_feedback: str | None = None,
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
        correction = ""
        if validation_feedback:
            correction = (
                "\n\nYour previous draft was rejected by application validation:\n"
                f"{validation_feedback}\n"
                "Return a fresh corrected draft. Ensure every citation ID exists in the "
                "approved sources and the inline [S#] markers exactly match citation_ids."
            )
        clients = self._clients.next()
        return self._invoke(
            clients.generator,
            AnswerDraft,
            GENERATE_SYSTEM_PROMPT,
            f"Question:\n{question}\n\nApproved sources:\n{context}{correction}",
        )

    def verify(
        self,
        question: str,
        draft: AnswerDraft,
        chunks: list[RetrievedChunk],
    ) -> GroundednessVerdict:
        context = self._format_context(chunks)
        clients = self._clients.next()
        return self._invoke(
            clients.verifier,
            GroundednessVerdict,
            VERIFY_SYSTEM_PROMPT,
            (
                f"Question:\n{question}\n\n"
                f"Draft answer:\n{draft.answer}\n\n"
                f"Approved sources:\n{context}"
            ),
        )

    def revise(
        self,
        question: str,
        draft: AnswerDraft,
        chunks: list[RetrievedChunk],
        unsupported_claims: list[str],
    ) -> AnswerDraft:
        context = self._format_context(chunks)
        problems = "\n".join(f"- {claim}" for claim in unsupported_claims)
        clients = self._clients.next()
        return self._invoke(
            clients.reviser,
            AnswerDraft,
            REVISE_SYSTEM_PROMPT,
            (
                f"Question:\n{question}\n\n"
                f"Draft answer:\n{draft.answer}\n\n"
                f"Unsupported claims:\n{problems}\n\n"
                f"Approved sources:\n{context}"
            ),
        )

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        return "\n\n---\n\n".join(
            (
                f"SOURCE_ID: S{index}\n"
                f"Title: {chunk.title}\n"
                f"Heading: {chunk.heading}\n"
                f"Content:\n{chunk.text}"
            )
            for index, chunk in enumerate(chunks, start=1)
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
