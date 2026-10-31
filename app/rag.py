"""Domain contracts for the retrieval and generation workflow."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.documents import RetrievedChunk, SourceKind


class QueryKind(StrEnum):
    CONCEPTUAL = "conceptual"
    HOW_TO = "how_to"
    TROUBLESHOOTING = "troubleshooting"
    API_REFERENCE = "api_reference"


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class StructuredModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class QueryAnalysis(StructuredModel):
    search_query: str = Field(min_length=1, max_length=2000)
    query_type: QueryKind


class RelevanceGrade(StructuredModel):
    relevant: bool
    reason: str = Field(min_length=1, max_length=500)


class QueryRewrite(StructuredModel):
    search_query: str = Field(min_length=1, max_length=2000)


class AnswerDraft(StructuredModel):
    answer: str = Field(min_length=1)
    citation_ids: list[str] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class GradedChunk:
    chunk_id: str
    relevant: bool
    reason: str


@dataclass(frozen=True, slots=True)
class AnswerSource:
    citation_id: str
    chunk_id: str
    document_id: str
    title: str
    heading: str
    source: str
    source_kind: SourceKind
    excerpt: str


@dataclass(frozen=True, slots=True)
class QueryResult:
    answer: str
    status: AnswerStatus
    sources: tuple[AnswerSource, ...]
    rewrite_count: int


class QueryModel(Protocol):
    def analyze(self, question: str) -> QueryAnalysis: ...

    def grade(self, question: str, chunk: RetrievedChunk) -> RelevanceGrade: ...

    def rewrite(
        self,
        question: str,
        current_query: str,
        attempted_queries: list[str],
    ) -> QueryRewrite: ...

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> AnswerDraft: ...


RouteAfterGrade = Literal["generate", "rewrite", "abstain"]
RouteAfterRewrite = Literal["retrieve", "abstain"]
