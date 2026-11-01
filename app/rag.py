"""Domain contracts for the retrieval and generation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.documents import RetrievedChunk, SourceKind, SourceOrigin


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class StructuredModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class QueryAnalysis(StructuredModel):
    standalone_question: str = Field(min_length=1, max_length=2000)
    search_query: str = Field(min_length=1, max_length=2000)


class RelevanceGrade(StructuredModel):
    relevant: bool
    reason: str = Field(min_length=1, max_length=500)


class QueryRewrite(StructuredModel):
    search_query: str = Field(min_length=1, max_length=2000)


class AnswerDraft(StructuredModel):
    answer: str = Field(
        min_length=1,
        description=(
            "Detailed technical answer. You MUST include inline citation markers such as "
            "[S1] or [S2] immediately after each factual claim."
        ),
    )
    citation_ids: list[str] = Field(
        min_length=1,
        description=(
            "List of unique source IDs (e.g. ['S1']) that appear as inline markers in the answer."
        ),
    )


class GroundednessVerdict(StructuredModel):
    supported: bool
    unsupported_claims: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def require_consistent_verdict(self) -> GroundednessVerdict:
        if self.supported and self.unsupported_claims:
            raise ValueError("A supported answer cannot contain unsupported claims")
        if not self.supported and not self.unsupported_claims:
            raise ValueError("An unsupported answer must identify at least one claim")
        return self


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    question: str
    answer: str


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
    origin: SourceOrigin = SourceOrigin.CORPUS


@dataclass(frozen=True, slots=True)
class QueryResult:
    answer: str
    status: AnswerStatus
    sources: tuple[AnswerSource, ...]
    rewrite_count: int
    web_attempted: bool = False


class QueryModel(Protocol):
    def analyze(
        self,
        question: str,
        history: tuple[ConversationTurn, ...],
    ) -> QueryAnalysis: ...

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
        validation_feedback: str | None = None,
    ) -> AnswerDraft: ...

    def verify(
        self,
        question: str,
        draft: AnswerDraft,
        chunks: list[RetrievedChunk],
    ) -> GroundednessVerdict: ...

    def revise(
        self,
        question: str,
        draft: AnswerDraft,
        chunks: list[RetrievedChunk],
        unsupported_claims: list[str],
    ) -> AnswerDraft: ...


RouteAfterGrade = Literal["generate", "rewrite", "search_web", "abstain"]
RouteAfterRewrite = Literal["retrieve", "search_web", "abstain"]
RouteAfterVerify = Literal["finalize", "revise", "abstain"]
