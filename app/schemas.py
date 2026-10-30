"""Public API contracts shared by routes and application services."""

from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, StringConstraints

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ApiModel(BaseModel):
    """Base model with strict, predictable request validation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class QueryStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class QueryType(StrEnum):
    CONCEPTUAL = "conceptual"
    HOW_TO = "how_to"
    TROUBLESHOOTING = "troubleshooting"
    API_REFERENCE = "api_reference"


class IngestionStatus(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


class FeedbackRating(StrEnum):
    UP = "up"
    DOWN = "down"


class QueryRequest(ApiModel):
    question: NonEmptyText = Field(max_length=2000, examples=["How do FastAPI dependencies work?"])


class SourceReference(ApiModel):
    chunk_id: NonEmptyText
    document_id: NonEmptyText
    title: NonEmptyText
    heading: str | None = None
    url: AnyHttpUrl | None = None
    excerpt: NonEmptyText


class QueryResponse(ApiModel):
    answer_id: UUID
    answer: NonEmptyText
    status: QueryStatus
    sources: list[SourceReference] = Field(default_factory=list)
    rewrite_count: int = Field(ge=0)


class IngestUrlRequest(ApiModel):
    url: AnyHttpUrl


class IngestResponse(ApiModel):
    document_id: NonEmptyText
    title: NonEmptyText
    source: NonEmptyText
    chunk_count: int = Field(ge=0)
    status: IngestionStatus


class DocumentSummary(ApiModel):
    document_id: NonEmptyText
    title: NonEmptyText
    source: NonEmptyText
    chunk_count: int = Field(ge=0)
    content_hash: NonEmptyText


class DocumentsResponse(ApiModel):
    documents: list[DocumentSummary] = Field(default_factory=list)
    total: int = Field(ge=0)


class FeedbackRequest(ApiModel):
    answer_id: UUID
    rating: FeedbackRating
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(ApiModel):
    feedback_id: UUID
    answer_id: UUID
    accepted: bool = True


class HealthResponse(ApiModel):
    status: str = "ok"
    version: NonEmptyText
    provider_configured: bool
