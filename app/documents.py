"""Domain models used by document ingestion and retrieval."""

from dataclasses import dataclass
from enum import StrEnum


class SourceKind(StrEnum):
    FILE = "file"
    URL = "url"


class IndexingStatus(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    document_id: str
    title: str
    source: str
    source_kind: SourceKind
    text: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    title: str
    source: str
    source_kind: SourceKind
    heading: str
    index: int
    text: str
    content_hash: str

    def metadata(self) -> dict[str, str | int]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "source": self.source,
            "source_kind": self.source_kind.value,
            "heading": self.heading,
            "chunk_index": self.index,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True, slots=True)
class IndexedDocument:
    document_id: str
    title: str
    source: str
    source_kind: SourceKind
    chunk_count: int
    content_hash: str


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document: IndexedDocument
    status: IndexingStatus
