"""Chroma-backed document index with deterministic replacement semantics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from app.documents import DocumentChunk, IndexedDocument, SourceKind
from app.errors import IndexConfigurationError

_SCHEMA_VERSION = 1


class ChromaDocumentStore:
    """Store pre-computed embeddings and source metadata in Chroma."""

    def __init__(
        self,
        *,
        collection_name: str,
        embedding_model: str,
        embedding_dimension: int,
        persist_path: Path | None = None,
        client: ClientAPI | None = None,
    ) -> None:
        if client is None:
            if persist_path is None:
                raise ValueError("persist_path is required when no Chroma client is supplied")
            client = chromadb.PersistentClient(path=persist_path)
        self.client = client
        expected_metadata: dict[str, str | int] = {
            "schema_version": _SCHEMA_VERSION,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
        }
        self.collection: Collection = client.get_or_create_collection(
            name=collection_name,
            metadata=expected_metadata,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )
        self._validate_collection_metadata(expected_metadata)

    def _validate_collection_metadata(self, expected: dict[str, str | int]) -> None:
        actual = self.collection.metadata or {}
        mismatches = {
            key: (actual.get(key), expected_value)
            for key, expected_value in expected.items()
            if actual.get(key) != expected_value
        }
        if mismatches:
            details = ", ".join(
                f"{key}: index={actual_value!r}, configured={expected_value!r}"
                for key, (actual_value, expected_value) in mismatches.items()
            )
            raise IndexConfigurationError(
                f"Chroma collection is incompatible with current settings ({details})"
            )

    def content_hash_for(self, document_id: str) -> str | None:
        result = self.collection.get(
            where={"document_id": document_id},
            limit=1,
            include=["metadatas"],
        )
        metadatas = result.get("metadatas") or []
        return str(metadatas[0]["content_hash"]) if metadatas else None

    def replace_document(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")
        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk must have exactly one embedding")
        document_ids = {chunk.document_id for chunk in chunks}
        if len(document_ids) != 1:
            raise ValueError("All chunks must belong to the same document")

        document_id = chunks[0].document_id
        existing = self.collection.get(
            where={"document_id": document_id},
            include=[],
        )
        previous_ids = set(existing.get("ids") or [])
        new_ids = [chunk.chunk_id for chunk in chunks]
        self.collection.upsert(
            ids=new_ids,
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata() for chunk in chunks],
            embeddings=[list(embedding) for embedding in embeddings],
        )
        stale_ids = sorted(previous_ids - set(new_ids))
        if stale_ids:
            self.collection.delete(ids=stale_ids)

    def list_documents(self) -> list[IndexedDocument]:
        result = self.collection.get(include=["metadatas"])
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for metadata in result.get("metadatas") or []:
            if metadata:
                grouped[str(metadata["document_id"])].append(dict(metadata))

        documents: list[IndexedDocument] = []
        for document_id, rows in grouped.items():
            first = rows[0]
            documents.append(
                IndexedDocument(
                    document_id=document_id,
                    title=str(first["title"]),
                    source=str(first["source"]),
                    source_kind=SourceKind(str(first["source_kind"])),
                    chunk_count=len(rows),
                    content_hash=str(first["content_hash"]),
                )
            )
        return sorted(documents, key=lambda item: (item.title.lower(), item.document_id))
