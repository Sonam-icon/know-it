import hashlib

import chromadb

from app.chunking import TechnicalDocumentChunker
from app.documents import IndexingStatus
from app.ingestion import IngestionService
from app.loaders import DocumentLoader
from app.vector_store import ChromaDocumentStore


class FakeEmbeddings:
    model_name = "fake-embeddings"
    dimension = 3

    def __init__(self) -> None:
        self.document_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [float(digest[index]) / 255 for index in range(3)]


def _service() -> tuple[IngestionService, FakeEmbeddings, ChromaDocumentStore]:
    embeddings = FakeEmbeddings()
    store = ChromaDocumentStore(
        collection_name="test_documents",
        embedding_model=embeddings.model_name,
        embedding_dimension=embeddings.dimension,
        client=chromadb.EphemeralClient(),
    )
    service = IngestionService(
        loader=DocumentLoader(
            max_upload_bytes=20_000,
            max_fetch_bytes=20_000,
            fetch_timeout_seconds=1,
            max_fetch_redirects=0,
        ),
        chunker=TechnicalDocumentChunker(chunk_size_tokens=40, chunk_overlap_tokens=5),
        embeddings=embeddings,
        store=store,
    )
    return service, embeddings, store


def test_ingestion_is_idempotent_and_skips_duplicate_embedding() -> None:
    service, embeddings, _ = _service()
    content = b"# Guide\n\n" + b" ".join([b"documentation"] * 100)

    created = service.ingest_file("guide.md", content)
    unchanged = service.ingest_file("guide.md", content)

    assert created.status is IndexingStatus.CREATED
    assert unchanged.status is IndexingStatus.UNCHANGED
    assert created.document.chunk_count == unchanged.document.chunk_count
    assert embeddings.document_calls == 1


def test_changed_document_replaces_old_chunk_version() -> None:
    service, _, store = _service()
    original = b"# Guide\n\n" + b" ".join([b"documentation"] * 100)
    updated = b"# Guide\n\nShort replacement."

    first = service.ingest_file("guide.md", original)
    second = service.ingest_file("guide.md", updated)

    assert second.status is IndexingStatus.UPDATED
    assert second.document.chunk_count < first.document.chunk_count
    stored_ids = store.collection.get(
        where={"document_id": second.document.document_id},
        include=[],
    )["ids"]
    assert len(stored_ids) == second.document.chunk_count
