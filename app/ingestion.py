"""Document ingestion use cases."""

from app.chunking import TechnicalDocumentChunker
from app.documents import IndexedDocument, IndexingStatus, IngestionResult, LoadedDocument
from app.errors import EmptyDocumentError
from app.loaders import DocumentLoader
from app.providers import EmbeddingProvider
from app.vector_store import ChromaDocumentStore


class IngestionService:
    """Coordinate loading, chunking, embedding, and safe document replacement."""

    def __init__(
        self,
        *,
        loader: DocumentLoader,
        chunker: TechnicalDocumentChunker,
        embeddings: EmbeddingProvider,
        store: ChromaDocumentStore,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.embeddings = embeddings
        self.store = store

    def ingest_file(self, filename: str, data: bytes) -> IngestionResult:
        return self.ingest(self.loader.load_file(filename, data))

    def ingest_url(self, url: str) -> IngestionResult:
        return self.ingest(self.loader.load_url(url))

    def ingest(self, document: LoadedDocument) -> IngestionResult:
        previous_hash = self.store.content_hash_for(document.document_id)
        if previous_hash == document.content_hash:
            indexed = self._find_document(document.document_id)
            return IngestionResult(document=indexed, status=IndexingStatus.UNCHANGED)

        chunks = self.chunker.split(document)
        if not chunks:
            raise EmptyDocumentError(f"No chunks were produced for {document.source}")
        vectors = self.embeddings.embed_documents([chunk.text for chunk in chunks])
        self.store.replace_document(chunks, vectors)
        indexed = self._find_document(document.document_id)
        status = IndexingStatus.UPDATED if previous_hash else IndexingStatus.CREATED
        return IngestionResult(document=indexed, status=status)

    def list_documents(self) -> list[IndexedDocument]:
        return self.store.list_documents()

    def _find_document(self, document_id: str) -> IndexedDocument:
        return next(
            document
            for document in self.store.list_documents()
            if document.document_id == document_id
        )
