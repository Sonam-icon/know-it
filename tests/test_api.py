from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from app.api import ApplicationServices
from app.config import Settings
from app.documents import (
    IndexedDocument,
    IndexingStatus,
    IngestionResult,
    SourceKind,
)
from app.errors import UnsupportedDocumentError
from app.main import create_app
from app.rag import AnswerSource, AnswerStatus, ConversationTurn, QueryResult
from app.repository import SQLiteRepository


class FakeQueryService:
    def __init__(self) -> None:
        self.histories: list[tuple[ConversationTurn, ...]] = []

    def run(
        self,
        question: str,
        history: tuple[ConversationTurn, ...] = (),
    ) -> QueryResult:
        self.histories.append(history)
        return QueryResult(
            answer=f"Answer to: {question} [S1]",
            status=AnswerStatus.ANSWERED,
            sources=(
                AnswerSource(
                    citation_id="S1",
                    chunk_id="chunk-1",
                    document_id="document-1",
                    title="FastAPI Guide",
                    heading="Dependencies",
                    source="https://example.com/dependencies",
                    source_kind=SourceKind.URL,
                    excerpt="Use Depends to declare a dependency.",
                ),
            ),
            rewrite_count=0,
        )


class FakeIngestionService:
    def ingest_file(self, filename: str, data: bytes) -> IngestionResult:
        if filename.endswith(".pdf"):
            raise UnsupportedDocumentError("Unsupported file type .pdf")
        return _ingestion_result(filename, SourceKind.FILE, len(data))

    def ingest_url(self, url: str) -> IngestionResult:
        return _ingestion_result(url, SourceKind.URL, 2)


class FakeDocumentStore:
    def list_documents(self) -> list[IndexedDocument]:
        return [
            IndexedDocument(
                document_id="document-1",
                title="FastAPI Guide",
                source="https://example.com/guide",
                source_kind=SourceKind.URL,
                chunk_count=3,
                content_hash="content-hash",
            )
        ]


def _ingestion_result(
    source: str,
    source_kind: SourceKind,
    chunk_count: int,
) -> IngestionResult:
    return IngestionResult(
        document=IndexedDocument(
            document_id="document-new",
            title="New Guide",
            source=source,
            source_kind=source_kind,
            chunk_count=chunk_count,
            content_hash="new-content-hash",
        ),
        status=IndexingStatus.CREATED,
    )


@dataclass
class ApiContext:
    client: TestClient
    repository: SQLiteRepository
    query: FakeQueryService


@pytest.fixture
def api(tmp_path):
    repository = SQLiteRepository(tmp_path / "application.sqlite3")
    query = FakeQueryService()
    services = ApplicationServices(
        query=query,
        ingestion=FakeIngestionService(),
        documents=FakeDocumentStore(),
        repository=repository,
    )
    settings = Settings(
        _env_file=None,
        application_db_path=tmp_path / "unused.sqlite3",
        chroma_path=tmp_path / "chroma",
        max_upload_bytes=1024,
    )
    with TestClient(create_app(settings, services)) as client:
        yield ApiContext(client=client, repository=repository, query=query)


def test_query_returns_sources_and_persists_answer(api: ApiContext) -> None:
    response = api.client.post(
        "/query",
        json={"question": "How do dependencies work?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "answered"
    assert payload["session_id"]
    assert payload["sources"][0]["citation_id"] == "S1"
    assert payload["sources"][0]["origin"] == "corpus"
    stored = api.repository.get_answer(payload["answer_id"])
    assert stored is not None
    assert stored.question == "How do dependencies work?"
    assert str(stored.session_id) == payload["session_id"]


def test_query_continues_a_known_session(api: ApiContext) -> None:
    first = api.client.post("/query", json={"question": "What is Depends?"}).json()

    second = api.client.post(
        "/query",
        json={"question": "What about its default?", "session_id": first["session_id"]},
    )

    assert second.status_code == 200
    assert second.json()["session_id"] == first["session_id"]
    assert api.query.histories[-1][0].question == "What is Depends?"


def test_new_query_does_not_inherit_another_session(api: ApiContext) -> None:
    api.client.post("/query", json={"question": "First session"})

    response = api.client.post("/query", json={"question": "Independent session"})

    assert response.status_code == 200
    assert api.query.histories[-1] == ()


def test_query_rejects_an_unknown_session(api: ApiContext) -> None:
    response = api.client.post(
        "/query",
        json={
            "question": "Follow-up",
            "session_id": "d868f67d-b0bd-4fd3-a9de-42e6b9b9cfb9",
        },
    )

    assert response.status_code == 404


def test_feedback_is_linked_to_a_known_answer(api: ApiContext) -> None:
    answer_id = api.client.post(
        "/query",
        json={"question": "Question?"},
    ).json()["answer_id"]

    response = api.client.post(
        "/feedback",
        json={"answer_id": answer_id, "rating": "up", "comment": "Useful"},
    )

    assert response.status_code == 200
    assert response.json()["answer_id"] == answer_id
    assert api.repository.feedback_count(answer_id) == 1


def test_feedback_returns_not_found_for_an_unknown_answer(api: ApiContext) -> None:
    response = api.client.post(
        "/feedback",
        json={
            "answer_id": "d868f67d-b0bd-4fd3-a9de-42e6b9b9cfb9",
            "rating": "down",
        },
    )

    assert response.status_code == 404


def test_documents_lists_indexed_sources(api: ApiContext) -> None:
    response = api.client.get("/documents")

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["documents"][0]["source_kind"] == "url"


def test_ingest_accepts_either_a_file_or_url(api: ApiContext) -> None:
    file_response = api.client.post(
        "/ingest",
        files={"file": ("guide.md", b"# Guide", "text/markdown")},
    )
    url_response = api.client.post(
        "/ingest",
        data={"url": "https://example.com/guide"},
    )

    assert file_response.status_code == 200
    assert file_response.json()["source_kind"] == "file"
    assert url_response.status_code == 200
    assert url_response.json()["source_kind"] == "url"


def test_ingest_requires_exactly_one_source(api: ApiContext) -> None:
    missing = api.client.post("/ingest")
    both = api.client.post(
        "/ingest",
        data={"url": "https://example.com/guide"},
        files={"file": ("guide.md", b"# Guide", "text/markdown")},
    )

    assert missing.status_code == 422
    assert both.status_code == 422


def test_ingest_enforces_upload_size_and_maps_media_errors(api: ApiContext) -> None:
    oversized = api.client.post(
        "/ingest",
        files={"file": ("guide.md", b"x" * 1025, "text/markdown")},
    )
    unsupported = api.client.post(
        "/ingest",
        files={"file": ("guide.pdf", b"pdf", "application/pdf")},
    )

    assert oversized.status_code == 413
    assert unsupported.status_code == 415


def test_provider_routes_are_unavailable_without_credentials(tmp_path) -> None:
    repository = SQLiteRepository(tmp_path / "application.sqlite3")
    services = ApplicationServices(
        query=None,
        ingestion=None,
        documents=FakeDocumentStore(),
        repository=repository,
    )
    settings = Settings(_env_file=None, chroma_path=tmp_path / "chroma")

    with TestClient(create_app(settings, services)) as client:
        query_response = client.post("/query", json={"question": "Question?"})
        ingest_response = client.post(
            "/ingest",
            data={"url": "https://example.com/guide"},
        )

    assert query_response.status_code == 503
    assert ingest_response.status_code == 503


def test_openapi_exposes_all_required_routes(api: ApiContext) -> None:
    paths = api.client.get("/openapi.json").json()["paths"]

    assert {"/query", "/ingest", "/documents", "/feedback"} <= set(paths)
