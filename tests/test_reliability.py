import sqlite3
from datetime import UTC, datetime, timedelta

import httpx
from google.genai.errors import APIError
from langgraph.types import RetryPolicy

from app.checkpoints import SQLiteCheckpointManager
from app.reliability import is_transient_error
from tests.test_query_workflow import (
    FakeEmbeddings,
    FakeQueryModel,
    FakeRetriever,
    _chunk,
    _workflow,
)


class FlakyEmbeddings(FakeEmbeddings):
    def __init__(self, failures: int, error: Exception) -> None:
        super().__init__()
        self.failures = failures
        self.error = error
        self.attempts = 0

    def embed_query(self, text: str) -> list[float]:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise self.error
        return super().embed_query(text)


def test_transient_error_classifier_is_narrow() -> None:
    request = httpx.Request("GET", "https://example.com")
    assert is_transient_error(httpx.ConnectError("offline", request=request))
    assert is_transient_error(APIError(429, {"message": "rate limited"}))
    assert not is_transient_error(APIError(400, {"message": "bad request"}))
    assert not is_transient_error(ValueError("invalid application input"))


def test_transient_node_failure_retries_without_semantic_rewrite() -> None:
    request = httpx.Request("POST", "https://example.com")
    embeddings = FlakyEmbeddings(
        failures=2,
        error=httpx.ConnectError("offline", request=request),
    )
    model = FakeQueryModel(relevant_ids={"relevant"})
    retriever = FakeRetriever([[_chunk("relevant")]])
    retry_policy = RetryPolicy(
        initial_interval=0.001,
        backoff_factor=1,
        max_interval=0.001,
        max_attempts=3,
        jitter=False,
        retry_on=is_transient_error,
    )

    result = _workflow(
        model,
        retriever,
        embeddings,
        retry_policy=retry_policy,
    ).run("Question")

    assert result.rewrite_count == 0
    assert embeddings.attempts == 3
    assert retriever.calls == 1


def test_permanent_node_failure_is_not_retried() -> None:
    embeddings = FlakyEmbeddings(failures=3, error=ValueError("permanent"))
    model = FakeQueryModel(relevant_ids={"relevant"})
    retry_policy = RetryPolicy(
        initial_interval=0.001,
        max_attempts=3,
        jitter=False,
        retry_on=is_transient_error,
    )

    try:
        _workflow(
            model,
            FakeRetriever([[_chunk("relevant")]]),
            embeddings,
            retry_policy=retry_policy,
        ).run("Question")
    except ValueError:
        pass
    else:
        raise AssertionError("Permanent failure should propagate")

    assert embeddings.attempts == 1


def test_sqlite_checkpoints_persist_state_and_expire_by_thread(tmp_path) -> None:
    path = tmp_path / "checkpoints.sqlite3"
    manager = SQLiteCheckpointManager(path)
    manager.initialize()
    try:
        model = FakeQueryModel(relevant_ids={"relevant"})
        workflow = _workflow(
            model,
            FakeRetriever([[_chunk("relevant")]]),
            FakeEmbeddings(),
            checkpointer=manager.saver,
            on_complete=manager.complete,
        )

        workflow.run("Question")

        assert manager.completed_thread_count() == 1
        assert len(list(manager.saver.list(None))) > 1

        old_timestamp = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        with sqlite3.connect(path) as connection:
            connection.execute(
                "UPDATE checkpoint_threads SET completed_at = ?",
                (old_timestamp,),
            )
        assert manager.cleanup(retention_days=7) == 1
        assert manager.completed_thread_count() == 0
        assert list(manager.saver.list(None)) == []
    finally:
        manager.close()
