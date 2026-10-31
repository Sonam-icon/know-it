from typing import Any

import app.providers as providers
from app.config import Settings
from app.providers import GeminiEmbeddingProvider, GeminiQueryModel
from app.rag import QueryAnalysis


def _key_value(value: Any) -> str:
    return value.get_secret_value() if hasattr(value, "get_secret_value") else str(value)


def test_chat_calls_rotate_across_configured_keys(monkeypatch) -> None:
    calls: list[str] = []

    class FakeStructuredRunnable:
        def __init__(self, key: str, output_type: type[Any]) -> None:
            self.key = key
            self.output_type = output_type

        def invoke(self, messages: object) -> object:
            del messages
            calls.append(self.key)
            if self.output_type is QueryAnalysis:
                return QueryAnalysis(
                    standalone_question="Question",
                    search_query="question",
                )
            raise AssertionError("This test invokes only the analyzer")

    class FakeChatModel:
        def __init__(self, *, api_key: object, **kwargs: object) -> None:
            del kwargs
            self.key = _key_value(api_key)

        def with_structured_output(self, output_type: type[Any]) -> FakeStructuredRunnable:
            return FakeStructuredRunnable(self.key, output_type)

    monkeypatch.setattr(providers, "ChatGoogleGenerativeAI", FakeChatModel)
    model = GeminiQueryModel(
        Settings(_env_file=None, google_api_key="key-a,key-b,key-c")
    )

    for _ in range(4):
        model.analyze("Question", ())

    assert calls == ["key-a", "key-b", "key-c", "key-a"]


def test_embedding_calls_rotate_across_configured_keys(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeEmbeddings:
        def __init__(self, *, api_key: object, task_type: str, **kwargs: object) -> None:
            del kwargs
            self.key = _key_value(api_key)
            self.task_type = task_type

        def embed_query(self, text: str) -> list[float]:
            del text
            calls.append((self.key, self.task_type))
            return [1.0]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            calls.append((self.key, self.task_type))
            return [[1.0] for _ in texts]

    monkeypatch.setattr(providers, "GoogleGenerativeAIEmbeddings", FakeEmbeddings)
    embeddings = GeminiEmbeddingProvider(
        Settings(_env_file=None, google_api_key="key-a,key-b,key-c")
    )

    embeddings.embed_query("one")
    embeddings.embed_query("two")
    embeddings.embed_query("three")
    embeddings.embed_documents(["four"])

    assert calls == [
        ("key-a", "RETRIEVAL_QUERY"),
        ("key-b", "RETRIEVAL_QUERY"),
        ("key-c", "RETRIEVAL_QUERY"),
        ("key-a", "RETRIEVAL_DOCUMENT"),
    ]
