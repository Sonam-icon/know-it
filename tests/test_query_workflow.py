from collections.abc import Sequence

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import RetryPolicy

from app.documents import RetrievedChunk, SourceKind
from app.errors import CitationValidationError, StructuredOutputError
from app.query_workflow import build_query_workflow
from app.rag import (
    AnswerDraft,
    AnswerStatus,
    QueryAnalysis,
    QueryKind,
    QueryRewrite,
    RelevanceGrade,
)


def _chunk(chunk_id: str, text: str = "Useful documentation.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        title="FastAPI Guide",
        source="https://example.com/guide",
        source_kind=SourceKind.URL,
        heading="Guide / Topic",
        text=text,
        distance=0.1,
    )


class FakeEmbeddings:
    model_name = "fake"
    dimension = 1

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [float(len(text))]


class FakeRetriever:
    def __init__(self, batches: list[list[RetrievedChunk]]) -> None:
        self.batches = batches
        self.calls = 0

    def search_by_vector(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        del query_embedding, top_k
        index = min(self.calls, len(self.batches) - 1)
        self.calls += 1
        return self.batches[index]


class FakeQueryModel:
    def __init__(
        self,
        *,
        relevant_ids: set[str],
        rewrites: list[str] | None = None,
        draft: AnswerDraft | None = None,
        analysis_error: bool = False,
    ) -> None:
        self.relevant_ids = relevant_ids
        self.rewrites = list(rewrites or [])
        self.draft = draft or AnswerDraft(
            answer="Use the documented API. [S1]",
            citation_ids=["S1"],
        )
        self.analysis_error = analysis_error
        self.generated_with: list[RetrievedChunk] = []

    def analyze(self, question: str) -> QueryAnalysis:
        if self.analysis_error:
            raise StructuredOutputError("invalid analysis")
        return QueryAnalysis(
            search_query="initial retrieval query",
            query_type=QueryKind.HOW_TO,
        )

    def grade(self, question: str, chunk: RetrievedChunk) -> RelevanceGrade:
        del question
        relevant = chunk.chunk_id in self.relevant_ids
        return RelevanceGrade(
            relevant=relevant,
            reason="contains the answer" if relevant else "unrelated",
        )

    def rewrite(
        self,
        question: str,
        current_query: str,
        attempted_queries: list[str],
    ) -> QueryRewrite:
        del question, current_query, attempted_queries
        return QueryRewrite(search_query=self.rewrites.pop(0))

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> AnswerDraft:
        del question
        self.generated_with = chunks
        return self.draft


def _workflow(
    model: FakeQueryModel,
    retriever: FakeRetriever,
    embeddings: FakeEmbeddings,
    *,
    max_rewrites: int = 2,
    max_context_characters: int = 10_000,
    retry_policy: RetryPolicy | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    on_complete=None,
):
    return build_query_workflow(
        model=model,
        embeddings=embeddings,
        retriever=retriever,
        top_k=5,
        max_query_rewrites=max_rewrites,
        grading_concurrency=1,
        max_context_characters=max_context_characters,
        max_question_length=2000,
        retry_policy=retry_policy,
        checkpointer=checkpointer,
        on_complete=on_complete,
    )


def test_relevant_chunks_are_filtered_and_cited() -> None:
    relevant = _chunk("relevant", "Use Depends to declare a dependency.")
    irrelevant = _chunk("irrelevant", "This discusses response status codes.")
    model = FakeQueryModel(relevant_ids={"relevant"})
    retriever = FakeRetriever([[relevant, irrelevant]])
    embeddings = FakeEmbeddings()

    result = _workflow(model, retriever, embeddings).run("How do I declare a FastAPI dependency?")

    assert result.status is AnswerStatus.ANSWERED
    assert result.rewrite_count == 0
    assert model.generated_with == [relevant]
    assert [source.chunk_id for source in result.sources] == ["relevant"]
    assert result.sources[0].citation_id == "S1"


def test_all_irrelevant_results_use_bounded_rewrites_then_abstain() -> None:
    chunk = _chunk("irrelevant")
    model = FakeQueryModel(
        relevant_ids=set(),
        rewrites=["second retrieval query", "third retrieval query"],
    )
    retriever = FakeRetriever([[chunk], [chunk], [chunk]])
    embeddings = FakeEmbeddings()

    result = _workflow(model, retriever, embeddings).run("Unsupported question")

    assert result.status is AnswerStatus.INSUFFICIENT_CONTEXT
    assert result.sources == ()
    assert result.rewrite_count == 2
    assert retriever.calls == 3
    assert embeddings.queries == [
        "initial retrieval query",
        "second retrieval query",
        "third retrieval query",
    ]


def test_duplicate_rewrite_terminates_without_another_retrieval() -> None:
    model = FakeQueryModel(
        relevant_ids=set(),
        rewrites=[" Initial Retrieval Query "],
    )
    retriever = FakeRetriever([[_chunk("irrelevant")]])
    embeddings = FakeEmbeddings()

    result = _workflow(model, retriever, embeddings).run("Unsupported question")

    assert result.status is AnswerStatus.INSUFFICIENT_CONTEXT
    assert result.rewrite_count == 1
    assert retriever.calls == 1


def test_analysis_schema_failure_falls_back_to_original_question() -> None:
    model = FakeQueryModel(relevant_ids=set(), analysis_error=True)
    retriever = FakeRetriever([[]])
    embeddings = FakeEmbeddings()

    result = _workflow(
        model,
        retriever,
        embeddings,
        max_rewrites=0,
    ).run("  What does Depends do?  ")

    assert result.status is AnswerStatus.INSUFFICIENT_CONTEXT
    assert embeddings.queries == ["What does Depends do?"]


def test_generation_rejects_unknown_citations() -> None:
    model = FakeQueryModel(
        relevant_ids={"relevant"},
        draft=AnswerDraft(answer="Unsupported reference. [S2]", citation_ids=["S2"]),
    )
    retriever = FakeRetriever([[_chunk("relevant")]])

    with pytest.raises(CitationValidationError, match="unknown source"):
        _workflow(model, retriever, FakeEmbeddings()).run("Question")


def test_generation_context_is_bounded_before_the_model_call() -> None:
    model = FakeQueryModel(relevant_ids={"relevant"})
    retriever = FakeRetriever([[_chunk("relevant", "x" * 100)]])

    _workflow(
        model,
        retriever,
        FakeEmbeddings(),
        max_context_characters=20,
    ).run("Question")

    assert len(model.generated_with[0].text) == 20


def test_graph_invocations_do_not_share_request_state() -> None:
    model = FakeQueryModel(relevant_ids={"relevant"})
    retriever = FakeRetriever([[_chunk("relevant")]])
    embeddings = FakeEmbeddings()
    workflow = _workflow(model, retriever, embeddings)

    first = workflow.run("First question")
    second = workflow.run("Second question")

    assert first.rewrite_count == second.rewrite_count == 0
    assert embeddings.queries == ["initial retrieval query", "initial retrieval query"]
