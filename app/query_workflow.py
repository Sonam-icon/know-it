"""Self-corrective LangGraph workflow for grounded documentation answers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.documents import RetrievedChunk
from app.errors import CitationValidationError, QueryValidationError, StructuredOutputError
from app.providers import EmbeddingProvider
from app.rag import (
    AnswerSource,
    AnswerStatus,
    GradedChunk,
    QueryKind,
    QueryModel,
    QueryResult,
    RouteAfterGrade,
    RouteAfterRewrite,
)

_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")
_ABSTENTION = (
    "I could not find enough relevant information in the indexed documentation "
    "to answer this question."
)


class VectorRetriever(Protocol):
    def search_by_vector(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]: ...


class QueryInput(TypedDict):
    question: str


class QueryOutput(TypedDict):
    answer: str
    status: AnswerStatus
    sources: list[AnswerSource]
    rewrite_count: int


class QueryState(TypedDict, total=False):
    question: str
    search_query: str
    query_type: QueryKind
    retrieved_chunks: list[RetrievedChunk]
    grades: list[GradedChunk]
    relevant_chunks: list[RetrievedChunk]
    attempted_queries: list[str]
    rewrite_count: int
    rewrite_exhausted: bool
    answer: str
    sources: list[AnswerSource]
    status: AnswerStatus


class QueryWorkflow:
    """Invoke a compiled graph with request-local state and bounded recursion."""

    def __init__(
        self,
        graph: CompiledStateGraph,
        *,
        max_question_length: int,
        recursion_limit: int,
    ) -> None:
        self.graph = graph
        self.max_question_length = max_question_length
        self.recursion_limit = recursion_limit

    def run(self, question: str) -> QueryResult:
        normalized = question.strip()
        if not normalized:
            raise QueryValidationError("Question must not be empty")
        if len(normalized) > self.max_question_length:
            raise QueryValidationError(
                f"Question exceeds the {self.max_question_length}-character limit"
            )
        output = self.graph.invoke(
            {"question": normalized},
            {"recursion_limit": self.recursion_limit},
        )
        return QueryResult(
            answer=output["answer"],
            status=output["status"],
            sources=tuple(output["sources"]),
            rewrite_count=output["rewrite_count"],
        )


def build_query_workflow(
    *,
    model: QueryModel,
    embeddings: EmbeddingProvider,
    retriever: VectorRetriever,
    top_k: int,
    max_query_rewrites: int,
    grading_concurrency: int,
    max_context_characters: int,
    max_question_length: int,
) -> QueryWorkflow:
    """Build the graph with dependencies captured by node closures."""

    def analyze(state: QueryState) -> QueryState:
        question = state["question"]
        try:
            result = model.analyze(question)
            search_query = _normalize_query(result.search_query)
            query_type = result.query_type
        except StructuredOutputError:
            search_query = question
            query_type = QueryKind.CONCEPTUAL
        return {
            "search_query": search_query,
            "query_type": query_type,
            "attempted_queries": [search_query],
            "rewrite_count": 0,
            "rewrite_exhausted": False,
            "retrieved_chunks": [],
            "grades": [],
            "relevant_chunks": [],
        }

    def retrieve(state: QueryState) -> QueryState:
        vector = embeddings.embed_query(state["search_query"])
        chunks = retriever.search_by_vector(vector, top_k=top_k)
        return {
            "retrieved_chunks": chunks,
            "grades": [],
            "relevant_chunks": [],
        }

    def grade(state: QueryState) -> QueryState:
        chunks = state.get("retrieved_chunks", [])

        def grade_one(chunk: RetrievedChunk) -> GradedChunk:
            result = model.grade(state["question"], chunk)
            return GradedChunk(
                chunk_id=chunk.chunk_id,
                relevant=result.relevant,
                reason=result.reason,
            )

        if len(chunks) < 2 or grading_concurrency == 1:
            grades = [grade_one(chunk) for chunk in chunks]
        else:
            workers = min(grading_concurrency, len(chunks))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                grades = list(executor.map(grade_one, chunks))
        relevant_ids = {item.chunk_id for item in grades if item.relevant}
        relevant = [chunk for chunk in chunks if chunk.chunk_id in relevant_ids]
        return {"grades": grades, "relevant_chunks": relevant}

    def rewrite(state: QueryState) -> QueryState:
        attempted = state.get("attempted_queries", [])
        result = model.rewrite(
            state["question"],
            state["search_query"],
            attempted,
        )
        search_query = _normalize_query(result.search_query)
        rewrite_count = state.get("rewrite_count", 0) + 1
        duplicate = _query_key(search_query) in {_query_key(query) for query in attempted}
        return {
            "search_query": search_query,
            "attempted_queries": [*attempted, search_query],
            "rewrite_count": rewrite_count,
            "rewrite_exhausted": duplicate,
            "retrieved_chunks": [],
            "grades": [],
            "relevant_chunks": [],
        }

    def generate(state: QueryState) -> QueryState:
        context = _bounded_context(
            state.get("relevant_chunks", []),
            max_characters=max_context_characters,
        )
        draft = model.generate(state["question"], context)
        sources = _validate_and_build_sources(draft.answer, draft.citation_ids, context)
        return {
            "answer": draft.answer,
            "sources": sources,
            "status": AnswerStatus.ANSWERED,
        }

    def abstain(state: QueryState) -> QueryState:
        return {
            "answer": _ABSTENTION,
            "sources": [],
            "status": AnswerStatus.INSUFFICIENT_CONTEXT,
            "rewrite_count": state.get("rewrite_count", 0),
        }

    def route_after_grade(state: QueryState) -> RouteAfterGrade:
        if state.get("relevant_chunks"):
            return "generate"
        if state.get("rewrite_count", 0) < max_query_rewrites:
            return "rewrite"
        return "abstain"

    def route_after_rewrite(state: QueryState) -> RouteAfterRewrite:
        return "abstain" if state.get("rewrite_exhausted", False) else "retrieve"

    builder = StateGraph(
        QueryState,
        input_schema=QueryInput,
        output_schema=QueryOutput,
    )
    builder.add_node("analyze", analyze)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("rewrite", rewrite)
    builder.add_node("generate", generate)
    builder.add_node("abstain", abstain)
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", route_after_grade)
    builder.add_conditional_edges("rewrite", route_after_rewrite)
    builder.add_edge("generate", END)
    builder.add_edge("abstain", END)
    graph = builder.compile(name="technical-documentation-rag")
    recursion_limit = 10 + (max_query_rewrites * 3)
    return QueryWorkflow(
        graph,
        max_question_length=max_question_length,
        recursion_limit=recursion_limit,
    )


def _bounded_context(
    chunks: list[RetrievedChunk],
    *,
    max_characters: int,
) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    characters = 0
    for chunk in chunks:
        remaining = max_characters - characters
        if remaining <= 0:
            break
        if len(chunk.text) > remaining:
            selected.append(replace(chunk, text=chunk.text[:remaining]))
            break
        selected.append(chunk)
        characters += len(chunk.text)
    return selected


def _validate_and_build_sources(
    answer: str,
    citation_ids: list[str],
    chunks: list[RetrievedChunk],
) -> list[AnswerSource]:
    known = {f"S{index}": chunk for index, chunk in enumerate(chunks, start=1)}
    unique_ids = list(dict.fromkeys(citation_ids))
    marker_ids = set(_CITATION_PATTERN.findall(answer))
    claimed_ids = set(unique_ids)
    unknown = (marker_ids | claimed_ids) - set(known)
    if unknown:
        raise CitationValidationError(
            f"Generated answer referenced unknown source IDs: {sorted(unknown)}"
        )
    if not marker_ids or marker_ids != claimed_ids:
        raise CitationValidationError("Inline citations and structured citation IDs must match")

    sources: list[AnswerSource] = []
    for citation_id in unique_ids:
        chunk = known[citation_id]
        excerpt = " ".join(chunk.text.split())
        if len(excerpt) > 300:
            excerpt = excerpt[:297].rstrip() + "..."
        sources.append(
            AnswerSource(
                citation_id=citation_id,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                title=chunk.title,
                heading=chunk.heading,
                source=chunk.source,
                source_kind=chunk.source_kind,
                excerpt=excerpt,
            )
        )
    return sources


def _normalize_query(query: str) -> str:
    return " ".join(query.split())


def _query_key(query: str) -> str:
    return _normalize_query(query).casefold()
