"""Self-corrective LangGraph workflow for grounded documentation answers."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from app.documents import RetrievedChunk, SourceOrigin
from app.errors import CitationValidationError, QueryValidationError, StructuredOutputError
from app.providers import EmbeddingProvider
from app.rag import (
    AnswerDraft,
    AnswerSource,
    AnswerStatus,
    ConversationTurn,
    GradedChunk,
    GroundednessVerdict,
    QueryModel,
    QueryResult,
    RouteAfterGrade,
    RouteAfterRewrite,
    RouteAfterVerify,
)

_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")
_ABSTENTION = (
    "I could not find enough relevant information in the available documentation "
    "to answer this question."
)
logger = logging.getLogger(__name__)


class VectorRetriever(Protocol):
    def search_by_vector(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]: ...


class WebSearcher(Protocol):
    def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]: ...


class QueryInput(TypedDict):
    question: str
    history: tuple[ConversationTurn, ...]


class QueryOutput(TypedDict):
    answer: str
    status: AnswerStatus
    sources: list[AnswerSource]
    rewrite_count: int
    web_attempted: bool


class QueryState(TypedDict, total=False):
    question: str
    history: tuple[ConversationTurn, ...]
    standalone_question: str
    search_query: str
    retrieved_chunks: list[RetrievedChunk]
    grades: list[GradedChunk]
    relevant_chunks: list[RetrievedChunk]
    attempted_queries: list[str]
    rewrite_count: int
    rewrite_exhausted: bool
    retrieval_origin: SourceOrigin
    web_attempted: bool
    context_chunks: list[RetrievedChunk]
    draft: AnswerDraft
    draft_sources: list[AnswerSource]
    verification: GroundednessVerdict
    revision_count: int
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

    def run(
        self,
        question: str,
        history: tuple[ConversationTurn, ...] = (),
    ) -> QueryResult:
        normalized = question.strip()
        if not normalized:
            raise QueryValidationError("Question must not be empty")
        if len(normalized) > self.max_question_length:
            raise QueryValidationError(
                f"Question exceeds the {self.max_question_length}-character limit"
            )
        output = self.graph.invoke(
            {"question": normalized, "history": history},
            {"recursion_limit": self.recursion_limit},
        )
        logger.info(
            "query_complete status=%s rewrite_count=%s source_count=%s",
            output["status"].value,
            output["rewrite_count"],
            len(output["sources"]),
        )
        return QueryResult(
            answer=output["answer"],
            status=output["status"],
            sources=tuple(output["sources"]),
            rewrite_count=output["rewrite_count"],
            web_attempted=output["web_attempted"],
        )


def build_query_workflow(
    *,
    model: QueryModel,
    embeddings: EmbeddingProvider,
    retriever: VectorRetriever,
    top_k: int,
    max_query_rewrites: int,
    max_context_characters: int,
    max_question_length: int,
    max_answer_revisions: int = 1,
    web_searcher: WebSearcher | None = None,
    web_top_k: int = 3,
    retry_policy: RetryPolicy | None = None,
) -> QueryWorkflow:
    """Build the graph with dependencies captured by node closures."""

    def analyze(state: QueryState) -> QueryState:
        question = state["question"]
        try:
            result = model.analyze(question, state.get("history", ()))
            standalone_question = _normalize_query(result.standalone_question)
            search_query = _normalize_query(result.search_query)
        except StructuredOutputError:
            logger.warning("query_analysis_fallback reason=invalid_structured_output")
            standalone_question = question
            search_query = question
        logger.info("graph_node=analyze")
        return {
            "standalone_question": standalone_question,
            "search_query": search_query,
            "attempted_queries": [search_query],
            "rewrite_count": 0,
            "rewrite_exhausted": False,
            "retrieval_origin": SourceOrigin.CORPUS,
            "web_attempted": False,
            "revision_count": 0,
            "retrieved_chunks": [],
            "grades": [],
            "relevant_chunks": [],
        }

    def retrieve(state: QueryState) -> QueryState:
        vector = embeddings.embed_query(state["search_query"])
        chunks = retriever.search_by_vector(vector, top_k=top_k)
        logger.info("graph_node=retrieve result_count=%s", len(chunks))
        return {
            "retrieved_chunks": chunks,
            "retrieval_origin": SourceOrigin.CORPUS,
            "grades": [],
            "relevant_chunks": [],
        }

    def grade(state: QueryState) -> QueryState:
        chunks = state.get("retrieved_chunks", [])

        def grade_one(chunk: RetrievedChunk) -> GradedChunk:
            result = model.grade(state["standalone_question"], chunk)
            return GradedChunk(
                chunk_id=chunk.chunk_id,
                relevant=result.relevant,
                reason=result.reason,
            )

        grades = [grade_one(chunk) for chunk in chunks]
        relevant_ids = {item.chunk_id for item in grades if item.relevant}
        relevant = [chunk for chunk in chunks if chunk.chunk_id in relevant_ids]
        logger.info(
            "graph_node=grade retrieved_count=%s relevant_count=%s",
            len(chunks),
            len(relevant),
        )
        return {"grades": grades, "relevant_chunks": relevant}

    def rewrite(state: QueryState) -> QueryState:
        attempted = state.get("attempted_queries", [])
        result = model.rewrite(
            state["standalone_question"],
            state["search_query"],
            attempted,
        )
        search_query = _normalize_query(result.search_query)
        rewrite_count = state.get("rewrite_count", 0) + 1
        duplicate = _query_key(search_query) in {_query_key(query) for query in attempted}
        logger.info(
            "graph_node=rewrite rewrite_count=%s duplicate=%s",
            rewrite_count,
            duplicate,
        )
        return {
            "search_query": search_query,
            "attempted_queries": [*attempted, search_query],
            "rewrite_count": rewrite_count,
            "rewrite_exhausted": duplicate,
            "retrieved_chunks": [],
            "grades": [],
            "relevant_chunks": [],
        }

    def search_web(state: QueryState) -> QueryState:
        if web_searcher is None:  # pragma: no cover - routing prevents this
            return {"retrieved_chunks": [], "web_attempted": True}
        chunks = web_searcher.search(state["search_query"], top_k=web_top_k)
        logger.info("graph_node=search_web result_count=%s", len(chunks))
        return {
            "retrieved_chunks": chunks,
            "retrieval_origin": SourceOrigin.WEB,
            "web_attempted": True,
            "grades": [],
            "relevant_chunks": [],
        }

    def generate(state: QueryState) -> QueryState:
        context = _bounded_context(
            state.get("relevant_chunks", []),
            max_characters=max_context_characters,
        )
        draft = model.generate(state["standalone_question"], context)
        sources = _validate_and_build_sources(draft.answer, draft.citation_ids, context)
        logger.info(
            "graph_node=generate context_count=%s citation_count=%s",
            len(context),
            len(sources),
        )
        return {
            "context_chunks": context,
            "draft": draft,
            "draft_sources": sources,
        }

    def verify(state: QueryState) -> QueryState:
        verdict = model.verify(
            state["standalone_question"],
            state["draft"],
            state["context_chunks"],
        )
        logger.info(
            "graph_node=verify supported=%s unsupported_count=%s",
            verdict.supported,
            len(verdict.unsupported_claims),
        )
        return {"verification": verdict}

    def revise(state: QueryState) -> QueryState:
        draft = model.revise(
            state["standalone_question"],
            state["draft"],
            state["context_chunks"],
            state["verification"].unsupported_claims,
        )
        sources = _validate_and_build_sources(
            draft.answer,
            draft.citation_ids,
            state["context_chunks"],
        )
        revision_count = state.get("revision_count", 0) + 1
        logger.info("graph_node=revise revision_count=%s", revision_count)
        return {
            "draft": draft,
            "draft_sources": sources,
            "revision_count": revision_count,
        }

    def finalize(state: QueryState) -> QueryState:
        return {
            "answer": state["draft"].answer,
            "sources": state["draft_sources"],
            "status": AnswerStatus.ANSWERED,
        }

    def abstain(state: QueryState) -> QueryState:
        logger.info(
            "graph_node=abstain rewrite_count=%s",
            state.get("rewrite_count", 0),
        )
        return {
            "answer": _ABSTENTION,
            "sources": [],
            "status": AnswerStatus.INSUFFICIENT_CONTEXT,
            "rewrite_count": state.get("rewrite_count", 0),
            "web_attempted": state.get("web_attempted", False),
        }

    def route_after_grade(state: QueryState) -> RouteAfterGrade:
        if state.get("relevant_chunks"):
            return "generate"
        if state.get("retrieval_origin") is SourceOrigin.WEB:
            return "abstain"
        if state.get("rewrite_count", 0) < max_query_rewrites:
            return "rewrite"
        if web_searcher is not None and not state.get("web_attempted", False):
            return "search_web"
        return "abstain"

    def route_after_rewrite(state: QueryState) -> RouteAfterRewrite:
        if not state.get("rewrite_exhausted", False):
            return "retrieve"
        if web_searcher is not None and not state.get("web_attempted", False):
            return "search_web"
        return "abstain"

    def route_after_verify(state: QueryState) -> RouteAfterVerify:
        if state["verification"].supported:
            return "finalize"
        if state.get("revision_count", 0) < max_answer_revisions:
            return "revise"
        return "abstain"

    builder = StateGraph(
        QueryState,
        input_schema=QueryInput,
        output_schema=QueryOutput,
    )
    external_node_options = {
        "retry_policy": retry_policy,
    }
    builder.add_node("analyze", analyze, **external_node_options)
    builder.add_node("retrieve", retrieve, **external_node_options)
    builder.add_node("grade", grade, **external_node_options)
    builder.add_node("rewrite", rewrite, **external_node_options)
    builder.add_node("search_web", search_web, **external_node_options)
    builder.add_node("generate", generate, **external_node_options)
    builder.add_node("verify", verify, **external_node_options)
    builder.add_node("revise", revise, **external_node_options)
    builder.add_node("finalize", finalize)
    builder.add_node("abstain", abstain)
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", route_after_grade)
    builder.add_conditional_edges("rewrite", route_after_rewrite)
    builder.add_edge("search_web", "grade")
    builder.add_edge("generate", "verify")
    builder.add_conditional_edges("verify", route_after_verify)
    builder.add_edge("revise", "verify")
    builder.add_edge("finalize", END)
    builder.add_edge("abstain", END)
    graph = builder.compile(name="technical-documentation-rag")
    recursion_limit = 16 + (max_query_rewrites * 3) + (max_answer_revisions * 2)
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
                origin=chunk.origin,
            )
        )
    return sources


def _normalize_query(query: str) -> str:
    return " ".join(query.split())


def _query_key(query: str) -> str:
    return _normalize_query(query).casefold()
