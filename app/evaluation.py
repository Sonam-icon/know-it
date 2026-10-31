"""Small, reproducible evaluation harness for grounded RAG behavior."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.rag import AnswerStatus, QueryResult


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationCase(EvaluationModel):
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_source_urls: list[str] = Field(default_factory=list)
    expected_terms: list[str] = Field(default_factory=list)
    expect_abstention: bool = False


class EvaluationResult(EvaluationModel):
    id: str
    category: str
    status: str
    passed: bool
    abstention_correct: bool
    sources_valid: bool
    expected_source_found: bool
    expected_terms_found: bool
    latency_ms: float
    source_urls: list[str]


class EvaluationSummary(EvaluationModel):
    total: int
    passed: int
    pass_rate: float
    average_latency_ms: float
    results: list[EvaluationResult]


class EvaluationQueryRunner(Protocol):
    def run(self, question: str) -> QueryResult: ...


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Evaluation dataset must contain a JSON list")
    return [EvaluationCase.model_validate(item) for item in raw]


def run_evaluation(
    query_runner: EvaluationQueryRunner,
    cases: list[EvaluationCase],
) -> EvaluationSummary:
    results: list[EvaluationResult] = []
    for case in cases:
        started = perf_counter()
        result = query_runner.run(case.question)
        latency_ms = (perf_counter() - started) * 1000
        source_urls = list(dict.fromkeys(source.source for source in result.sources))
        abstained = result.status is AnswerStatus.INSUFFICIENT_CONTEXT
        abstention_correct = abstained == case.expect_abstention
        sources_valid = not result.sources if abstained else bool(result.sources)
        expected_source_found = case.expect_abstention or any(
            expected in source_urls for expected in case.expected_source_urls
        )
        normalized_answer = result.answer.casefold()
        expected_terms_found = case.expect_abstention or all(
            term.casefold() in normalized_answer for term in case.expected_terms
        )
        passed = (
            abstention_correct and sources_valid and expected_source_found and expected_terms_found
        )
        results.append(
            EvaluationResult(
                id=case.id,
                category=case.category,
                status=result.status.value,
                passed=passed,
                abstention_correct=abstention_correct,
                sources_valid=sources_valid,
                expected_source_found=expected_source_found,
                expected_terms_found=expected_terms_found,
                latency_ms=round(latency_ms, 2),
                source_urls=source_urls,
            )
        )

    passed_count = sum(item.passed for item in results)
    average_latency = sum(item.latency_ms for item in results) / len(results) if results else 0.0
    return EvaluationSummary(
        total=len(results),
        passed=passed_count,
        pass_rate=round(passed_count / len(results), 4) if results else 0.0,
        average_latency_ms=round(average_latency, 2),
        results=results,
    )


def write_evaluation_summary(summary: EvaluationSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )
