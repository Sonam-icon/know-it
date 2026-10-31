from pathlib import Path

from app.documents import SourceKind
from app.evaluation import EvaluationCase, load_evaluation_cases, run_evaluation
from app.rag import AnswerSource, AnswerStatus, QueryResult


class FakeEvaluationRunner:
    def run(self, question: str) -> QueryResult:
        if "outside" in question:
            return QueryResult(
                answer="The indexed documentation does not cover that topic.",
                status=AnswerStatus.INSUFFICIENT_CONTEXT,
                sources=(),
                rewrite_count=2,
            )
        return QueryResult(
            answer="Use Depends for dependency injection. [S1]",
            status=AnswerStatus.ANSWERED,
            sources=(
                AnswerSource(
                    citation_id="S1",
                    chunk_id="chunk-1",
                    document_id="document-1",
                    title="Dependencies",
                    heading="Dependencies",
                    source="https://example.com/dependencies",
                    source_kind=SourceKind.URL,
                    excerpt="Use Depends.",
                ),
            ),
            rewrite_count=0,
        )


def test_versioned_dataset_has_supported_and_abstention_cases() -> None:
    cases = load_evaluation_cases(Path("evaluation/questions.json"))

    assert len(cases) == 15
    assert any(case.expect_abstention for case in cases)
    assert any(not case.expect_abstention for case in cases)
    assert len({case.id for case in cases}) == len(cases)


def test_evaluation_scores_sources_terms_and_abstention() -> None:
    cases = [
        EvaluationCase(
            id="supported",
            category="how_to",
            question="How do dependencies work?",
            expected_source_urls=["https://example.com/dependencies"],
            expected_terms=["Depends"],
        ),
        EvaluationCase(
            id="unsupported",
            category="out_of_scope",
            question="This is outside the corpus",
            expect_abstention=True,
        ),
    ]

    summary = run_evaluation(FakeEvaluationRunner(), cases)

    assert summary.total == 2
    assert summary.passed == 2
    assert summary.pass_rate == 1.0
