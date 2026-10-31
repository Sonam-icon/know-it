from uuid import uuid4

import pytest

from app.errors import AnswerNotFoundError
from app.rag import AnswerStatus, QueryResult
from app.repository import SQLiteRepository
from app.schemas import FeedbackRating


def test_answer_and_feedback_are_persisted(tmp_path) -> None:
    repository = SQLiteRepository(tmp_path / "application.sqlite3")
    repository.initialize()
    result = QueryResult(
        answer="Grounded answer.",
        status=AnswerStatus.ANSWERED,
        sources=(),
        rewrite_count=1,
    )

    answer_id = repository.save_answer("Question?", result)
    feedback_id = repository.add_feedback(
        answer_id=answer_id,
        rating=FeedbackRating.UP,
        comment="Helpful",
    )

    stored = repository.get_answer(answer_id)
    assert stored is not None
    assert stored.answer_id == answer_id
    assert stored.question == "Question?"
    assert stored.rewrite_count == 1
    assert repository.feedback_count(answer_id) == 1
    assert feedback_id != answer_id


def test_feedback_rejects_an_unknown_answer(tmp_path) -> None:
    repository = SQLiteRepository(tmp_path / "application.sqlite3")
    repository.initialize()

    with pytest.raises(AnswerNotFoundError):
        repository.add_feedback(
            answer_id=uuid4(),
            rating=FeedbackRating.DOWN,
            comment=None,
        )
