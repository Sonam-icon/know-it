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


def test_session_history_is_bounded_and_ordered(tmp_path) -> None:
    repository = SQLiteRepository(tmp_path / "application.sqlite3")
    repository.initialize()
    session_id = uuid4()
    result = QueryResult(
        answer="Answer",
        status=AnswerStatus.ANSWERED,
        sources=(),
        rewrite_count=0,
    )
    for index in range(4):
        repository.save_answer(f"Question {index}", result, session_id=session_id)

    history = repository.recent_turns(session_id, limit=3, max_characters=1000)

    assert [turn.question for turn in history] == ["Question 1", "Question 2", "Question 3"]
    assert repository.session_exists(session_id)


def test_initialize_migrates_an_existing_answers_table(tmp_path) -> None:
    import sqlite3

    path = tmp_path / "application.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE answers (
                answer_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                status TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                rewrite_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

    repository = SQLiteRepository(path)
    repository.initialize()

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(answers)")}
    assert "session_id" in columns
