from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas import FeedbackRating, FeedbackRequest, QueryRequest


def test_query_strips_surrounding_whitespace() -> None:
    request = QueryRequest(question="  How do dependencies work?  ")

    assert request.question == "How do dependencies work?"


def test_query_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        QueryRequest(question="Valid question", unexpected="value")


def test_query_accepts_an_optional_session_id() -> None:
    session_id = uuid4()

    request = QueryRequest(question="Follow-up", session_id=session_id)

    assert request.session_id == session_id


def test_feedback_uses_a_bounded_rating_contract() -> None:
    request = FeedbackRequest(answer_id=uuid4(), rating=FeedbackRating.UP)

    assert request.rating is FeedbackRating.UP
