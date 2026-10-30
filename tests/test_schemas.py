from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas import FeedbackRating, FeedbackRequest, QueryRequest


def test_query_strips_surrounding_whitespace() -> None:
    request = QueryRequest(question="  How do dependencies work?  ")

    assert request.question == "How do dependencies work?"


def test_query_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        QueryRequest(question="Valid question", session_id="unexpected")


def test_feedback_uses_a_bounded_rating_contract() -> None:
    request = FeedbackRequest(answer_id=uuid4(), rating=FeedbackRating.UP)

    assert request.rating is FeedbackRating.UP
