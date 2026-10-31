from uuid import uuid4

import httpx
import pytest

from app.ui_client import KnowItApiClient


def test_ui_client_sends_session_and_feedback() -> None:
    session_id = uuid4()
    answer_id = uuid4()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/query":
            return httpx.Response(
                200,
                json={"session_id": str(session_id), "answer_id": str(answer_id)},
            )
        return httpx.Response(200, json={"accepted": True})

    http = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )
    client = KnowItApiClient("http://testserver", client=http)

    client.query("Follow-up", session_id)
    client.feedback(answer_id, "up")

    assert b'"session_id"' in requests[0].content
    assert requests[1].url.path == "/feedback"


def test_ui_client_surfaces_api_errors() -> None:
    http = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, json={"detail": "Provider unavailable"})
        ),
    )

    with pytest.raises(RuntimeError, match="Provider unavailable"):
        KnowItApiClient("http://testserver", client=http).query("Question")
