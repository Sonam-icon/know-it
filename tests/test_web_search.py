import json

import httpx
import pytest

from app.config import Settings
from app.documents import SourceOrigin
from app.errors import ConfigurationError, WebSearchError
from app.web_search import TavilySearcher


def _settings(**overrides):
    return Settings(
        _env_file=None,
        web_fallback_enabled=True,
        tavily_api_key="test-key",
        **overrides,
    )


def test_tavily_results_are_bounded_deduplicated_and_allowlisted() -> None:
    payload = {
        "results": [
            {
                "title": "Dependencies",
                "url": "https://fastapi.tiangolo.com/tutorial/dependencies/",
                "content": "Use Depends.",
                "score": 0.9,
            },
            {
                "title": "Duplicate",
                "url": "https://fastapi.tiangolo.com/tutorial/dependencies/",
                "content": "Duplicate content.",
            },
            {
                "title": "Untrusted",
                "url": "https://example.com/guide",
                "content": "Do not include this.",
            },
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["include_domains"] == ["fastapi.tiangolo.com"]
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    results = TavilySearcher(_settings(), client=client).search("Depends", top_k=3)

    assert len(results) == 1
    assert results[0].origin is SourceOrigin.WEB
    assert results[0].source == "https://fastapi.tiangolo.com/tutorial/dependencies/"


def test_tavily_http_errors_are_preserved_for_retry_policy() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(429, request=request))
    )

    with pytest.raises(httpx.HTTPStatusError):
        TavilySearcher(_settings(), client=client).search("Depends", top_k=1)


def test_tavily_requires_enabled_configuration() -> None:
    with pytest.raises(ConfigurationError):
        TavilySearcher(Settings(_env_file=None))


def test_tavily_rejects_invalid_json() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text="not-json", request=request)
        )
    )

    with pytest.raises(WebSearchError, match="invalid JSON"):
        TavilySearcher(_settings(), client=client).search("Depends", top_k=1)
