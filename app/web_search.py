"""Bounded Tavily search adapter for optional documentation fallback."""

from __future__ import annotations

import hashlib
import json
from urllib.parse import urlsplit

import httpx

from app.config import Settings
from app.documents import RetrievedChunk, SourceKind, SourceOrigin
from app.errors import ConfigurationError, WebSearchError


class TavilySearcher:
    """Convert allowlisted Tavily results into the workflow's evidence model."""

    endpoint = "https://api.tavily.com/search"

    def __init__(self, settings: Settings, *, client: httpx.Client | None = None) -> None:
        if not settings.web_search_configured or settings.tavily_api_key is None:
            raise ConfigurationError("TAVILY_API_KEY is required when WEB_FALLBACK_ENABLED is true")
        if not settings.allowed_web_domains:
            raise ConfigurationError("WEB_SEARCH_DOMAINS must contain at least one domain")
        self._api_key = settings.tavily_api_key.get_secret_value()
        self._domains = settings.allowed_web_domains
        self._timeout = settings.provider_timeout_seconds
        self._max_response_bytes = settings.web_search_max_response_bytes
        self._max_excerpt_characters = settings.web_search_max_excerpt_characters
        self._client = client

    def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        client = self._client or httpx.Client(timeout=self._timeout, trust_env=False)
        should_close = self._client is None
        try:
            with client.stream(
                "POST",
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "search_depth": "basic",
                    "max_results": top_k,
                    "include_answer": False,
                    "include_raw_content": False,
                    "include_domains": list(self._domains),
                    "include_domains_mode": "restrict",
                },
            ) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if (
                    content_length
                    and content_length.isdigit()
                    and int(content_length) > self._max_response_bytes
                ):
                    raise WebSearchError("Tavily response exceeded the configured size limit")
                body = bytearray()
                for part in response.iter_bytes():
                    body.extend(part)
                    if len(body) > self._max_response_bytes:
                        raise WebSearchError("Tavily response exceeded the configured size limit")
        finally:
            if should_close:
                client.close()

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise WebSearchError("Tavily returned invalid JSON") from exc
        raw_results = payload.get("results", []) if isinstance(payload, dict) else []
        if not isinstance(raw_results, list):
            return []

        chunks: list[RetrievedChunk] = []
        seen_urls: set[str] = set()
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            content = str(item.get("content", "")).strip()
            if not content or not self._is_allowed_url(url) or url in seen_urls:
                continue
            seen_urls.add(url)
            excerpt = content[: self._max_excerpt_characters]
            title = str(item.get("title", "")).strip() or urlsplit(url).hostname or "Web result"
            document_id = _sha256(url)
            chunks.append(
                RetrievedChunk(
                    chunk_id=_sha256(f"{url}:{excerpt}"),
                    document_id=document_id,
                    title=title[:200],
                    source=url,
                    source_kind=SourceKind.URL,
                    heading="Web search result",
                    text=excerpt,
                    distance=1.0 - _bounded_score(item.get("score")),
                    origin=SourceOrigin.WEB,
                )
            )
        return chunks[:top_k]

    def _is_allowed_url(self, url: str) -> bool:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        hostname = parsed.hostname.casefold().rstrip(".")
        return any(
            hostname == domain or hostname.endswith(f".{domain}") for domain in self._domains
        )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bounded_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(score, 1.0))
