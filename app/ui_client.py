"""Small HTTP client used by the optional Streamlit interface."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import httpx


class KnowItApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 60,
        client: httpx.Client | None = None,
    ) -> None:
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )
        self._owns_client = client is None

    def query(self, question: str, session_id: UUID | None = None) -> dict[str, Any]:
        payload: dict[str, str] = {"question": question}
        if session_id is not None:
            payload["session_id"] = str(session_id)
        return self._request("POST", "/query", json=payload)

    def feedback(self, answer_id: UUID, rating: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/feedback",
            json={"answer_id": str(answer_id), "rating": rating},
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self._client.request(method, path, **kwargs)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"Know-It API returned {response.status_code}: {detail}") from exc
        value = response.json()
        if not isinstance(value, dict):
            raise RuntimeError("Know-It API returned an unexpected response")
        return value
