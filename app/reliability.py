"""Retry classification and policies for transient external failures."""

import sqlite3

import httpx
from google.genai.errors import APIError
from langgraph.types import RetryPolicy

from app.config import Settings

_TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def is_transient_error(exc: Exception) -> bool:
    """Return true only for failures likely to succeed on a later attempt."""

    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, APIError):
        return exc.code in _TRANSIENT_STATUS_CODES
    if isinstance(exc, sqlite3.OperationalError):
        message = str(exc).casefold()
        return "locked" in message or "busy" in message
    return False


def build_retry_policy(settings: Settings) -> RetryPolicy:
    return RetryPolicy(
        initial_interval=settings.retry_initial_interval_seconds,
        backoff_factor=2.0,
        max_interval=settings.retry_max_interval_seconds,
        max_attempts=settings.provider_max_attempts,
        jitter=True,
        retry_on=is_transient_error,
    )
