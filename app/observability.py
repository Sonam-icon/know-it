"""Request correlation and concise structured operational logging."""

from __future__ import annotations

import logging
import re
from contextvars import ContextVar, Token
from time import perf_counter
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,100}$")
_request_id: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format=("%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s"),
    )
    request_filter = RequestIdFilter()
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(request_filter)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if _REQUEST_ID_PATTERN.fullmatch(supplied) else str(uuid4())
        token: Token[str] = _request_id.set(request_id)
        started = perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            elapsed_ms = (perf_counter() - started) * 1000
            logging.getLogger("app.http").info(
                "method=%s path=%s status=%s elapsed_ms=%.2f",
                request.method,
                request.url.path,
                status_code,
                elapsed_ms,
            )
            _request_id.reset(token)
