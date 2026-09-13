"""Custom middleware for FastAPI application."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log requests with correlation IDs.

    Adds a unique correlation ID to each request and logs request/response details.
    The correlation ID is available in request.state.correlation_id.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        # Generate or extract correlation ID
        correlation_id = request.headers.get(
            "X-Correlation-ID",
            str(uuid.uuid4()),
        )
        request.state.correlation_id = correlation_id

        # Bind correlation ID to structlog context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

        # Log request
        await logger.ainfo(
            "Request started",
            method=request.method,
            path=request.url.path,
            query=str(request.query_params) if request.query_params else None,
            client_ip=request.client.host if request.client else None,
        )

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id

        # Log response
        await logger.ainfo(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
        )

        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware to add response timing header.

    Adds X-Response-Time header with request processing duration in milliseconds.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        start_time = time.perf_counter()

        response = await call_next(request)

        # Calculate processing time in milliseconds
        process_time_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Response-Time"] = f"{process_time_ms:.2f}ms"

        return response
