"""Structured logging system with correlation ID support and FastAPI integration."""

from __future__ import annotations

import logging
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog
from structlog.types import EventDict, Processor

if TYPE_CHECKING:
    from collections.abc import Iterator

# Context variable for correlation ID (request tracing)
correlation_id_ctx: ContextVar[str | None] = ContextVar("correlation_id", default=None)

# Module-level state
_configured: bool = False

LogFormat = Literal["json", "console"]


def _add_correlation_id(
    _logger: logging.Logger, _method_name: str, event_dict: EventDict
) -> EventDict:
    """Add correlation ID to log event if present in context."""
    correlation_id = correlation_id_ctx.get()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def _add_caller_info(
    _logger: logging.Logger, _method_name: str, event_dict: EventDict
) -> EventDict:
    """Add caller information (module, function, line) to log event."""
    record = event_dict.get("_record")
    if record:
        event_dict["module"] = record.module
        event_dict["function"] = record.funcName
        event_dict["line"] = record.lineno
    return event_dict


def _get_shared_processors() -> list[Processor]:
    """Get processors shared between all formatters."""
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        _add_correlation_id,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]


def _get_json_processors() -> list[Processor]:
    """Get processors for JSON output format."""
    return [
        *_get_shared_processors(),
        structlog.processors.format_exc_info,
        _add_caller_info,
        structlog.processors.JSONRenderer(),
    ]


def _get_console_processors() -> list[Processor]:
    """Get processors for colored console output format."""
    return [
        *_get_shared_processors(),
        structlog.processors.format_exc_info,
        _add_caller_info,
        structlog.dev.ConsoleRenderer(
            colors=True, exception_formatter=structlog.dev.plain_traceback
        ),
    ]


def _create_file_handler(
    log_file: str | Path,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> RotatingFileHandler:
    """Create a rotating file handler for log output."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    return handler


def configure_logging(
    level: str = "INFO",
    format: LogFormat = "json",  # noqa: A002
    log_file: str | Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    Configure the structured logging system.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format ('json' or 'console')
        log_file: Optional path for file output with rotation
        max_bytes: Maximum size of log file before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)

    Example:
        >>> configure_logging(level="INFO", format="json")
        >>> configure_logging(level="DEBUG", format="console", log_file="logs/app.log")
    """
    global _configured

    # Convert level string to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *_get_shared_processors(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Create formatter with appropriate renderer
    renderer: structlog.typing.Processor
    if format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=True, exception_formatter=structlog.dev.plain_traceback
        )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            _add_correlation_id,
            _add_caller_info,
        ],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = _create_file_handler(
            log_file=log_file,
            max_bytes=max_bytes,
            backup_count=backup_count,
        )
        # File always uses JSON format for machine parsing
        file_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                _add_correlation_id,
                _add_caller_info,
            ],
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        A bound structlog logger

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing trade", ticker="AAPL", quantity=100)
    """
    if not _configured:
        # Auto-configure with defaults if not already configured
        configure_logging()

    return structlog.stdlib.get_logger(name)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def get_correlation_id() -> str | None:
    """Get the current correlation ID from context."""
    return correlation_id_ctx.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID in context."""
    correlation_id_ctx.set(correlation_id)


@contextmanager
def correlation_id_context(correlation_id: str | None = None) -> Iterator[str]:
    """
    Context manager for setting correlation ID within a scope.

    Args:
        correlation_id: Optional correlation ID. If not provided, generates a new one.

    Yields:
        The correlation ID being used

    Example:
        >>> with correlation_id_context() as cid:
        ...     logger.info("Inside context", correlation_id=cid)
    """
    cid = correlation_id or generate_correlation_id()
    token = correlation_id_ctx.set(cid)
    try:
        yield cid
    finally:
        correlation_id_ctx.reset(token)


def bind_context(**kwargs: Any) -> None:
    """
    Bind additional context to all subsequent log messages in the current context.

    Args:
        **kwargs: Key-value pairs to bind

    Example:
        >>> bind_context(user_id="123", tenant="acme")
        >>> logger.info("User action")  # Will include user_id and tenant
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """
    Remove keys from the bound context.

    Args:
        *keys: Keys to remove from context
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


# FastAPI Middleware Integration
class LoggingMiddleware:
    """
    FastAPI/Starlette middleware for request logging with correlation ID support.

    Example:
        >>> from fastapi import FastAPI
        >>> from daytrade_it.shared.logging import LoggingMiddleware
        >>>
        >>> app = FastAPI()
        >>> app.add_middleware(LoggingMiddleware)
    """

    CORRELATION_ID_HEADER = "X-Correlation-ID"

    def __init__(self, app: Any) -> None:
        self.app = app
        self.logger = get_logger("daytrade_it.middleware")

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Import here to avoid circular imports and make starlette optional
        from starlette.requests import Request

        request = Request(scope, receive)

        # Extract or generate correlation ID
        correlation_id = (
            request.headers.get(self.CORRELATION_ID_HEADER) or generate_correlation_id()
        )

        # Set correlation ID in context
        token = correlation_id_ctx.set(correlation_id)

        # Bind request context
        bind_context(
            correlation_id=correlation_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else None,
        )

        # Track response status
        response_status: int | None = None

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message["status"]
                # Add correlation ID to response headers
                headers = list(message.get("headers", []))
                headers.append(
                    (self.CORRELATION_ID_HEADER.lower().encode(), correlation_id.encode())
                )
                message["headers"] = headers
            await send(message)

        try:
            self.logger.info(
                "Request started",
                method=request.method,
                path=request.url.path,
                query_params=str(request.query_params) if request.query_params else None,
            )

            await self.app(scope, receive, send_wrapper)

            self.logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status_code=response_status,
            )
        except Exception as exc:
            self.logger.exception(
                "Request failed",
                method=request.method,
                path=request.url.path,
                error=str(exc),
            )
            raise
        finally:
            clear_context()
            correlation_id_ctx.reset(token)


# Convenience exports
__all__ = [
    "configure_logging",
    "get_logger",
    "generate_correlation_id",
    "get_correlation_id",
    "set_correlation_id",
    "correlation_id_context",
    "bind_context",
    "unbind_context",
    "clear_context",
    "LoggingMiddleware",
]
