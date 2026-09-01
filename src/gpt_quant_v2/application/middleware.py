"""Middleware components for command and query buses."""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import ValidationError

from .base import Command, Query

logger = logging.getLogger(__name__)

TMessage = TypeVar("TMessage", bound=Command | Query)
TResult = TypeVar("TResult")

NextHandler = Callable[[Any], Awaitable[Any]]


class Middleware(ABC):
    """Abstract base class for middleware components.

    Middleware can intercept and process commands/queries before and after
    they are handled, enabling cross-cutting concerns like logging,
    validation, and timing.
    """

    @abstractmethod
    async def __call__(
        self,
        message: Command | Query,
        next_handler: NextHandler,
    ) -> Any:
        """Process the message and call the next handler in the chain.

        Args:
            message: The command or query being processed.
            next_handler: The next handler in the middleware chain.

        Returns:
            The result from the handler.
        """
        ...


class LoggingMiddleware(Middleware):
    """Middleware that logs all commands and queries.

    Logs the message type and content before execution,
    and the result or any errors after execution.
    """

    def __init__(self, log_level: int = logging.INFO) -> None:
        """Initialize the logging middleware.

        Args:
            log_level: The logging level to use (default: INFO).
        """
        self.log_level = log_level

    async def __call__(
        self,
        message: Command | Query,
        next_handler: NextHandler,
    ) -> Any:
        """Log the message and delegate to the next handler."""
        message_type = type(message).__name__
        correlation_id = getattr(message, "correlation_id", "N/A")

        logger.log(
            self.log_level,
            "Executing %s [correlation_id=%s]: %s",
            message_type,
            correlation_id,
            message.model_dump_json(),
        )

        try:
            result = await next_handler(message)
            logger.log(
                self.log_level,
                "Completed %s [correlation_id=%s] successfully",
                message_type,
                correlation_id,
            )
            return result
        except Exception as e:
            logger.exception(
                "Failed %s [correlation_id=%s]: %s",
                message_type,
                correlation_id,
                str(e),
            )
            raise


class ValidationMiddleware(Middleware):
    """Middleware that validates command and query inputs.

    Ensures that all messages conform to their Pydantic model schemas
    before being processed by handlers.
    """

    async def __call__(
        self,
        message: Command | Query,
        next_handler: NextHandler,
    ) -> Any:
        """Validate the message and delegate to the next handler."""
        try:
            # Re-validate the model to ensure all constraints are met
            message.model_validate(message.model_dump())
        except ValidationError as e:
            message_type = type(message).__name__
            logger.error("Validation failed for %s: %s", message_type, str(e))
            raise ValueError(f"Validation failed for {message_type}: {e}") from e

        return await next_handler(message)


class TimingMiddleware(Middleware):
    """Middleware that tracks execution time of commands and queries.

    Logs the time taken to execute each message, useful for
    performance monitoring and optimization.
    """

    def __init__(
        self,
        slow_threshold_ms: float = 1000.0,
        log_level: int = logging.DEBUG,
    ) -> None:
        """Initialize the timing middleware.

        Args:
            slow_threshold_ms: Threshold in milliseconds above which
                               execution is considered slow (default: 1000ms).
            log_level: The logging level for normal timing logs (default: DEBUG).
        """
        self.slow_threshold_ms = slow_threshold_ms
        self.log_level = log_level

    async def __call__(
        self,
        message: Command | Query,
        next_handler: NextHandler,
    ) -> Any:
        """Measure execution time and delegate to the next handler."""
        message_type = type(message).__name__
        correlation_id = getattr(message, "correlation_id", "N/A")

        start_time = time.perf_counter()
        try:
            result = await next_handler(message)
            return result
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if elapsed_ms > self.slow_threshold_ms:
                logger.warning(
                    "SLOW %s [correlation_id=%s] took %.2fms (threshold: %.2fms)",
                    message_type,
                    correlation_id,
                    elapsed_ms,
                    self.slow_threshold_ms,
                )
            else:
                logger.log(
                    self.log_level,
                    "%s [correlation_id=%s] completed in %.2fms",
                    message_type,
                    correlation_id,
                    elapsed_ms,
                )
