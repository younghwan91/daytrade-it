"""Command and Query buses for CQRS pattern implementation."""

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from .base import Command, CommandHandler, Query, QueryHandler
from .middleware import Middleware, NextHandler

logger = logging.getLogger(__name__)

TCommand = TypeVar("TCommand", bound=Command)
TQuery = TypeVar("TQuery", bound=Query)
TResult = TypeVar("TResult")


class CommandBus:
    """Bus for dispatching commands to their handlers.

    The CommandBus routes commands to registered handlers and supports
    middleware for cross-cutting concerns like logging, validation, and timing.

    Example:
        bus = CommandBus()
        bus.use(LoggingMiddleware())
        bus.use(ValidationMiddleware())
        bus.register(CreateUserCommand, CreateUserHandler())

        result = await bus.execute(CreateUserCommand(name="John"))
    """

    def __init__(self) -> None:
        """Initialize an empty command bus."""
        self._handlers: dict[type[Command], CommandHandler] = {}
        self._middleware: list[Middleware] = []

    def register(
        self,
        command_type: type[TCommand],
        handler: CommandHandler[TCommand, Any],
    ) -> "CommandBus":
        """Register a handler for a command type.

        Args:
            command_type: The command type to handle.
            handler: The handler instance for this command type.

        Returns:
            The bus instance for method chaining.

        Raises:
            ValueError: If a handler is already registered for this command type.
        """
        if command_type in self._handlers:
            raise ValueError(f"Handler already registered for {command_type.__name__}")

        self._handlers[command_type] = handler
        logger.debug(
            "Registered handler %s for command %s",
            type(handler).__name__,
            command_type.__name__,
        )

        return self

    def use(self, middleware: Middleware) -> "CommandBus":
        """Add middleware to the command processing pipeline.

        Middleware is executed in the order it is added.

        Args:
            middleware: The middleware to add.

        Returns:
            The bus instance for method chaining.
        """
        self._middleware.append(middleware)
        logger.debug("Added middleware %s", type(middleware).__name__)
        return self

    async def execute(self, command: TCommand) -> Any:
        """Execute a command through the middleware pipeline and handler.

        Args:
            command: The command to execute.

        Returns:
            The result from the command handler.

        Raises:
            KeyError: If no handler is registered for the command type.
        """
        command_type = type(command)

        if command_type not in self._handlers:
            raise KeyError(f"No handler registered for {command_type.__name__}")

        handler = self._handlers[command_type]

        # Build the middleware chain
        async def final_handler(cmd: Command) -> Any:
            return await handler.handle(cmd)

        chain = self._build_middleware_chain(final_handler)

        return await chain(command)

    def _build_middleware_chain(
        self,
        final_handler: NextHandler,
    ) -> NextHandler:
        """Build the middleware chain ending with the final handler."""
        chain = final_handler

        # Build chain in reverse order so middleware executes in registration order
        for middleware in reversed(self._middleware):
            next_handler = chain

            async def create_handler(
                mw: Middleware,
                next_h: NextHandler,
            ) -> Callable[[Command], Any]:
                async def handler(cmd: Command) -> Any:
                    return await mw(cmd, next_h)

                return handler

            # Capture middleware and next_handler in closure
            chain = self._create_middleware_handler(middleware, next_handler)

        return chain

    def _create_middleware_handler(
        self,
        middleware: Middleware,
        next_handler: NextHandler,
    ) -> NextHandler:
        """Create a handler function for a middleware component."""

        async def handler(message: Command) -> Any:
            return await middleware(message, next_handler)

        return handler

    def is_registered(self, command_type: type[Command]) -> bool:
        """Check if a handler is registered for a command type."""
        return command_type in self._handlers


class QueryBus:
    """Bus for dispatching queries to their handlers.

    The QueryBus routes queries to registered handlers and supports
    middleware for cross-cutting concerns like logging, caching, and timing.

    Example:
        bus = QueryBus()
        bus.use(LoggingMiddleware())
        bus.use(TimingMiddleware())
        bus.register(GetUserQuery, GetUserHandler())

        user = await bus.execute(GetUserQuery(user_id="123"))
    """

    def __init__(self) -> None:
        """Initialize an empty query bus."""
        self._handlers: dict[type[Query], QueryHandler] = {}
        self._middleware: list[Middleware] = []

    def register(
        self,
        query_type: type[TQuery],
        handler: QueryHandler[TQuery, Any],
    ) -> "QueryBus":
        """Register a handler for a query type.

        Args:
            query_type: The query type to handle.
            handler: The handler instance for this query type.

        Returns:
            The bus instance for method chaining.

        Raises:
            ValueError: If a handler is already registered for this query type.
        """
        if query_type in self._handlers:
            raise ValueError(f"Handler already registered for {query_type.__name__}")

        self._handlers[query_type] = handler
        logger.debug(
            "Registered handler %s for query %s",
            type(handler).__name__,
            query_type.__name__,
        )

        return self

    def use(self, middleware: Middleware) -> "QueryBus":
        """Add middleware to the query processing pipeline.

        Middleware is executed in the order it is added.

        Args:
            middleware: The middleware to add.

        Returns:
            The bus instance for method chaining.
        """
        self._middleware.append(middleware)
        logger.debug("Added middleware %s", type(middleware).__name__)
        return self

    async def execute(self, query: TQuery) -> Any:
        """Execute a query through the middleware pipeline and handler.

        Args:
            query: The query to execute.

        Returns:
            The result from the query handler.

        Raises:
            KeyError: If no handler is registered for the query type.
        """
        query_type = type(query)

        if query_type not in self._handlers:
            raise KeyError(f"No handler registered for {query_type.__name__}")

        handler = self._handlers[query_type]

        # Build the middleware chain
        async def final_handler(qry: Query) -> Any:
            return await handler.handle(qry)

        chain = self._build_middleware_chain(final_handler)

        return await chain(query)

    def _build_middleware_chain(
        self,
        final_handler: NextHandler,
    ) -> NextHandler:
        """Build the middleware chain ending with the final handler."""
        chain = final_handler

        for middleware in reversed(self._middleware):
            chain = self._create_middleware_handler(middleware, chain)

        return chain

    def _create_middleware_handler(
        self,
        middleware: Middleware,
        next_handler: NextHandler,
    ) -> NextHandler:
        """Create a handler function for a middleware component."""

        async def handler(message: Query) -> Any:
            return await middleware(message, next_handler)

        return handler

    def is_registered(self, query_type: type[Query]) -> bool:
        """Check if a handler is registered for a query type."""
        return query_type in self._handlers
