"""Base classes for CQRS pattern implementation."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field


class Command(BaseModel):
    """Base class for all commands (write operations).

    Commands represent intent to change state in the system.
    Each command has a correlation_id for tracing across the system.
    """

    correlation_id: str = Field(default_factory=lambda: str(uuid4()))

    class Config:
        frozen = True


class Query(BaseModel):
    """Base class for all queries (read operations).

    Queries represent requests for data that do not modify state.
    """

    class Config:
        frozen = True


TCommand = TypeVar("TCommand", bound=Command)
TQuery = TypeVar("TQuery", bound=Query)
TResult = TypeVar("TResult")


class CommandHandler(ABC, Generic[TCommand, TResult]):
    """Abstract base class for command handlers.

    Command handlers execute commands and return results.
    They are responsible for orchestrating domain logic for write operations.
    """

    @abstractmethod
    async def handle(self, command: TCommand) -> TResult:
        """Execute the command and return the result.

        Args:
            command: The command to execute.

        Returns:
            The result of the command execution.
        """
        ...


class QueryHandler(ABC, Generic[TQuery, TResult]):
    """Abstract base class for query handlers.

    Query handlers execute queries and return results.
    They are responsible for fetching and returning data without side effects.
    """

    @abstractmethod
    async def handle(self, query: TQuery) -> TResult:
        """Execute the query and return the result.

        Args:
            query: The query to execute.

        Returns:
            The result of the query execution.
        """
        ...
