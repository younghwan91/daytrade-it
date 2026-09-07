"""Common API schemas for FastAPI endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(description="Service health status", examples=["healthy"])
    version: str = Field(description="Application version", examples=["0.1.0"])
    timestamp: datetime = Field(description="Current server timestamp")
    database: str = Field(
        default="unknown",
        description="Database connection status",
        examples=["connected", "disconnected"],
    )
    redis: str = Field(
        default="unknown",
        description="Redis connection status",
        examples=["connected", "disconnected"],
    )


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    error_code: str = Field(
        description="Unique error identifier",
        examples=["GPTQ-200"],
    )
    message: str = Field(
        description="Human-readable error message",
        examples=["Validation failed for field 'symbol'"],
    )
    details: dict | None = Field(
        default=None,
        description="Additional error context",
    )
    timestamp: datetime = Field(description="When the error occurred")
    correlation_id: str | None = Field(
        default=None,
        description="Request correlation ID for tracing",
    )


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints."""

    page: int = Field(
        default=1,
        ge=1,
        description="Page number (1-indexed)",
    )
    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of items per page",
    )

    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """Alias for page_size for database queries."""
        return self.page_size


class PaginatedResponse(BaseModel, Generic[T]):  # noqa: UP046
    """Generic paginated response wrapper."""

    items: list[T] = Field(description="List of items in current page")
    total: int = Field(description="Total number of items across all pages")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Number of items per page")
    total_pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether there is a next page")
    has_previous: bool = Field(description="Whether there is a previous page")

    @classmethod
    def create(
        cls,
        items: list[T],
        total: int,
        page: int,
        page_size: int,
    ) -> PaginatedResponse[T]:
        """Factory method to create a paginated response."""
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )
