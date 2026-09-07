"""Base Entity class for domain entities."""

from datetime import UTC, datetime
from typing import Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class Entity(BaseModel):
    """Base class for all domain entities.

    Provides:
    - id: Unique identifier (UUID)
    - created_at: Timestamp when entity was created
    - updated_at: Timestamp when entity was last updated
    - version: Optimistic locking version number
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        from_attributes=True,
    )

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: int = Field(default=1, ge=1)

    def __eq__(self, other: object) -> bool:
        """Entities are equal if they have the same id."""
        if not isinstance(other, Entity):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on id for use in sets and dicts."""
        return hash(self.id)

    def bump_version(self) -> Self:
        """Increment version and update timestamp for optimistic locking."""
        self.version += 1
        self.updated_at = datetime.now(UTC)
        return self

    def touch(self) -> Self:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now(UTC)
        return self
