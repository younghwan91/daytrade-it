"""Base value object class for domain models."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class ValueObject(BaseModel):
    """Base class for all value objects.

    Value objects are immutable, hashable, and comparable by value.
    They represent descriptive aspects of the domain with no conceptual identity.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    def __hash__(self) -> int:
        """Make value object hashable based on its fields."""
        return hash(tuple(sorted(self.model_dump().items())))

    def __eq__(self, other: Any) -> bool:
        """Compare value objects by their values."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.model_dump() == other.model_dump()

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        fields = ", ".join(f"{k}={v!r}" for k, v in self.model_dump().items())
        return f"{self.__class__.__name__}({fields})"
