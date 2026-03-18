"""Volume value object representing trading volume."""

from typing import Any, Self

from pydantic import field_validator

from gpt_quant_v2.domain.value_objects.base import ValueObject


class Volume(ValueObject):
    """Represents trading volume as a non-negative integer.

    Volume represents the number of shares or contracts traded
    during a specific time period.
    """

    value: int

    @field_validator("value")
    @classmethod
    def validate_volume(cls, v: int) -> int:
        """Validate volume is non-negative."""
        if v < 0:
            raise ValueError("Volume cannot be negative")
        return v

    def __add__(self, other: Self) -> "Volume":
        """Add two volumes together."""
        if not isinstance(other, Volume):
            return NotImplemented
        return Volume(value=self.value + other.value)

    def __sub__(self, other: Self) -> "Volume":
        """Subtract another volume from this volume."""
        if not isinstance(other, Volume):
            return NotImplemented
        result = self.value - other.value
        if result < 0:
            raise ValueError("Resulting volume cannot be negative")
        return Volume(value=result)

    def __str__(self) -> str:
        """Return formatted volume string."""
        return f"{self.value:,}"

    def __hash__(self) -> int:
        """Make volume hashable."""
        return hash(self.value)

    def __eq__(self, other: Any) -> bool:
        """Compare volumes by value."""
        if isinstance(other, Volume):
            return self.value == other.value
        if isinstance(other, int):
            return self.value == other
        return NotImplemented

    def __lt__(self, other: Self) -> bool:
        """Compare volumes for less than."""
        if not isinstance(other, Volume):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: Self) -> bool:
        """Compare volumes for less than or equal."""
        if not isinstance(other, Volume):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other: Self) -> bool:
        """Compare volumes for greater than."""
        if not isinstance(other, Volume):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other: Self) -> bool:
        """Compare volumes for greater than or equal."""
        if not isinstance(other, Volume):
            return NotImplemented
        return self.value >= other.value

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Volume({self.value})"
