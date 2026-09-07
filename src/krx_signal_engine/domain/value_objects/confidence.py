"""Confidence value object representing a confidence level."""

from typing import Any, Self

from pydantic import field_validator

from krx_signal_engine.domain.value_objects.base import ValueObject


class Confidence(ValueObject):
    """Represents a confidence level as a float between 0.0 and 1.0.

    Confidence is used to express certainty in predictions, signals,
    or other analytical outputs.
    """

    value: float

    @field_validator("value")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence is in range [0.0, 1.0]."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    @property
    def is_high(self) -> bool:
        """Check if confidence is high (> 0.7)."""
        return self.value > 0.7

    @property
    def is_low(self) -> bool:
        """Check if confidence is low (< 0.3)."""
        return self.value < 0.3

    @property
    def is_medium(self) -> bool:
        """Check if confidence is medium (0.3 to 0.7 inclusive)."""
        return 0.3 <= self.value <= 0.7

    @property
    def as_percentage(self) -> float:
        """Return confidence as a percentage (0-100)."""
        return self.value * 100

    def __str__(self) -> str:
        """Return formatted confidence string."""
        return f"{self.value:.0%}"

    def __hash__(self) -> int:
        """Make confidence hashable."""
        return hash(self.value)

    def __eq__(self, other: Any) -> bool:
        """Compare confidences by value."""
        if isinstance(other, Confidence):
            return self.value == other.value
        if isinstance(other, float):
            return self.value == other
        return NotImplemented

    def __lt__(self, other: Self) -> bool:
        """Compare confidences for less than."""
        if not isinstance(other, Confidence):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: Self) -> bool:
        """Compare confidences for less than or equal."""
        if not isinstance(other, Confidence):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other: Self) -> bool:
        """Compare confidences for greater than."""
        if not isinstance(other, Confidence):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other: Self) -> bool:
        """Compare confidences for greater than or equal."""
        if not isinstance(other, Confidence):
            return NotImplemented
        return self.value >= other.value

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Confidence({self.value})"
