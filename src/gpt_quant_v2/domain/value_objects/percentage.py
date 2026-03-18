"""Percentage value object for representing percentage values."""

from decimal import Decimal
from typing import Any, Self

from gpt_quant_v2.domain.value_objects.base import ValueObject


class Percentage(ValueObject):
    """Represents a percentage value.

    The value is stored as a percentage (e.g., 50.0 for 50%).
    Provides conversions to decimal and basis points.
    """

    value: float

    @property
    def as_decimal(self) -> float:
        """Return percentage as a decimal (e.g., 50% -> 0.5)."""
        return self.value / 100.0

    @property
    def as_basis_points(self) -> float:
        """Return percentage as basis points (e.g., 1% -> 100 bps)."""
        return self.value * 100.0

    def of(self, amount: float | int | Decimal) -> float:
        """Calculate this percentage of an amount.

        Args:
            amount: The amount to calculate the percentage of.

        Returns:
            The calculated percentage value.

        Example:
            >>> p = Percentage(value=10.0)  # 10%
            >>> p.of(200)  # 10% of 200
            20.0
        """
        if isinstance(amount, Decimal):
            return float(amount) * self.as_decimal
        return float(amount) * self.as_decimal

    def __str__(self) -> str:
        """Return formatted percentage string."""
        return f"{self.value:.2f}%"

    def __hash__(self) -> int:
        """Make percentage hashable."""
        return hash(self.value)

    def __eq__(self, other: Any) -> bool:
        """Compare percentages by value."""
        if isinstance(other, Percentage):
            return self.value == other.value
        if isinstance(other, (int, float)):
            return self.value == other
        return NotImplemented

    def __lt__(self, other: Self) -> bool:
        """Compare percentages for less than."""
        if not isinstance(other, Percentage):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: Self) -> bool:
        """Compare percentages for less than or equal."""
        if not isinstance(other, Percentage):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other: Self) -> bool:
        """Compare percentages for greater than."""
        if not isinstance(other, Percentage):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other: Self) -> bool:
        """Compare percentages for greater than or equal."""
        if not isinstance(other, Percentage):
            return NotImplemented
        return self.value >= other.value

    def __add__(self, other: Self) -> "Percentage":
        """Add two percentages."""
        if not isinstance(other, Percentage):
            return NotImplemented
        return Percentage(value=self.value + other.value)

    def __sub__(self, other: Self) -> "Percentage":
        """Subtract another percentage from this percentage."""
        if not isinstance(other, Percentage):
            return NotImplemented
        return Percentage(value=self.value - other.value)

    def __mul__(self, factor: int | float) -> "Percentage":
        """Multiply percentage by a factor."""
        if not isinstance(factor, (int, float)):
            return NotImplemented
        return Percentage(value=self.value * factor)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Percentage({self.value})"
