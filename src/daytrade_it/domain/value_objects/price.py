"""Price value object representing monetary values."""

from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Self

from pydantic import field_validator

from daytrade_it.domain.value_objects.base import ValueObject


class Price(ValueObject):
    """Represents a monetary price value.

    Prices are positive decimal values with a maximum of 2 decimal places,
    associated with a currency (default USD).
    """

    value: Decimal
    currency: str = "USD"

    @field_validator("value", mode="before")
    @classmethod
    def coerce_to_decimal(cls, v: Any) -> Decimal:
        """Convert input to Decimal."""
        if isinstance(v, Decimal):
            return v
        if isinstance(v, (int, float, str)):
            return Decimal(str(v))
        raise ValueError(f"Cannot convert {type(v).__name__} to Decimal")

    @field_validator("value")
    @classmethod
    def validate_price(cls, v: Decimal) -> Decimal:
        """Validate price: must be positive with max 2 decimal places."""
        if v <= 0:
            raise ValueError("Price must be positive")

        # Round to 2 decimal places
        quantized = v.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Check if already had <= 2 decimal places
        if v != quantized:
            raise ValueError("Price cannot have more than 2 decimal places")

        return v

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency code format."""
        v = v.upper()
        if len(v) != 3 or not v.isalpha():
            raise ValueError("Currency must be a 3-letter ISO code")
        return v

    def add(self, other: "Price") -> "Price":
        """Add two prices together."""
        if self.currency != other.currency:
            raise ValueError(
                f"Cannot add prices with different currencies: {self.currency} vs {other.currency}"
            )
        return Price(value=self.value + other.value, currency=self.currency)

    def subtract(self, other: "Price") -> "Price":
        """Subtract another price from this price."""
        if self.currency != other.currency:
            raise ValueError(
                f"Cannot subtract prices with different currencies: {self.currency} vs {other.currency}"
            )
        result = self.value - other.value
        if result <= 0:
            raise ValueError("Resulting price must be positive")
        return Price(value=result, currency=self.currency)

    def multiply(self, factor: int | float | Decimal) -> "Price":
        """Multiply price by a factor."""
        if isinstance(factor, float):
            factor = Decimal(str(factor))
        elif isinstance(factor, int):
            factor = Decimal(factor)

        result = (self.value * factor).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if result <= 0:
            raise ValueError("Resulting price must be positive")
        return Price(value=result, currency=self.currency)

    def __str__(self) -> str:
        """Return formatted price string."""
        return f"{self.currency} {self.value:.2f}"

    def __hash__(self) -> int:
        """Make price hashable."""
        return hash((self.value, self.currency))

    def __eq__(self, other: Any) -> bool:
        """Compare prices by value and currency."""
        if isinstance(other, Price):
            return self.value == other.value and self.currency == other.currency
        return NotImplemented

    def __lt__(self, other: Self) -> bool:
        """Compare prices for less than."""
        if not isinstance(other, Price):
            return NotImplemented
        if self.currency != other.currency:
            raise ValueError("Cannot compare prices with different currencies")
        return self.value < other.value

    def __le__(self, other: Self) -> bool:
        """Compare prices for less than or equal."""
        if not isinstance(other, Price):
            return NotImplemented
        if self.currency != other.currency:
            raise ValueError("Cannot compare prices with different currencies")
        return self.value <= other.value

    def __gt__(self, other: Self) -> bool:
        """Compare prices for greater than."""
        if not isinstance(other, Price):
            return NotImplemented
        if self.currency != other.currency:
            raise ValueError("Cannot compare prices with different currencies")
        return self.value > other.value

    def __ge__(self, other: Self) -> bool:
        """Compare prices for greater than or equal."""
        if not isinstance(other, Price):
            return NotImplemented
        if self.currency != other.currency:
            raise ValueError("Cannot compare prices with different currencies")
        return self.value >= other.value

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Price(value={self.value}, currency='{self.currency}')"
