"""Ticker value object representing a stock ticker symbol."""

import re
from typing import Any

from pydantic import field_validator

from gpt_quant_v2.domain.value_objects.base import ValueObject


class Ticker(ValueObject):
    """Represents a stock ticker symbol.

    A ticker is a 1-5 character uppercase alphabetic string
    used to identify a publicly traded company.
    """

    value: str

    @field_validator("value")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Validate ticker format: 1-5 uppercase letters only."""
        if not v:
            raise ValueError("Ticker cannot be empty")

        # Convert to uppercase
        v = v.upper()

        # Check length
        if not 1 <= len(v) <= 5:
            raise ValueError("Ticker must be 1-5 characters")

        # Check for uppercase letters only (no special chars or numbers)
        if not re.match(r"^[A-Z]+$", v):
            raise ValueError("Ticker must contain only uppercase letters (A-Z)")

        return v

    def __str__(self) -> str:
        """Return the ticker symbol."""
        return self.value

    def __hash__(self) -> int:
        """Make ticker hashable."""
        return hash(self.value)

    def __eq__(self, other: Any) -> bool:
        """Compare tickers by value."""
        if isinstance(other, Ticker):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other.upper()
        return NotImplemented

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Ticker('{self.value}')"
