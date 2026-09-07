"""Ticker value object representing a stock ticker symbol."""

import re
from typing import Any

from pydantic import field_validator

from gpt_quant_v2.domain.value_objects.base import ValueObject


class Ticker(ValueObject):
    """Represents a stock ticker symbol.

    A ticker is a 6-digit numeric KRX (Korea Exchange) code used to
    identify a publicly traded company, e.g. "005930" (Samsung
    Electronics). Preferred-share codes are also 6 digits
    (e.g. "005935").
    """

    value: str

    @field_validator("value")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Validate ticker format: 6-digit numeric KRX code."""
        if not v:
            raise ValueError("Ticker cannot be empty")

        # Check for exactly 6 digits (KRX stock code format)
        if not re.match(r"^\d{6}$", v):
            raise ValueError("Ticker must be a 6-digit KRX code (e.g. '005930')")

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
