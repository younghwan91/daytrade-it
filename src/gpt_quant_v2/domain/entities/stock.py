"""Stock entity representing a tradeable security."""

from decimal import Decimal
from enum import StrEnum

from pydantic import Field

from gpt_quant_v2.domain.entities.base import Entity
from gpt_quant_v2.domain.value_objects import Ticker


class Exchange(StrEnum):
    """Stock exchange identifiers."""

    NASDAQ = "NASDAQ"
    NYSE = "NYSE"
    AMEX = "AMEX"
    ARCA = "ARCA"
    BATS = "BATS"
    OTC = "OTC"


class Stock(Entity):
    """Domain entity representing a stock/security.

    Attributes:
        ticker: Stock ticker symbol (value object)
        name: Company name
        exchange: Stock exchange (NASDAQ, NYSE, etc.)
        sector: Industry sector (optional)
        market_cap: Market capitalization in USD (optional)
    """

    ticker: Ticker
    name: str = Field(min_length=1, max_length=500)
    exchange: Exchange
    sector: str | None = Field(default=None, max_length=200)
    market_cap: Decimal | None = Field(default=None, ge=0)

    def __str__(self) -> str:
        return f"Stock({self.ticker.value}:{self.exchange})"

    def __repr__(self) -> str:
        return (
            f"Stock(ticker={self.ticker!r}, name={self.name!r}, "
            f"exchange={self.exchange!r}, sector={self.sector!r})"
        )

    @property
    def is_large_cap(self) -> bool:
        """Check if stock is large cap (>$10B market cap)."""
        if self.market_cap is None:
            return False
        return self.market_cap >= Decimal("10_000_000_000")

    @property
    def is_mid_cap(self) -> bool:
        """Check if stock is mid cap ($2B-$10B market cap)."""
        if self.market_cap is None:
            return False
        return Decimal("2_000_000_000") <= self.market_cap < Decimal("10_000_000_000")

    @property
    def is_small_cap(self) -> bool:
        """Check if stock is small cap (<$2B market cap)."""
        if self.market_cap is None:
            return False
        return self.market_cap < Decimal("2_000_000_000")
