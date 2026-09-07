"""Stock entity representing a tradeable security."""

from decimal import Decimal
from enum import StrEnum

from pydantic import Field

from gpt_quant_v2.domain.entities.base import Entity
from gpt_quant_v2.domain.value_objects import Ticker


class Exchange(StrEnum):
    """Stock exchange identifiers (Korea Exchange markets)."""

    KOSPI = "KOSPI"
    KOSDAQ = "KOSDAQ"
    KONEX = "KONEX"


class Stock(Entity):
    """Domain entity representing a stock/security.

    Attributes:
        ticker: Stock ticker symbol (value object)
        name: Company name
        exchange: Stock exchange (KOSPI, KOSDAQ, KONEX)
        sector: Industry sector (optional)
        market_cap: Market capitalization (currency TBD; see NOTE below, optional)
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

    # NOTE: is_large_cap / is_mid_cap / is_small_cap below are USD-denominated
    # market-cap buckets left over from the US-equity system. `market_cap` is
    # now expected to carry KRW values (KOSPI/KOSDAQ/KONEX), so these
    # thresholds are meaningless as-is and would silently misclassify every
    # stock if left enabled. They are intentionally disabled here rather than
    # guessed at; defining real KRW thresholds is out of scope for this task
    # and is tracked under the redevelopment plan's Phase 2 section (market-cap
    # bucket thresholds / KRW conversion). See .superpowers/sdd/PLAN for details.

    @property
    def is_large_cap(self) -> bool:
        """Whether the stock is large cap.

        Raises:
            NotImplementedError: KRW large-cap threshold not yet defined
                (see Phase 2 of the redevelopment plan).
        """
        raise NotImplementedError(
            "is_large_cap uses stale USD thresholds; KRW market-cap buckets "
            "are defined in Phase 2 of the redevelopment plan, not here."
        )

    @property
    def is_mid_cap(self) -> bool:
        """Whether the stock is mid cap.

        Raises:
            NotImplementedError: KRW mid-cap threshold not yet defined
                (see Phase 2 of the redevelopment plan).
        """
        raise NotImplementedError(
            "is_mid_cap uses stale USD thresholds; KRW market-cap buckets "
            "are defined in Phase 2 of the redevelopment plan, not here."
        )

    @property
    def is_small_cap(self) -> bool:
        """Whether the stock is small cap.

        Raises:
            NotImplementedError: KRW small-cap threshold not yet defined
                (see Phase 2 of the redevelopment plan).
        """
        raise NotImplementedError(
            "is_small_cap uses stale USD thresholds; KRW market-cap buckets "
            "are defined in Phase 2 of the redevelopment plan, not here."
        )
