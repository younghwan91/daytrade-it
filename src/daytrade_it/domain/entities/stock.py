"""Stock entity representing a tradeable security."""

from decimal import Decimal
from enum import StrEnum

from pydantic import Field

from daytrade_it.domain.entities.base import Entity
from daytrade_it.domain.value_objects import Ticker


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

    # NOTE (Task 9 / Phase 2 completion, researched 2026-09): is_large_cap /
    # is_mid_cap / is_small_cap below are still disabled, and deliberately so
    # -- not because no research was done, but because the research shows a
    # fixed KRW-amount threshold would misrepresent how KRX actually defines
    # these buckets.
    #
    # KRX's official 대형주/중형주/소형주 (large/mid/small-cap) classification
    # -- used to construct the KOSPI 대형주/중형주/소형주 indices -- is a
    # *market-relative rank* convention, not an absolute KRW cutoff:
    #   - 대형주 (large-cap): KOSPI-wide market-cap rank 1-100
    #   - 중형주 (mid-cap):   KOSPI-wide market-cap rank 101-300
    #   - 소형주 (small-cap): KOSPI-wide market-cap rank 301+
    # Ranks are based on the 3-month average daily market cap as of each
    # semiannual index-rebalancing date (KOSPI 200 futures/options June and
    # December expiry, effective the next trading day), so "large-cap" for
    # one stock is only meaningful relative to every other listed stock's
    # market cap at that same rebalancing date -- it is not a stable KRW
    # number that can be hardcoded on this entity. A single `Stock.market_cap`
    # value has no way to know its own market-wide rank, so any fixed-KRW
    # threshold implemented here (e.g. "large-cap = market_cap > 10 trillion
    # KRW") would be an invented approximation of the real KRX convention,
    # not the convention itself -- exactly the kind of guess the redevelopment
    # plan told Task 5 not to make. Implementing this correctly would require
    # a cross-sectional ranking service (comparing against the full listed
    # universe as of a rebalancing date), which is out of scope for a Stock
    # entity property; it belongs in a market-data/ranking service instead.
    #
    # Sources (see task-9-report.md for full citation list):
    #   - KRX index methodology summaries describing 대형주(1-100위)/
    #     중형주(101-300위)/소형주(301위~) rank-based classification and the
    #     semiannual (6월물/12월물 최종거래일 익일) rebalancing schedule.
    #
    # Left as NotImplementedError rather than guessed at. If/when a ranking
    # service exists, these properties should delegate to it instead of
    # reading `market_cap` in isolation.

    @property
    def is_large_cap(self) -> bool:
        """Whether the stock is large cap.

        Raises:
            NotImplementedError: KRX's large-cap definition is a market-wide
                rank (top 100 by market cap), not a fixed KRW threshold, so
                it cannot be computed from this entity alone. See the NOTE
                above this property for sources and rationale.
        """
        raise NotImplementedError(
            "is_large_cap: KRX defines 대형주 as KOSPI market-cap rank "
            "1-100, a market-relative rank recomputed semiannually, not a "
            "fixed KRW threshold. A single Stock has no way to know its "
            "market-wide rank, so this cannot be answered from market_cap "
            "alone; it needs a ranking service. See stock.py's module notes."
        )

    @property
    def is_mid_cap(self) -> bool:
        """Whether the stock is mid cap.

        Raises:
            NotImplementedError: KRX's mid-cap definition is a market-wide
                rank (101st-300th by market cap), not a fixed KRW threshold,
                so it cannot be computed from this entity alone. See the
                NOTE above this property for sources and rationale.
        """
        raise NotImplementedError(
            "is_mid_cap: KRX defines 중형주 as KOSPI market-cap rank "
            "101-300, a market-relative rank recomputed semiannually, not a "
            "fixed KRW threshold. A single Stock has no way to know its "
            "market-wide rank, so this cannot be answered from market_cap "
            "alone; it needs a ranking service. See stock.py's module notes."
        )

    @property
    def is_small_cap(self) -> bool:
        """Whether the stock is small cap.

        Raises:
            NotImplementedError: KRX's small-cap definition is everything
                below market-wide rank 300, not a fixed KRW threshold, so
                it cannot be computed from this entity alone. See the NOTE
                above this property for sources and rationale.
        """
        raise NotImplementedError(
            "is_small_cap: KRX defines 소형주 as everything below KOSPI "
            "market-cap rank 300, a market-relative rank recomputed "
            "semiannually, not a fixed KRW threshold. A single Stock has no "
            "way to know its market-wide rank, so this cannot be answered "
            "from market_cap alone; it needs a ranking service. See "
            "stock.py's module notes."
        )
