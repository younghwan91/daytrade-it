"""KRX tick-size (호가가격단위) rules.

KRX unified the price-band tick-size table across KOSPI, KOSDAQ, and KONEX
effective 2023-01-25 -- before that date KOSDAQ used a coarser table at low
price bands, but the current table below applies identically to all three
markets. Sources (see task-9-report.md for the full citation list):
  - KRX host/member notices on the 2023-01-25 호가가격단위 개편 (e.g. broker
    notices such as Samsung Securities' "국내 증권/파생시장 호가가격단위
    변경 안내" and contemporaneous news coverage of the change), which
    describe the table as applying uniformly to 코스피/코스닥/코넥스/주식선물
    from that date forward.

The table (KRW, price P, tick size):
    P <         2,000            ->      1
    2,000  <= P <     5,000       ->      5
    5,000  <= P <    20,000       ->     10
    20,000 <= P <    50,000       ->     50
    50,000 <= P <   200,000       ->    100
    200,000 <= P <   500,000       ->    500
    P >=      500,000             ->  1,000
"""

from __future__ import annotations

from decimal import Decimal

from krx_signal_engine.domain.entities.stock import Exchange

# KRX board lot size for common stock: 1 share. Odd-lot-only trading was
# abolished; regular KOSPI/KOSDAQ/KONEX orders trade in single-share units.
KRX_LOT_SIZE = 1

# (upper_bound_exclusive, tick_size) pairs, ascending by price. The last
# entry's upper bound is unused (falls through as "and above").
_TICK_BANDS: tuple[tuple[Decimal, Decimal], ...] = (
    (Decimal("2000"), Decimal("1")),
    (Decimal("5000"), Decimal("5")),
    (Decimal("20000"), Decimal("10")),
    (Decimal("50000"), Decimal("50")),
    (Decimal("200000"), Decimal("100")),
    (Decimal("500000"), Decimal("500")),
)
_TOP_TICK = Decimal("1000")


def tick_size_for(price: Decimal, market: Exchange) -> Decimal:
    """Return the KRX minimum quote increment (호가가격단위) for a price.

    The table has applied uniformly to KOSPI, KOSDAQ, and KONEX since the
    2023-01-25 unification (see module docstring for sources); `market` is
    accepted for forward compatibility and API clarity (and to fail loudly
    on unrecognized exchanges) even though it does not currently change the
    result.

    Args:
        price: The price to quote, in KRW. Must be positive.
        market: The exchange the price is quoted on.

    Returns:
        The minimum tick size (KRW) applicable at that price.

    Raises:
        ValueError: If price is not positive.
    """
    if price <= 0:
        raise ValueError("price must be positive")
    if not isinstance(market, Exchange):
        raise ValueError(f"Unrecognized market: {market!r}")

    for upper_bound, tick in _TICK_BANDS:
        if price < upper_bound:
            return tick
    return _TOP_TICK


def round_to_tick(price: Decimal, market: Exchange) -> Decimal:
    """Round a price down to the nearest valid KRX tick for its band.

    Args:
        price: The price to round, in KRW. Must be positive.
        market: The exchange the price is quoted on.

    Returns:
        `price` rounded down to the nearest multiple of its tick size.
    """
    tick = tick_size_for(price, market)
    return (price // tick) * tick
