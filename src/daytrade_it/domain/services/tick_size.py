"""KRX tick-size (호가가격단위) rules.

The actual table now lives in :mod:`kiwoom_client.tick_size` -- this repo,
scalp-it, and kiwoom-client itself all needed the same KRX price-band table
independently, which is exactly the kind of duplication where one boundary
value drifting out of sync between repos silently corrupts an order price
or a limit-up判定. kiwoom-client is the repo that actually places orders,
so it is the canonical source; this module is a thin adapter that keeps
this package's existing `Exchange`-aware, `Decimal`-typed public API for
callers here, without re-deriving the table.
"""

from __future__ import annotations

from decimal import Decimal

from kiwoom_client.tick_size import round_to_tick as _round_to_tick
from kiwoom_client.tick_size import tick_size as _tick_size

from daytrade_it.domain.entities.stock import Exchange

# KRX board lot size for common stock: 1 share. Odd-lot-only trading was
# abolished; regular KOSPI/KOSDAQ/KONEX orders trade in single-share units.
KRX_LOT_SIZE = 1


def tick_size_for(price: Decimal, market: Exchange) -> Decimal:
    """Return the KRX minimum quote increment (호가가격단위) for a price.

    The table has applied uniformly to KOSPI, KOSDAQ, and KONEX since the
    2023-01-25 unification (see kiwoom_client.tick_size for sources);
    `market` is accepted for forward compatibility and API clarity (and to
    fail loudly on unrecognized exchanges) even though it does not
    currently change the result.

    Args:
        price: The price to quote, in KRW. Must be positive.
        market: The exchange the price is quoted on.

    Returns:
        The minimum tick size (KRW) applicable at that price.

    Raises:
        ValueError: If price is not positive.
    """
    if not isinstance(market, Exchange):
        raise ValueError(f"Unrecognized market: {market!r}")
    return _tick_size(price)


def round_to_tick(price: Decimal, market: Exchange) -> Decimal:
    """Round a price down to the nearest valid KRX tick for its band.

    Args:
        price: The price to round, in KRW. Must be positive.
        market: The exchange the price is quoted on.

    Returns:
        `price` rounded down to the nearest multiple of its tick size.
    """
    if not isinstance(market, Exchange):
        raise ValueError(f"Unrecognized market: {market!r}")
    return _round_to_tick(price)
