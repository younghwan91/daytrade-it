"""KRX tick-size (호가가격단위) rules.

The table lives in :mod:`kiwoom_client.tick_size` and the market-generic
helpers on top of it (including ``KRX_LOT_SIZE``) in
:mod:`krx_quant_core.market.ticks`, shared with scalp-it -- one boundary
value drifting between repos silently corrupts an order price or a limit-up
check. This module is a thin adapter that keeps this package's existing
`Exchange`-aware, `Decimal`-typed public API for callers here, without
re-deriving the table.
"""

from __future__ import annotations

from decimal import Decimal

from krx_quant_core.market.ticks import KRX_LOT_SIZE
from krx_quant_core.market.ticks import round_to_tick as _round_to_tick
from krx_quant_core.market.ticks import tick_size as _tick_size

from daytrade_it.domain.entities.stock import Exchange

__all__ = ["KRX_LOT_SIZE", "round_to_tick", "tick_size_for"]


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
    # krx-quant-core ships no py.typed marker, so mypy sees its API as Any.
    result: Decimal = _tick_size(price)
    return result


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
    result: Decimal = _round_to_tick(price)
    return result
