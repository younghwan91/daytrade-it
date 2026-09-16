"""The round trip, from the broker's own model -- never a literal.

Research across this repo netted out `COST = 0.00215` for months. That constant
is the 0.20% sell tax plus ONE side of commission, with spread and slippage at
zero, and it is 15+ bp below what `PaperBroker` actually charges. Every net
figure computed with it was optimistic by that much. The fix is not a better
literal, it is to stop writing literals: this module delegates to
`krx_quant_core.costs.model.round_trip_cost`, the same code the broker uses, so
research and execution cannot drift apart again.

The one thing the core model cannot know is what the book costs us, because
that is a property of the names we trade and the minute we trade them. The
slippage inputs below are measured, not assumed -- `scripts/eval/edge_spread.py`
over 18 trading days of 10-level depth in kr_quant `quotes`:

    거래소   09:00-09:04  17.2 bp      after 09:05  14.0 bp
    코스닥   09:00-09:04  30.9 bp      after 09:05  19.2 bp

Those are ROUND-TRIP crossing costs -- buy the ask, sell the bid is one spread
-- so each is halved to get the one-way figure the core model doubles back.
Depth at the touch is 180x our order size on KOSDAQ and over 2,000x on 거래소,
so nothing walks the book and the spread is the whole execution cost.

Typical values this produces, per round trip:

    거래소  at the open 40.2 bp,  later 37.0 bp
    코스닥  at the open 53.9 bp,  later 42.2 bp

Usage:

    from costs import round_trip
    net = ret - round_trip(day, market, at_open=act_min < 5)
"""

from __future__ import annotations

import datetime as dt

from krx_quant_core.costs.model import round_trip_cost
from krx_quant_core.market.codes import Market

#: Measured mean touch, round trip, by market and by whether we are inside the
#: first five minutes. `scripts/eval/edge_spread.py`, 2026-08-24..2026-09-16.
MEASURED_SPREAD: dict[tuple[str, bool], float] = {
    ("거래소", True): 0.00172,
    ("거래소", False): 0.00140,
    ("코스닥", True): 0.00309,
    ("코스닥", False): 0.00192,
}

_MARKET = {"거래소": Market.KOSPI, "코스닥": Market.KOSDAQ}


def round_trip(on: dt.date | str, market: str, *, at_open: bool = False) -> float:
    """Total round-trip cost rate for one trade in `market`, sold on `on`.

    `at_open` selects the first-five-minutes spread, which is 3.2 bp wider on
    거래소 and 11.7 bp wider on 코스닥 -- the open is the worst minute to cross
    as well as the worst minute to hold.
    """
    day = dt.date.fromisoformat(on) if isinstance(on, str) else on
    spread = MEASURED_SPREAD.get((market, at_open), MEASURED_SPREAD[("거래소", at_open)])
    return round_trip_cost(day, _MARKET.get(market, Market.KOSPI), slippage_one_way=spread / 2)
