"""Score an exit rule offline on recorded entry paths, using the live predicate.

The exit decision is `auto_trader.exit_rule_hit` -- the production function,
called here unmodified on the recorded bars. Nothing about stop-loss, trailing,
take-profit, time-stop, ORB or VWAP logic is reimplemented; this module only
walks the clock and models the sell fill.

Clock and fill model, matching the replay contract:

- The position is checked once a minute from the minute after the fill.
- At minute `now`, `exit_rule_hit` sees only bars whose minute ended before
  `now` (its own `b.ts < minute` filter) -- no look-ahead, ever.
- If a rule fires, the sell is placed at `now` and fills at that minute's open
  (the replay contract's "the order minute's open is the bid"). The retry and
  discount ladder is not modeled; it only matters when the first attempt misses.
- Otherwise the position is flattened at `flatten_at` (live default 15:00) at
  that minute's open, and if there is no such bar, at the last bar's close.

Reward is the same quantity `bandit_dataset` produces:
    r = net / buy_value, net = proceeds x (1 - SELL_COST) - buy_value x (1 + BUY_COST)

`parity_check` re-scores the live-default limits and compares against the
rewards `bandit_dataset` recorded through the real engine. Any candidate that
wins here must still be re-run through the replay engine before it is judged.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal

from replay_baseline import BUY_COST, SELL_COST

from daytrade_it.application.trading.auto_trader import TradingLimits, exit_rule_hit
from daytrade_it.infrastructure.brokers.base import MinuteBar

FLATTEN = dt.time(15, 0)


@dataclass(frozen=True)
class Path:
    """One recorded entry: its fill and its ticker's regular-session bars."""

    day: dt.date
    ticker: str
    buy_minute: int
    buy_price: Decimal
    bars: list[MinuteBar]
    features: dict[str, float]

    @classmethod
    def from_row(cls, row: dict) -> Path:
        day = dt.date.fromisoformat(row["day"])
        bars = [
            MinuteBar(
                ts=dt.datetime.combine(day, dt.time(m // 60, m % 60)),
                open=Decimal(str(o)),
                high=Decimal(str(h)),
                low=Decimal(str(lo)),
                close=Decimal(str(c)),
                volume=int(v),
            )
            for m, o, h, lo, c, v in row["bars"]
        ]
        return cls(
            day=day,
            ticker=row["ticker"],
            buy_minute=row["buy_minute"],
            buy_price=Decimal(str(row["buy_price"])),
            bars=bars,
            features=row["features"],
        )


def _net(buy: Decimal, proceeds: Decimal) -> float:
    return float((proceeds * (1 - SELL_COST) - buy * (1 + BUY_COST)) / buy)


def simulate(
    path: Path, lim: TradingLimits, flatten_at: dt.time = FLATTEN
) -> tuple[float, str, int]:
    """(reward, exit reason, exit minute) for holding `path` under `lim`."""
    entered_at = dt.datetime.combine(path.day, dt.time(path.buy_minute // 60, path.buy_minute % 60))
    flat_minute = flatten_at.hour * 60 + flatten_at.minute
    by_minute = {b.ts.hour * 60 + b.ts.minute: b for b in path.bars}

    for m in range(path.buy_minute + 1, flat_minute + 1):
        bar = by_minute.get(m)
        if bar is None:
            continue
        now = dt.datetime.combine(path.day, dt.time(m // 60, m % 60))
        if m == flat_minute:
            return _net(path.buy_price, bar.open), "flatten", m
        hit = exit_rule_hit(path.bars, path.buy_price, entered_at, now, lim)
        if hit is not None:
            return _net(path.buy_price, bar.open), hit, m

    # No bar at or before the flatten minute: settle at the last bar we have.
    tail = [b for b in path.bars if b.ts.hour * 60 + b.ts.minute > path.buy_minute]
    last = tail[-1] if tail else by_minute.get(path.buy_minute)
    if last is None:
        return 0.0, "no_bars", path.buy_minute
    return _net(path.buy_price, last.close), "last_bar", last.ts.hour * 60 + last.ts.minute


def mfe_mae(path: Path, flatten_at: dt.time = FLATTEN) -> tuple[float, float]:
    """Best and worst unrealized return between the fill minute and `flatten_at`.

    Measured on bar highs/lows after the entry minute (the entry minute's own
    high can predate the fill, which is why `exit_rule_hit`'s trailing rule
    starts at entry + 1 too).
    """
    flat_minute = flatten_at.hour * 60 + flatten_at.minute
    after = [b for b in path.bars if path.buy_minute < b.ts.hour * 60 + b.ts.minute <= flat_minute]
    if not after:
        return 0.0, 0.0
    hi = max(b.high for b in after)
    lo = min(b.low for b in after)
    return float(hi / path.buy_price - 1), float(lo / path.buy_price - 1)
