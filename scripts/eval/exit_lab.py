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


def flatten_index(path: Path, flat_minute: int) -> tuple[int, bool]:
    """(minute the position is closed if no rule fires, whether that is a real flatten).

    Live places the exit at 15:00 and retries every minute, more aggressively each
    time, until it fills; continuous trading ends at 15:20. So the position leaves
    at the FIRST tradable minute at or after 15:00, not at 15:00 exactly -- a stock
    that does not print a 15:00 bar is exited on its next print. If the ticker
    prints nothing at or after 15:00 at all, the position is settled at the last
    bar of the day (the replay's leftover valuation).
    """
    minutes = sorted(b.ts.hour * 60 + b.ts.minute for b in path.bars)
    after = [m for m in minutes if m > path.buy_minute]
    if not after:
        return path.buy_minute, False
    at_or_after = [m for m in after if m >= flat_minute]
    return (at_or_after[0], True) if at_or_after else (after[-1], False)


def simulate(
    path: Path, lim: TradingLimits, flatten_at: dt.time = FLATTEN
) -> tuple[float, str, int]:
    """(reward, exit reason, exit minute) for holding `path` under `lim`."""
    entered_at = dt.datetime.combine(path.day, dt.time(path.buy_minute // 60, path.buy_minute % 60))
    flat_minute = flatten_at.hour * 60 + flatten_at.minute
    by_minute = {b.ts.hour * 60 + b.ts.minute: b for b in path.bars}
    close_at, real_flatten = flatten_index(path, flat_minute)
    if close_at == path.buy_minute:
        return 0.0, "no_bars", path.buy_minute

    for m in sorted(by_minute):
        if m <= path.buy_minute or m > close_at:
            continue
        bar = by_minute[m]
        if m == close_at:
            # Sold into this minute if it is a real flatten; otherwise the day ran
            # out and the leftover is valued at the last close.
            price = bar.open if real_flatten else bar.close
            return _net(path.buy_price, price), ("flatten" if real_flatten else "last_bar"), m
        now = dt.datetime.combine(path.day, dt.time(m // 60, m % 60))
        hit = exit_rule_hit(path.bars, path.buy_price, entered_at, now, lim)
        if hit is not None:
            return _net(path.buy_price, bar.open), hit, m

    return 0.0, "no_bars", path.buy_minute


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
