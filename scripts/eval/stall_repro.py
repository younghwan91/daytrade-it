"""Gate: the two new exit rules must read only bars at or after entry.

Originally a bug reproduction -- both rules measured their peak from 09:00, so a
position that never gained anything fired the break-even stop on a spike that
happened hours before entry. Fixed on main in `69bfc1a`; this now stands as the
regression gate the lab runs before any grid, in the same spirit as the
`exit_rule_hit` twin's exactness check.

Run from a worktree at main:  uv run python scripts/eval/stall_repro.py

The case: a stock runs 100 -> 110 in the morning; we enter at 100 in the
afternoon; the position is then flat and never gains. Both rules must decline,
because the position's own peak since entry is 0.0%.

Before the fix, both fired:

- The break-even stop arms on a gain the POSITION never made, and then fires at
  entry x (1 + buffer) on a flat position. That inverts the constraint it exists
  to enforce -- it turns a scratch into a small loss rather than preventing a
  winner becoming a loser -- and it does so most eagerly on names the no-chase
  filter admits after a morning run-up.
- The stall rule's `earlier_peak` is the day high, which the last N bars can
  essentially never exceed, so for an afternoon entry it degenerates into "sell
  as soon as you are up min_gain".

The fix takes `entered_at` and measures the peak over `_after_entry`, matching
what `exit_rule_hit`'s trailing rule already did ("the entry minute's high can
predate the fill").
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal as D

from daytrade_it.application.trading.auto_trader import breakeven_stop_hit, stall_exit_hit
from daytrade_it.infrastructure.brokers.base import MinuteBar

DAY = dt.date(2026, 9, 10)


def bar(minute: int, o: float, h: float, low: float, c: float, v: int = 1000) -> MinuteBar:
    return MinuteBar(
        ts=dt.datetime.combine(DAY, dt.time(minute // 60, minute % 60)),
        open=D(str(o)),
        high=D(str(h)),
        low=D(str(low)),
        close=D(str(c)),
        volume=v,
    )


def main() -> None:
    # 09:00-09:10 the stock runs 100 -> 110. We enter at 100 at 13:00.
    # From 13:00 the position is flat: it never gains anything.
    bars = [bar(9 * 60 + i, 100 + i, 100 + i, 99 + i, 100 + i) for i in range(11)]
    bars += [bar(m, 100, 100.1, 99.9, 100.0) for m in range(13 * 60, 13 * 60 + 10)]
    entry = D("100")
    now = dt.datetime.combine(DAY, dt.time(13, 9))

    entered_at = dt.datetime.combine(DAY, dt.time(13, 0))

    hit, peak_gain, close = breakeven_stop_hit(bars, entry, now, D("0.02"), D("0.003"), entered_at)
    print(f"breakeven_stop_hit: hit={hit}  peak_gain={peak_gain}  close={close}")
    assert not hit, "break-even stop armed on a gain the position never made"

    hit, gain = stall_exit_hit(bars, entry, now, D("0.005"), 5, entered_at)
    print(f"stall_exit_hit:     hit={hit}  gain={gain}")
    assert not hit, "stall fired on a peak that predates the entry"

    print("\nBoth decline, as they must: the position's peak since entry is 0.0%.")


if __name__ == "__main__":
    main()
