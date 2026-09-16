"""Does delaying entry help? The news decay against the intraday drift.

edge-hunt priced the whole top-300 cross-section at every acting minute over 125
days: buying ANY top-300 name and holding to close returns -56.8 bp at the 09:00
open, -30.4 bp at 09:01-09:29, -6.3 bp at 11:00-12:59 and +5.6 bp after 13:00.
Monotone, market-wide, and 42% of our pairs act at the worst minute.

Note this REVERSES my own dashboard-pool result, where 09:00-09:01 looked best at
+166 bp. That was 7 days, it failed day-clustering (t = 1.70 on 6 df), and 3 of
its 47 entries carried 49% of the total. The 125-day cross-sectional number is
the one to believe.

The tension this measures, on OUR pairs rather than the cross-section: waiting
buys into a smaller headwind but acts on staler news. Both effects are on the
same pairs here, so neither is assumed to dominate.

Live parity for a delayed entry:
- the no-chase filter is re-applied at the delayed minute, because live would
  (a stock that has run past the cap is refused, and delay makes that more
  likely). Prior close is reconstructed from the recorded `pre_entry_move` and
  the price at the decision minute.
- `entry_end` (14:30) still applies: a delay past it is simply no trade.
- exits stay live-default (VWAP break + 15:00 flatten), via main's own
  `vwap_exit_hit`.

Reported per-pair AND through the live capital budget (5 slots, chronological
order), because the slot mechanics dominated the `entry_end` sweep and could make
a later `entry_start` a no-op or a different book entirely.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

from daytrade_it.application.trading.auto_trader import TradingLimits, vwap_exit_hit
from daytrade_it.infrastructure.brokers.base import MinuteBar

SELL_COST = 0.00215
BUY_COST = 0.00015
FLATTEN = 15 * 60
ENTRY_END = 14 * 60 + 30
MAX_PRE_MOVE = 0.02
NOTIONAL = 100_000.0
TOP_K = 5


def bars_of(row: dict) -> tuple[list[MinuteBar], dict[int, MinuteBar]]:
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
    return bars, {b.ts.hour * 60 + b.ts.minute: b for b in bars}


def hold_from(row: dict, bars: list[MinuteBar], by_min: dict, m: int) -> float | None:
    """Reward of entering at minute `m` and holding under the live exits."""
    bar = by_min.get(m)
    if bar is None or m > ENTRY_END:
        return None
    entry = float(bar.open)
    if entry <= 0:
        return None
    day = dt.date.fromisoformat(row["day"])
    lim = TradingLimits()
    later = sorted(x for x in by_min if x > m)
    at_flat = [x for x in later if x >= FLATTEN]
    close_at = at_flat[0] if at_flat else (later[-1] if later else m)
    real = bool(at_flat)
    for x in later:
        if x > close_at:
            break
        b = by_min[x]
        if x == close_at:
            px = float(b.open) if real else float(b.close)
            return (px * (1 - SELL_COST) - entry * (1 + BUY_COST)) / entry
        now = dt.datetime.combine(day, dt.time(x // 60, x % 60))
        hit, _c, _v = vwap_exit_hit(
            bars, Decimal(str(entry)), now, lim.vwap_exit_gap, lim.vwap_exit_min_gain
        )
        if hit:
            px = float(b.open)
            return (px * (1 - SELL_COST) - entry * (1 + BUY_COST)) / entry
    return None


def entry_minute(row: dict, by_min: dict, spec: tuple[str, int]) -> int | None:
    kind, val = spec
    base = int(row["buy_minute"])
    want = base if kind == "base" else (base + val if kind == "delay" else max(base, val))
    later = sorted(x for x in by_min if x >= want)
    return later[0] if later else None


def passes_no_chase(row: dict, by_min: dict, m: int) -> bool:
    """Re-apply the no-chase cap at the (possibly delayed) entry minute."""
    dec = by_min.get(int(row["buy_minute"]))
    bar = by_min.get(m)
    if dec is None or bar is None:
        return False
    prior = float(dec.open) / (1.0 + float(row["pre_entry_move"]))
    if prior <= 0:
        return False
    return (float(bar.open) / prior - 1.0) <= MAX_PRE_MOVE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=Path, required=True)
    ap.add_argument("--label", default="pool")
    args = ap.parse_args()

    from bayes_judge import judge

    rows = [json.loads(line) for line in args.paths.read_text().splitlines() if line]
    rows = [r for r in rows if r.get("engine_reward") is not None]
    cache = {r["key"]: bars_of(r) for r in rows}
    print(f"{args.label}: {len(rows)} entries, {len({r['day'] for r in rows})} days")
    acts = np.array([int(r["buy_minute"]) - 9 * 60 for r in rows])
    print(
        f"  act minute: median {np.median(acts):.0f}, share at <=1 min after open"
        f" {float((acts <= 1).mean()):.3f}"
    )

    specs: list[tuple[str, tuple[str, int]]] = [("base (live)", ("base", 0))]
    specs += [(f"+{n}m", ("delay", n)) for n in (5, 10, 20, 30, 60, 120)]
    specs += [(f"at {h:02d}:00", ("at", h * 60)) for h in (10, 11, 12, 13)]

    print(
        f"\n{'policy':14s} {'n':>5s} {'per-pair bp':>12s} {'win':>6s}"
        f" {'capped trades':>14s} {'capped KRW':>12s} {'p(mu>0)':>9s} {'days>=0':>9s}"
    )
    base_days = None
    for label, spec in specs:
        taken: list[tuple[str, int, float]] = []
        for r in rows:
            bars, by_min = cache[r["key"]]
            m = entry_minute(r, by_min, spec)
            if m is None or not passes_no_chase(r, by_min, m):
                continue
            rew = hold_from(r, bars, by_min, m)
            if rew is not None:
                taken.append((r["day"], m, rew))
        if not taken:
            print(f"{label:14s} {0:>5d}")
            continue
        arr = np.array([t[2] for t in taken])
        by_day: dict[str, list] = defaultdict(list)
        for d, m, rew in taken:
            by_day[d].append((m, rew))
        recs = []
        for d, v in sorted(by_day.items()):
            sel = [rew for _m, rew in sorted(v)[:TOP_K]]
            recs.append(
                {
                    "day": d,
                    "net": NOTIONAL * float(sum(sel)),
                    "trades": len(sel),
                    "losers": int(sum(1 for x in sel if x < 0)),
                    "worst_trade": NOTIONAL * min(sel),
                    "errors": 0,
                    "leftovers": 0,
                }
            )
        if base_days is None:
            base_days = {r["day"]: r for r in recs}
            v = None
        else:
            common = sorted(set(base_days) & {r["day"] for r in recs})
            v = (
                judge(
                    {r["day"]: r for r in recs if r["day"] in common},
                    {d: base_days[d] for d in common},
                )
                if len(common) >= 2
                else None
            )
        print(
            f"{label:14s} {len(arr):>5d} {arr.mean() * 1e4:>+12.0f} {float((arr > 0).mean()):>6.3f}"
            f" {sum(r['trades'] for r in recs):>14d} {sum(r['net'] for r in recs):>+12,.0f}"
            + (f" {v['p_mu_gt_0']:>9.3f} {v['days_ge_0']:>4d}/{v['n_days']:<4d}" if v else "")
        )


if __name__ == "__main__":
    main()
