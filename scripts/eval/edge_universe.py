"""Is the open-to-close headwind ours, or the market's?

Our universe is the top 300 by the PREVIOUS day's traded value -- which is a
momentum screen wearing a liquidity costume: it selects yesterday's movers.
If the -56.8 bp open-to-close drift is a property of that screen, a different
universe changes the project's arithmetic. If it is a property of the Korean
market, it does not, and "pick a different universe" comes off the list.

Same equal-weight open-to-close return (`close / open - 1`, the daily
equivalent of the 09:00 null), computed on the same days for:

  all            every code that traded
  top300         ours: rank 1-300 by the previous day's traded value
  rank 301-600   the next slice down, same screen
  rank 601-900   further down
  top300_calm    ours, minus the third that moved most yesterday -- if the
                 drift is "yesterday's movers give it back", this is where it
                 goes away
  top300_movers  the complement: the third that moved most yesterday

READ-ONLY on kr_quant.
"""

from __future__ import annotations

import argparse
import datetime as dt
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_db import connect  # noqa: E402

BP = 10_000


def load(cur, lo: dt.date, hi: dt.date) -> dict[dt.date, dict[str, tuple[float, float, float]]]:
    """{date: {code: (open, close, trade_value)}} for plain 6-digit codes."""
    cur.execute(
        "SELECT date, code, open, close, trade_value FROM daily_bars_adjusted "
        "WHERE date >= %s AND date <= %s",
        (lo, hi),
    )
    out: dict[dt.date, dict[str, tuple[float, float, float]]] = defaultdict(dict)
    for date, code, o, c, v in cur.fetchall():
        if o and c and float(o) > 0 and len(code) == 6 and code.isdigit():
            out[date][code] = (float(o), float(c), float(v or 0.0))
    return dict(out)


def ew(day: dict[str, tuple[float, float, float]], codes: list[str]) -> float | None:
    rets = [day[c][1] / day[c][0] - 1 for c in codes if c in day]
    return statistics.fmean(rets) if rets else None


#: KRX price bands (2023 revision), won. The narrowest possible spread is one
#: of these, and crossing it costs one tick per round trip.
TICKS = ((2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50), (200_000, 100), (500_000, 500))
TOP_TICK = 1_000
#: sell tax + commission both sides; the spread is what varies by universe
TAX_AND_COMMISSION = 0.0020 + 2 * 0.00015


def tick_cost(day: dict[str, tuple[float, float, float]], codes: list[str]) -> float | None:
    """Mean tick/price over the universe -- the spread floor it would pay.

    A cheaper universe is not a free universe: thinner, lower-priced names sit
    on a coarser relative tick grid, so a universe that drifts less can easily
    cost more to trade. This prices that directly from the entry prices rather
    than assuming it.
    """
    out = []
    for c in codes:
        if c not in day:
            continue
        px = day[c][0]
        if px <= 0:
            continue
        tick = next((t for bound, t in TICKS if px < bound), TOP_TICK)
        out.append(tick / px)
    return statistics.fmean(out) if out else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-03-17")
    ap.add_argument("--end", default="2026-09-15")
    args = ap.parse_args()

    lo = dt.date.fromisoformat(args.start)
    hi = dt.date.fromisoformat(args.end)
    conn = connect()
    cur = conn.cursor()
    bars = load(cur, lo - dt.timedelta(days=10), hi)
    conn.close()

    cal = sorted(bars)
    series: dict[str, list[float]] = defaultdict(list)
    ticks: dict[str, list[float]] = defaultdict(list)
    months: dict[str, list[str]] = defaultdict(list)
    for i, day in enumerate(cal):
        if i == 0 or day < lo:
            continue
        prev, today = bars[cal[i - 1]], bars[day]
        traded = [c for c, (_o, _c, v) in prev.items() if v > 0]
        ranked = sorted(traded, key=lambda c: (-prev[c][2], c))
        groups = {
            "all": list(today),
            "top300 (ours)": ranked[:300],
            "rank 301-600": ranked[300:600],
            "rank 601-900": ranked[600:900],
        }
        top = ranked[:300]
        moved = sorted(top, key=lambda c: -abs(prev[c][1] / prev[c][0] - 1))
        k = len(top) // 3
        groups["top300 movers (top 1/3 by |prev move|)"] = moved[:k]
        groups["top300 calm (bottom 2/3)"] = moved[k:]
        for name, codes in groups.items():
            r = ew(today, codes)
            if r is not None:
                series[name].append(r)
                months[name].append(day.strftime("%Y-%m"))
                t = tick_cost(today, codes)
                if t is not None:
                    ticks[name].append(t)

    all_months = sorted({m for v in months.values() for m in v})
    print(f"\nequal-weight open-to-close, {args.start}..{args.end}")
    print("cost = 0.20% sell tax + 0.03% commission + one tick of spread, per round trip\n")
    print(
        f"{'universe':<42}{'days':>5}{'drift':>8}{'se':>6}{'spread':>8}{'cost':>7}"
        f"{'net':>8}   " + "".join(f"{m[5:]:>7}" for m in all_months) + "  +/-"
    )
    order = [
        "all",
        "top300 (ours)",
        "rank 301-600",
        "rank 601-900",
        "top300 movers (top 1/3 by |prev move|)",
        "top300 calm (bottom 2/3)",
    ]
    for name in order:
        v = series.get(name)
        if not v:
            continue
        se = statistics.stdev(v) / (len(v) ** 0.5)
        spread = statistics.fmean(ticks[name])
        cost = TAX_AND_COMMISSION + spread
        drift = statistics.fmean(v)
        per_month = []
        for m in all_months:
            sel = [x for x, mm in zip(v, months[name], strict=True) if mm == m]
            per_month.append(statistics.fmean(sel) if sel else None)
        cells = "".join(f"{x * BP:>+7.0f}" if x is not None else f"{'.':>7}" for x in per_month)
        pos = sum(1 for x in per_month if x is not None and x > 0)
        have = sum(1 for x in per_month if x is not None)
        print(
            f"{name:<42}{len(v):>5}{drift * BP:>+8.1f}{se * BP:>6.1f}{spread * BP:>8.1f}"
            f"{cost * BP:>7.1f}{(drift - cost) * BP:>+8.1f}   {cells}  {pos}/{have}"
        )
    print(
        "\n`net` is a long-only round trip in that universe with zero selection skill:\n"
        "the drift it hands you, minus what it costs to take it.\n"
    )


if __name__ == "__main__":
    main()
