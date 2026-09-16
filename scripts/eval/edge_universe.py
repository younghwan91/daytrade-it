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

    print(f"\nequal-weight open-to-close, {args.start}..{args.end}\n")
    print(f"{'universe':<42}{'days':>6}{'mean bp':>10}{'se':>8}{'median':>9}{'days>=0':>10}")
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
        print(
            f"{name:<42}{len(v):>6}{statistics.fmean(v) * BP:>+10.1f}{se * BP:>8.1f}"
            f"{statistics.median(v) * BP:>+9.1f}{sum(x >= 0 for x in v):>6}/{len(v):<3}"
        )
    print()


if __name__ == "__main__":
    main()
