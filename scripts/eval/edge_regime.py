"""Is there a regime, known before the open, in which the day is worth trading?

A quarter of this system's reward variance sits at the day level, and the
ladder shows why: most of what a pair earns is the drift of the whole universe
from the decision minute to the close. So the question "which pair" may be the
wrong one and "which day" the right one.

Every feature here is built only from sessions strictly before the day being
scored -- prev-day index return, 5- and 20-session index returns, the index's
position against its 20-session mean, its trailing 20-session volatility, and
the breadth of the universe (share of top-300 names up) on the previous day.
Nothing from the day itself.

Three targets are reported side by side, because they answer different
questions:
    net    the pool's own net return that day -- what we would have earned
    null   the matched universe null that day -- market timing, no news
    alpha  net - null - cost -- selection, with the day differenced out
If a regime moves `net` and `null` together and leaves `alpha` flat, it is a
market-timing signal that has nothing to do with news, and it should be said
that way rather than sold as a news result.

READ-ONLY on kr_quant.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_db import connect  # noqa: E402
from edge_feat import COST  # noqa: E402
from edge_null import trading_days  # noqa: E402

BP = 10_000


def market_history(
    cur, lo: dt.date, hi: dt.date
) -> tuple[list[tuple[dt.date, float]], dict[dt.date, float]]:
    """The universe's own daily index, and its breadth.

    069500 is an ETF and is not in `daily_bars_adjusted`, so the market here is
    the thing we actually trade against: the equal-weight top-300 by that day's
    traded value, chained into a level series, plus the share of it that closed
    above its open.
    """
    cur.execute(
        "SELECT date, code, open, close, trade_value FROM daily_bars_adjusted "
        "WHERE date >= %s AND date <= %s",
        (lo, hi),
    )
    per: dict[dt.date, list[tuple[float, float, float]]] = defaultdict(list)
    for date, code, o, c, v in cur.fetchall():
        if o and c and v and float(o) > 0 and float(v) > 0 and len(code) == 6 and code.isdigit():
            per[date].append((float(v), float(o), float(c)))
    level, series, brd = 1.0, [], {}
    for date in sorted(per):
        top = sorted(per[date], key=lambda t: -t[0])[:300]
        if not top:
            continue
        level *= 1 + statistics.fmean(c / o - 1 for _v, o, c in top)
        series.append((date, level))
        brd[date] = sum(c > o for _v, o, c in top) / len(top)
    return series, brd


def regimes(cur, days: list[dt.date]) -> dict[str, dict[dt.date, float]]:
    lo = min(days) - dt.timedelta(days=90)
    hi = max(days)
    idx, brd = market_history(cur, lo, hi)
    cal = [d for d, _ in idx]
    pos = {d: i for i, d in enumerate(cal)}
    closes = [c for _d, c in idx]

    out: dict[str, dict[dt.date, float]] = defaultdict(dict)
    for day in days:
        # the last session strictly before `day`
        prior = [d for d in cal if d < day]
        if len(prior) < 25:
            continue
        i = pos[prior[-1]]
        w = closes[: i + 1]
        rets = [w[k] / w[k - 1] - 1 for k in range(1, len(w))]
        out["idx_1d"][day] = rets[-1]
        out["idx_5d"][day] = w[-1] / w[-6] - 1
        out["idx_20d"][day] = w[-1] / w[-21] - 1
        out["idx_vs_ma20"][day] = w[-1] / statistics.fmean(w[-20:]) - 1
        out["idx_vol20"][day] = statistics.pstdev(rets[-20:])
        if prior[-1] in brd:
            out["breadth"][day] = brd[prior[-1]]
    return dict(out)


def tercile(values: dict[dt.date, float]) -> dict[dt.date, str]:
    s = sorted(values.values())
    lo, hi = s[len(s) // 3], s[2 * len(s) // 3]
    return {d: ("low" if v < lo else "high" if v >= hi else "mid") for d, v in values.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="data/eval/edge/rows_sweep.jsonl")
    args = ap.parse_args()

    per_day: dict[dt.date, list[dict]] = defaultdict(list)
    with Path(args.rows).open() as f:
        for line in f:
            r = json.loads(line)
            per_day[dt.date.fromisoformat(r["day"])].append(r)
    days = sorted(per_day)

    conn = connect()
    cur = conn.cursor()
    trading_days(cur, days[0], days[-1])  # calendar sanity; raises if the range is empty
    regs = regimes(cur, days)
    conn.close()

    net = {d: statistics.fmean(r["net"] for r in rs) for d, rs in per_day.items()}
    null = {d: statistics.fmean(r["null"] for r in rs) for d, rs in per_day.items()}
    alpha = {d: statistics.fmean(r["alpha"] for r in rs) for d, rs in per_day.items()}

    print(
        f"\n{len(days)} days   net {statistics.fmean(net.values()) * BP:+.1f} bp   "
        f"null {statistics.fmean(null.values()) * BP:+.1f}   "
        f"alpha {statistics.fmean(alpha.values()) * BP:+.1f}\n"
    )
    print(
        f"{'regime':<14}{'level':<7}{'days':>6}{'net':>9}{'null':>9}{'alpha':>9}"
        f"{'d>=0':>8}{'worst':>10}  months"
    )
    months = sorted({d.strftime("%Y-%m") for d in days})
    for name, values in regs.items():
        if len(values) < len(days) * 0.8:
            continue
        buckets = tercile(values)
        for level in ("low", "mid", "high"):
            sel = [d for d in days if buckets.get(d) == level]
            if len(sel) < 10:
                continue
            marks = "".join(
                "+"
                if (sm := [net[d] for d in sel if d.strftime("%Y-%m") == m])
                and statistics.fmean(sm) > 0
                else ("-" if sm else ".")
                for m in months
            )
            print(
                f"{name:<14}{level:<7}{len(sel):>6}"
                f"{statistics.fmean(net[d] for d in sel) * BP:>+9.1f}"
                f"{statistics.fmean(null[d] for d in sel) * BP:>+9.1f}"
                f"{(statistics.fmean(alpha[d] for d in sel) - COST) * BP:>+9.1f}"
                f"{sum(net[d] >= 0 for d in sel):>5}/{len(sel):<3}"
                f"{min(net[d] for d in sel) * BP:>+10.1f}  {marks}"
            )
        print()


if __name__ == "__main__":
    main()
