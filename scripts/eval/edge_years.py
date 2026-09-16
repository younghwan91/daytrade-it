"""Ten years of the drift term, by year and by horizon.

The intraday drift is the only rung in this investigation with a real
t-statistic, so the project's conclusion rests on it -- and the six-month
version of it was already shown to be regime once (the "market-wide -30 bp"
that turned out to be -6.4 over a decade). Everything about the multi-day
result rests on the same term at longer horizons: `net = alpha + null - cost`,
and at +40d the null was -697 bp, which is what made a positive alpha lose
money. If that is 2026 and not structure, the multi-day arithmetic changes.

No labels are needed for any of this, which is why it comes first: the drift
and the cost amortisation are properties of the universe, not of the news.

For each trading day the universes are rebuilt from the PREVIOUS day's traded
value, exactly as `select_universe` does, and each is held from that day's open
to the close `h` trading days later:

    ret_h(code) = close(d+h) / open(d) - 1

Reported per year so a single decade-long average cannot repeat the mistake
this script exists to correct.

Caveat on the market reference: 069500 is an ETF and is absent from
`daily_bars_adjusted`, and `sector_index` only begins 2024-01-26, so there is
no index series over this window. "all traded codes, equal weight" stands in
for the market and is labelled as the proxy it is.

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
HORIZONS = (0, 1, 2, 5, 10, 20, 40)

#: Rule changes inside the window that can masquerade as regime shifts.
BREAKS = {
    2020: "short-selling ban from 2020-03-16",
    2021: "ban partially lifted 2021-05-03",
    2023: "KRX tick-size revision (2023-01); ban reimposed 2023-11-06",
    2025: "ban lifted 2025-03-31",
}


def load(cur, lo: dt.date, hi: dt.date) -> dict[dt.date, dict[str, tuple[float, float, float]]]:
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


def universes(prev: dict[str, tuple[float, float, float]]) -> dict[str, list[str]]:
    traded = [c for c, (_o, _c, v) in prev.items() if v > 0]
    ranked = sorted(traded, key=lambda c: (-prev[c][2], c))
    top = ranked[:300]
    moved = sorted(top, key=lambda c: -abs(prev[c][1] / prev[c][0] - 1))
    k = len(top) // 3
    return {
        "all": traded,
        "top300": top,
        "rank601-900": ranked[600:900],
        "movers": moved[:k],
        "calm": moved[k:],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-09-12")
    ap.add_argument("--end", default="2026-09-15")
    ap.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    lo, hi = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    conn = connect()
    cur = conn.cursor()
    bars = load(cur, lo - dt.timedelta(days=15), hi + dt.timedelta(days=90))
    conn.close()

    cal = sorted(bars)
    idx = {d: i for i, d in enumerate(cal)}
    # series[(universe, horizon)][year] -> list of daily universe means
    series: dict[tuple[str, int], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for i, day in enumerate(cal):
        if i == 0 or day < lo or day > hi:
            continue
        prev, today = bars[cal[i - 1]], bars[day]
        groups = universes(prev)
        for h in horizons:
            j = idx[day] + h
            if j >= len(cal):
                continue
            future = bars[cal[j]]
            for name, codes in groups.items():
                rets = [
                    future[c][1] / today[c][0] - 1
                    for c in codes
                    if c in today and c in future and today[c][0] > 0
                ]
                if rets:
                    series[(name, h)][day.year].append(statistics.fmean(rets))

    years = sorted({y for v in series.values() for y in v})
    print(f"\nequal-weight, open of day d to close of day d+h, {args.start}..{args.end}")
    print("069500 is not in daily_bars_adjusted and sector_index begins 2024-01-26,")
    print('so "all" (every traded code, equal weight) stands in for the market.\n')

    for name in ("all", "top300", "rank601-900", "movers", "calm"):
        print(f"--- {name}")
        print(f"{'horizon':<9}" + "".join(f"{y:>8}" for y in years) + f"{'pooled':>10}{'se':>7}")
        for h in horizons:
            per = series[(name, h)]
            cells = "".join(
                f"{statistics.fmean(per[y]) * BP:>8.0f}" if per.get(y) else f"{'.':>8}"
                for y in years
            )
            flat = [v for y in years for v in per.get(y, [])]
            if not flat:
                continue
            se = statistics.stdev(flat) / (len(flat) ** 0.5)
            label = "same day" if h == 0 else f"+{h}d"
            print(f"{label:<9}{cells}{statistics.fmean(flat) * BP:>+10.1f}{se * BP:>7.1f}")
        print()

    print("structural breaks inside the window:")
    for year, what in sorted(BREAKS.items()):
        print(f"  {year}  {what}")
    print(
        "\nse ignores the overlap between consecutive windows at h > 0 and is "
        "optimistic there;\nthe by-year columns are the honest read.\n"
    )


if __name__ == "__main__":
    main()
