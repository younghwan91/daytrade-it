"""Are we capturing the wrong half of the day?

scalp-it decomposed the drift by prior-day turnover rank over ten years and
found it splits cleanly: the overnight leg (prev close -> open) is positive and
the intraday leg (open -> close) is negative, both scaling with turnover rank.
Our universe is the extreme cell -- and our strategy enters intraday and exits
at the close, which is precisely the negative half.

This measures four things the claim needs before it can be acted on:

  1. The decomposition itself, by year, on our own code -- overnight and
     intraday legs separately for each universe, so the reproduction is ours
     and the year-by-year spread is visible rather than pooled.

  2. **The arithmetic that decides it.** Capturing the overnight leg means a
     round trip per night. The toll is charged per round trip and does not care
     which half of the day it brackets, so `overnight drift - cost` is the
     whole proposal in one number. Holding through the day instead avoids the
     toll and collects the negative intraday leg. There is no third option:
     the two legs alternate and you cannot own one without paying for the
     other, either in tolls or in drift.

  3. Whether our news selection adds anything to the overnight leg. Alpha for
     a close-entry / next-open-exit on the same judged pairs, against a null
     matched the usual way -- same day, same universe, same volatility x
     liquidity cell.

  4. The stale-news objection, which must be measured rather than argued: by
     the close the article is hours old. If alpha survives to a close entry it
     is real; if it decays, the structural advantage is unreachable and that
     should be said plainly.

Both legs come from `daily_bars_adjusted` alone, so the ten-year view is free
and split-safe.

READ-ONLY on kr_quant.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_db import connect  # noqa: E402
from edge_strata import quantile_edges, sigma_at  # noqa: E402

BP = 10_000
#: measured round trip, pool-weighted (edge_cost.py)
COST = 0.00367
NQ = 4
MIN_CELL = 5


def load(cur, lo: dt.date, hi: dt.date):
    cur.execute(
        "SELECT date, code, open, close, trade_value FROM daily_bars_adjusted "
        "WHERE date >= %s AND date <= %s",
        (lo, hi),
    )
    by_date: dict[dt.date, dict[str, tuple[float, float, float]]] = defaultdict(dict)
    hist: dict[str, list[tuple]] = defaultdict(list)
    for date, code, o, c, v in cur.fetchall():
        if o and c and float(o) > 0 and float(c) > 0 and len(code) == 6 and code.isdigit():
            by_date[date][code] = (float(o), float(c), float(v or 0.0))
            hist[code].append((date, float(c), float(v or 0.0)))
    return dict(by_date), hist


def universes(prev: dict[str, tuple[float, float, float]]) -> dict[str, list[str]]:
    ranked = sorted((c for c, (_o, _c, v) in prev.items() if v > 0), key=lambda c: (-prev[c][2], c))
    return {
        "top300 (ours)": ranked[:300],
        "301-600": ranked[300:600],
        "601-900": ranked[600:900],
        "901+": ranked[900:],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-09-12")
    ap.add_argument("--end", default="2026-09-15")
    ap.add_argument("--pairs", default="data/eval/edge/horizon_pairs.jsonl")
    ap.add_argument("--nights", default="1,2,5,10")
    args = ap.parse_args()

    lo, hi = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    conn = connect()
    cur = conn.cursor()
    by_date, hist = load(cur, lo - dt.timedelta(days=120), hi)
    cal = sorted(by_date)
    idx = {d: i for i, d in enumerate(cal)}

    # ---- 1. the decomposition, by year
    legs: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for i, day in enumerate(cal):
        if i == 0 or day < lo or day > hi:
            continue
        prev, today = by_date[cal[i - 1]], by_date[day]
        for name, codes in universes(prev).items():
            on = [today[c][0] / prev[c][1] - 1 for c in codes if c in today and prev[c][1] > 0]
            intra = [today[c][1] / today[c][0] - 1 for c in codes if c in today]
            if on:
                legs[(name, "overnight")][day.year].append(statistics.fmean(on))
            if intra:
                legs[(name, "intraday")][day.year].append(statistics.fmean(intra))

    years = sorted({y for v in legs.values() for y in v})
    print(f"\n=== 1. the day, split in two. {args.start}..{args.end}, by year (bp)\n")
    print(
        f"{'universe':<16}{'leg':<11}"
        + "".join(f"{y:>7}" for y in years)
        + f"{'pooled':>9}{'se':>6}"
    )
    for name in ("top300 (ours)", "301-600", "601-900", "901+"):
        for leg in ("overnight", "intraday"):
            per = legs[(name, leg)]
            flat = [v for y in years for v in per.get(y, [])]
            if not flat:
                continue
            cells = "".join(
                f"{statistics.fmean(per[y]) * BP:>7.0f}" if per.get(y) else f"{'.':>7}"
                for y in years
            )
            se = statistics.stdev(flat) / (len(flat) ** 0.5)
            print(f"{name:<16}{leg:<11}{cells}{statistics.fmean(flat) * BP:>+9.1f}{se * BP:>6.1f}")
        tot = [
            a + b
            for y in years
            for a, b in zip(
                legs[(name, "overnight")].get(y, []),
                legs[(name, "intraday")].get(y, []),
                strict=False,
            )
        ]
        if tot:
            print(
                f"{name:<16}{'total':<11}{'':>{7 * len(years)}}{statistics.fmean(tot) * BP:>+9.1f}"
            )
        print()

    # ---- 2. the arithmetic
    print("=== 2. can the good half be owned without paying for the bad half?\n")
    print(f"{'universe':<16}{'overnight':>11}{'- cost':>9}{'intraday':>10}{'hold 1 night+day':>19}")
    for name in ("top300 (ours)", "301-600", "601-900", "901+"):
        o = statistics.fmean([v for y in years for v in legs[(name, "overnight")].get(y, [])])
        d = statistics.fmean([v for y in years for v in legs[(name, "intraday")].get(y, [])])
        print(
            f"{name:<16}{o * BP:>+11.1f}{(o - COST) * BP:>+9.1f}{d * BP:>+10.1f}"
            f"{(o + d - COST) * BP:>+19.1f}"
        )
    print(
        f"\n  a round trip costs {COST * BP:.1f} bp and does not care which half of the day it\n"
        "  brackets. The legs alternate, so owning one means paying a toll per night or\n"
        "  collecting the other leg for free.\n"
    )

    # ---- 3 & 4. does the news add anything at a close entry?
    pairs: dict[dt.date, list[dict]] = defaultdict(list)
    with Path(args.pairs).open() as f:
        for line in f:
            r = json.loads(line)
            pairs[dt.date.fromisoformat(r["day"])].append(r)
    nights = [int(x) for x in args.nights.split(",")]

    ours: dict[int, dict[str, list[float]]] = {n: defaultdict(list) for n in nights}
    null: dict[int, dict[str, list[float]]] = {n: defaultdict(list) for n in nights}
    for day in sorted(pairs):
        i = idx.get(day)
        if i is None or i == 0:
            continue
        prev, today = by_date[cal[i - 1]], by_date[day]
        univ = universes(prev)["top300 (ours)"]
        codes = {r["code"] for r in pairs[day]}
        sig = {c: sigma_at(hist.get(c, []), day) for c in set(univ) | codes}
        u_sig = [sig[c] for c in univ if sig.get(c) is not None]
        if len(u_sig) < NQ * 4:
            continue
        se_, ve_ = quantile_edges(u_sig, NQ), quantile_edges([prev[c][2] for c in univ], NQ)

        def cell(c: str, sig=sig, prev=prev, se_=se_, ve_=ve_) -> tuple[int, int]:
            s, v = sig.get(c), prev.get(c, (0, 0, 0))[2]
            return (
                bisect.bisect_right(se_, s) if s is not None else -1,
                bisect.bisect_right(ve_, v) if v else -1,
            )

        for n in nights:
            j = i + n
            if j >= len(cal):
                continue
            exitd = by_date[cal[j]]
            # close of the article day -> OPEN n sessions later: n overnights,
            # n-1 whole sessions in between
            cells: dict[tuple[int, int], list[float]] = defaultdict(list)
            for c in univ:
                if c in today and c in exitd and today[c][1] > 0:
                    cells[cell(c)].append(exitd[c][0] / today[c][1] - 1)
            means = {k: statistics.fmean(v) for k, v in cells.items() if len(v) >= MIN_CELL}
            for r in pairs[day]:
                c = r["code"]
                if c not in today or c not in exitd or today[c][1] <= 0:
                    continue
                m = means.get(cell(c))
                if m is None:
                    continue
                ours[n][str(day)].append(exitd[c][0] / today[c][1] - 1)
                null[n][str(day)].append(m)
    conn.close()

    print("=== 3/4. close of the article day -> open n sessions later, our judged pairs\n")
    print(f"{'nights':<9}{'n':>9}{'days':>6}{'ours':>9}{'null':>9}{'alpha':>9}{'se':>7}{'net':>9}")
    for n in nights:
        common = sorted(set(ours[n]) & set(null[n]))
        if not common:
            continue
        o = [statistics.fmean(ours[n][d]) for d in common]
        nl = [statistics.fmean(null[n][d]) for d in common]
        a = [x - y for x, y in zip(o, nl, strict=True)]
        se = statistics.stdev(a) / (len(a) ** 0.5) if len(a) > 1 else float("nan")
        npairs = sum(len(ours[n][d]) for d in common)
        print(
            f"{n:<9}{npairs:>9,}{len(common):>6}{statistics.fmean(o) * BP:>+9.1f}"
            f"{statistics.fmean(nl) * BP:>+9.1f}{statistics.fmean(a) * BP:>+9.1f}{se * BP:>7.1f}"
            f"{(statistics.fmean(o) - COST) * BP:>+9.1f}"
        )
    print(
        "\n  `alpha` is our selection over a matched null on the SAME close-to-open\n"
        "  window, so the overnight drift is differenced out and what is left is\n"
        "  whether the news picks better names for that window.\n"
    )


if __name__ == "__main__":
    main()
