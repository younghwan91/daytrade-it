"""Univariate scan: does any decision-time attribute buy positive alpha?

For each attribute and each of its levels this reports the mean of
`alpha = ret_close - stratum null`, clustered on the day, alongside the raw net
return so the two questions stay apart:

    alpha > 0   the level picks better names than its own day/minute/volatility
                bucket -- that is selection
    net  > 0    the level is tradeable on its own after the 21.5 bp round trip

A level can have positive alpha and still lose money (the bucket it beats was
falling), and can make money with zero alpha (it rode a rising bucket). Only
the first is an edge; only the second pays. Both are printed.

`--months` runs the same scan month by month so sign stability is visible
before anything is called a finding; a level that is positive in 6 of 6 months
is a different object from one carried by May.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from edge_feat import AXES, load

BP = 10_000


def day_mean(rows: list[tuple[str, float]]) -> tuple[float, float, int, int]:
    """Day-clustered mean, its se, the day count, and days at or above zero."""
    per: dict[str, list[float]] = defaultdict(list)
    for day, v in rows:
        per[day].append(v)
    means = [statistics.fmean(v) for v in per.values()]
    n = len(means)
    mu = statistics.fmean(means)
    se = statistics.stdev(means) / math.sqrt(n) if n > 1 else float("nan")
    return mu, se, n, sum(m >= 0 for m in means)


def scan(rows: list[dict], axis: str, min_pairs: int, min_days: int) -> list[tuple]:
    by_level: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_level[str(r["f"][axis])].append(r)
    out = []
    for level, rs in by_level.items():
        if len(rs) < min_pairs:
            continue
        a_mu, a_se, n_days, d_ge = day_mean([(r["day"], r["alpha"]) for r in rs])
        if n_days < min_days:
            continue
        n_mu, _, _, n_ge = day_mean([(r["day"], r["net"]) for r in rs])
        out.append((level, len(rs), n_days, a_mu, a_se, d_ge, n_mu, n_ge))
    return sorted(out, key=lambda t: -t[3])


def monthly_signs(rows: list[dict], axis: str, level: str) -> str:
    months = sorted({r["month"] for r in rows})
    marks = []
    for m in months:
        rs = [r for r in rows if r["month"] == m and str(r["f"][axis]) == level]
        if len(rs) < 30:
            marks.append(".")
            continue
        mu, _, _, _ = day_mean([(r["day"], r["alpha"]) for r in rs])
        marks.append("+" if mu > 0 else "-")
    return "".join(marks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/eval/prompt_v3/labels.jsonl")
    ap.add_argument("--master", default="data/eval/prompt_v3/master.jsonl")
    ap.add_argument("--null", default="data/eval/edge/null_sweep_strat.jsonl")
    ap.add_argument("--strata", default="data/eval/edge/strata_sweep.jsonl")
    ap.add_argument("--no-strata", action="store_true", help="null = whole universe")
    ap.add_argument("--min-pairs", type=int, default=300)
    ap.add_argument("--min-days", type=int, default=20)
    ap.add_argument("--axes", help="comma-separated subset of AXES")
    ap.add_argument("--dump", help="write the joined rows as jsonl for reuse")
    args = ap.parse_args()

    rows = list(
        load(
            Path(args.labels),
            Path(args.master),
            Path(args.null),
            None if args.no_strata else Path(args.strata),
        )
    )
    a_mu, a_se, n_days, d_ge = day_mean([(r["day"], r["alpha"]) for r in rows])
    n_mu, _, _, n_ge = day_mean([(r["day"], r["net"]) for r in rows])
    print(
        f"\npool {len(rows)} pairs / {n_days} days   "
        f"alpha {a_mu * BP:+.1f} bp (se {a_se * BP:.1f}, {d_ge}/{n_days} days >= 0)   "
        f"net {n_mu * BP:+.1f} bp ({n_ge}/{n_days})\n"
    )

    if args.dump:
        with Path(args.dump).open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    axes = args.axes.split(",") if args.axes else AXES
    for axis in axes:
        res = scan(rows, axis, args.min_pairs, args.min_days)
        if not res:
            continue
        print(f"--- {axis}")
        print(
            f"    {'level':<16}{'n':>7}{'days':>6}{'alpha':>9}{'se':>7}"
            f"{'d>=0':>8}{'net':>9}{'d>=0':>8}  months"
        )
        for level, n, nd, am, ase, dge, nm, nge in res:
            print(
                f"    {level[:16]:<16}{n:>7}{nd:>6}{am * BP:>+9.1f}{ase * BP:>7.1f}"
                f"{dge:>5}/{nd:<3}{nm * BP:>+9.1f}{nge:>5}/{nd:<3}  "
                f"{monthly_signs(rows, axis, level)}"
            )
        print()


if __name__ == "__main__":
    main()
