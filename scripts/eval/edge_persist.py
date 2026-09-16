"""Does news alpha grow with the holding horizon, and does `persistence` predict it?

Every evaluation in this project's history has been same-day. `persistence`
(one_off / multi_quarter / structural) has therefore never been scored on a
horizon where it could express itself -- a "structural" judgment has no way to
show up by 15:00 on the day of the article. Whatever the answer is, that is a
gap in how the prompt was validated, and it was validated on 209 pairs over
10 days.

Two questions, kept apart:

  1. Does alpha grow with horizon at all? If news has lasting impact, arriving
     21.8 minutes late costs little, and the per-round-trip toll amortises:
     36.7 bp is the whole cost at T+0 and 7.3 bp/day at T+5.
  2. Do the judgment fields order the horizon response the way the prompt
     intends -- structural > multi_quarter > one_off, and more so as h grows?

`alpha` is the pair's return minus its own matched null at the SAME horizon,
so the extra drift and extra variance a longer hold brings are removed before
anything is claimed. Reported raw and sigma-normalised, because variance grows
with horizon and an unnormalised comparison across horizons mostly measures
that -- sigma is the trailing 20-session daily volatility ending the previous
session, scaled by sqrt(h+1).

Reported per horizon and per bucket: alpha and its day-clustered se, net per
trade, net per capital-day, the worst day, and the loser ratio. A strategy that
earns more per trade but takes a tenth as many trades may not earn more per
unit of capital, and a longer hold carries overnight gap risk that a mean does
not price.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

BP = 10_000
COST = 0.00367
HORIZONS = ("0", "1", "2", "5", "10", "20", "40")


def day_stats(
    by_day: dict[str, list[float]], block: int = 1, reps: int = 2000, seed: int = 11
) -> tuple[float, float, int, int, float]:
    """Day-clustered mean, with a block bootstrap when windows overlap.

    A +40d hold started today and one started tomorrow share 39 of their 40
    days. Treating those as two independent observations understates the
    standard error by roughly sqrt(40). The moving-block bootstrap resamples
    contiguous runs of `block` days, so the overlap is carried into the
    resamples and the error comes out honest. `block = 1` is the ordinary
    day-clustered case and reproduces the plain standard error.
    """
    days = sorted(by_day)
    means = [statistics.fmean(by_day[d]) for d in days]
    n = len(means)
    mu = statistics.fmean(means)
    if n < 2:
        return mu, float("nan"), n, sum(m >= 0 for m in means), min(means)
    if block <= 1:
        se = statistics.stdev(means) / math.sqrt(n)
        return mu, se, n, sum(m >= 0 for m in means), min(means)

    block = min(block, n)
    rng = random.Random(seed)
    n_blocks = max(1, n // block)
    boots = []
    for _ in range(reps):
        sample: list[float] = []
        for _ in range(n_blocks):
            start = rng.randrange(0, n - block + 1)
            sample += means[start : start + block]
        boots.append(statistics.fmean(sample))
    se = statistics.stdev(boots)
    return mu, se, n, sum(m >= 0 for m in means), min(means)


def summarise(
    rows: list[dict], h: str, use_cell: bool, sigma: dict[str, float] | None
) -> dict | None:
    key = "nc" if use_cell else "na"
    a_by: dict[str, list[float]] = defaultdict(list)
    n_by: dict[str, list[float]] = defaultdict(list)
    z_by: dict[str, list[float]] = defaultdict(list)
    losers = total = 0
    for r in rows:
        if h not in r["r"] or h not in r[key]:
            continue
        alpha = r["r"][h] - r[key][h]
        a_by[r["day"]].append(alpha)
        n_by[r["day"]].append(r["r"][h] - COST)
        total += 1
        losers += r["r"][h] - COST < 0
        if sigma is not None:
            s = sigma.get(f"{r['code']}|{r['day']}")
            if s and s > 0:
                z_by[r["day"]].append(alpha / (s * math.sqrt(int(h) + 1)))
    if total < 200:
        return None
    # windows of a +h hold overlap on h of their h+1 days
    block = int(h) + 1
    a_mu, a_se, n_days, _, _ = day_stats(a_by, block)
    n_mu, _, _, n_ge, worst = day_stats(n_by, block)
    z_mu, z_se = day_stats(z_by, block)[:2] if z_by else (float("nan"), float("nan"))
    days_held = max(int(h), 1)
    return {
        "n": total,
        "days": n_days,
        "alpha": a_mu,
        "alpha_se": a_se,
        "z": z_mu,
        "z_se": z_se,
        "net": n_mu,
        "per_day": n_mu / days_held,
        "worst": worst,
        "losers": losers / total,
        "days_ge0": n_ge,
    }


def row(label: str, s: dict | None) -> str:
    if s is None:
        return f"  {label:<22} (too few)"
    return (
        f"  {label:<22}{s['n']:>8,}{s['alpha'] * BP:>+9.1f}{s['alpha_se'] * BP:>6.1f}"
        f"{s['z']:>+8.3f}{s['z_se']:>6.3f}{s['net'] * BP:>+9.1f}{s['per_day'] * BP:>+9.1f}"
        f"{s['worst'] * BP:>+10.1f}{s['losers']:>8.3f}{s['days_ge0']:>5}/{s['days']:<3}"
    )


HEAD = (
    f"  {'bucket':<22}{'n':>8}{'alpha':>9}{'se':>6}{'z':>8}{'se':>6}"
    f"{'net':>9}{'net/day':>9}{'worst day':>10}{'losers':>8}{'d>=0':>9}"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/eval/edge/horizon_pairs.jsonl")
    ap.add_argument("--master", default="data/eval/prompt_v3/master.jsonl")
    ap.add_argument("--sigma", default="data/eval/prompt_v3/sigma.json")
    ap.add_argument("--cell", action="store_true", help="null = own volatility x liquidity cell")
    ap.add_argument(
        "--common-days",
        action="store_true",
        help="keep only days where EVERY horizon resolves, so a longer horizon is "
        "not silently also an earlier sample",
    )
    args = ap.parse_args()

    sigma = json.loads(Path(args.sigma).read_text()) if Path(args.sigma).exists() else None
    v2: dict[str, dict] = {}
    with Path(args.master).open() as f:
        for line in f:
            m = json.loads(line)
            v2[m["key"]] = m.get("v2") or {}

    rows = []
    with Path(args.pairs).open() as f:
        for line in f:
            r = json.loads(line)
            r["v2"] = v2.get(r["key"], {})
            rows.append(r)

    # A +40d exit needs 40 trading days of future, which only the early days
    # have -- the sample is 2026-03-17..09-15 and today is 2026-09-17. Left
    # alone, "alpha grows with horizon" and "the long horizons are March and
    # April" are the same statement. This makes them separable.
    span = ""
    if args.common_days:
        ok = {r["day"] for r in rows if all(h in r["r"] for h in HORIZONS)}
        dropped = len({r["day"] for r in rows}) - len(ok)
        rows = [r for r in rows if r["day"] in ok]
        span = f"   [common days only: {len(ok)} kept, {dropped} dropped]"

    null_name = "own sigma x liquidity cell" if args.cell else "whole top-300"
    print(
        f"\n{len(rows):,} pairs   null = {null_name}   cost {COST * BP:.1f} bp per round trip{span}"
    )
    print("\n=== 1. alpha by horizon (all pairs)\n" + HEAD)
    for h in HORIZONS:
        label = "same day" if h == "0" else f"+{h}d"
        print(row(label, summarise(rows, h, args.cell, sigma)))

    print("\n=== 2. alpha by persistence x horizon\n")
    for level in ("one_off", "multi_quarter", "structural"):
        sel = [r for r in rows if r["v2"].get("persistence") == level]
        print(f"{level}  (n={len(sel):,})\n" + HEAD)
        for h in HORIZONS:
            label = "same day" if h == "0" else f"+{h}d"
            print(row(label, summarise(sel, h, args.cell, sigma)))
        print()

    print("=== 3. alpha by scale_vs_size x horizon\n")
    for level in ("small", "medium", "large"):
        sel = [r for r in rows if r["v2"].get("scale_vs_size") == level]
        print(f"{level}  (n={len(sel):,})\n" + HEAD)
        for h in ("0", "5", "10"):
            print(row("same day" if h == "0" else f"+{h}d", summarise(sel, h, args.cell, sigma)))
        print()

    print("=== 4. month by month: does the persistence ordering hold out of sample?\n")
    print(
        "    The pooled ordering above is in-sample over all 125 days. A contrast\n"
        "    that is real should keep its sign in most months on its own; one that\n"
        "    is a few months carrying the average should not.\n"
    )
    months = sorted({r["day"][:7] for r in rows})
    lasting = [r for r in rows if r["v2"].get("persistence") in ("multi_quarter", "structural")]
    one_off = [r for r in rows if r["v2"].get("persistence") == "one_off"]
    key = "nc" if args.cell else "na"
    print(f"    {'horizon':<10}{'pooled':>9}  " + "".join(f"{m[5:]:>8}" for m in months) + "   +/-")
    for h in HORIZONS:
        per_month = []
        for m in months:
            la = [
                r["r"][h] - r[key][h]
                for r in lasting
                if r["day"][:7] == m and h in r["r"] and h in r[key]
            ]
            oo = [
                r["r"][h] - r[key][h]
                for r in one_off
                if r["day"][:7] == m and h in r["r"] and h in r[key]
            ]
            per_month.append(
                statistics.fmean(la) - statistics.fmean(oo)
                if len(la) > 50 and len(oo) > 50
                else None
            )
        la_all = [r["r"][h] - r[key][h] for r in lasting if h in r["r"] and h in r[key]]
        oo_all = [r["r"][h] - r[key][h] for r in one_off if h in r["r"] and h in r[key]]
        pooled = statistics.fmean(la_all) - statistics.fmean(oo_all)
        marks = sum(1 for v in per_month if v is not None and v > 0)
        have = sum(1 for v in per_month if v is not None)
        label = "same day" if h == "0" else f"+{h}d"
        cells = "".join(f"{v * BP:>+8.0f}" if v is not None else f"{'.':>8}" for v in per_month)
        print(f"    {label:<10}{pooled * BP:>+9.1f}  {cells}   {marks}/{have}")
    print()

    print("=== 5. the intraday survivor at longer horizons: pre-entry move <= +1%\n")
    for label, sel in (
        ("keep <= +1%", [r for r in rows if (p := r.get("pre_move")) is not None and p <= 0.01]),
        ("drop  > +1%", [r for r in rows if (p := r.get("pre_move")) is not None and p > 0.01]),
    ):
        print(f"{label}  (n={len(sel):,})\n" + HEAD)
        for h in HORIZONS:
            print(row("same day" if h == "0" else f"+{h}d", summarise(sel, h, args.cell, sigma)))
        print()


if __name__ == "__main__":
    main()
