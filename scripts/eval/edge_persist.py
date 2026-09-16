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
import statistics
from collections import defaultdict
from pathlib import Path

BP = 10_000
COST = 0.00367
HORIZONS = ("0", "1", "2", "5", "10")


def day_stats(by_day: dict[str, list[float]]) -> tuple[float, float, int, int, float]:
    means = [statistics.fmean(v) for v in by_day.values()]
    n = len(means)
    mu = statistics.fmean(means)
    se = statistics.stdev(means) / math.sqrt(n) if n > 1 else float("nan")
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
    a_mu, a_se, n_days, _, _ = day_stats(a_by)
    n_mu, _, _, n_ge, worst = day_stats(n_by)
    z_mu, z_se = day_stats(z_by)[:2] if z_by else (float("nan"), float("nan"))
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

    null_name = "own sigma x liquidity cell" if args.cell else "whole top-300"
    print(f"\n{len(rows):,} pairs   null = {null_name}   cost {COST * BP:.1f} bp per round trip")
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

    print("=== 4. the intraday survivor at longer horizons: pre-entry move <= +1%\n")
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
