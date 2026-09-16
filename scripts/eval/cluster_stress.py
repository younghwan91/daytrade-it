"""Does the open-batch effect survive day clustering and a gap control?

+166 bp +/- 63 on 47 entries looks like 2.6 se, but those 47 entries sit in 7 days
and ~24% of per-pair reward variance is day-level, so treating them as independent
overstates the evidence. Two checks:

1. **Day-clustered.** Pair each day's open-batch mean against that same day's
   later-entry mean and test the 7 paired differences. The effective n is 7, not 47.
2. **Gap control.** "We buy gappers at the open and gappers ran that week" is
   regime, not edge. Split the open batch by the stock's overnight gap
   (pre_entry_move at the 09:01 decision is exactly that) and see whether the
   effect survives inside gap strata.

A block bootstrap over DAYS gives the interval that respects the clustering.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from freshness import load


def block_bootstrap(by_day: dict[str, list[float]], reps: int, rng: np.random.Generator) -> tuple:
    days = sorted(by_day)
    means = []
    for _ in range(reps):
        pick = rng.choice(len(days), size=len(days), replace=True)
        vals = [v for i in pick for v in by_day[days[i]]]
        if vals:
            means.append(float(np.mean(vals)))
    a = np.array(means)
    return float(np.quantile(a, 0.05) * 1e4), float(np.quantile(a, 0.95) * 1e4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=20000)
    args = ap.parse_args()
    rng = np.random.default_rng(11)

    rows = load(args.paths, args.events)
    opens = [r for r in rows if r["act_min"] <= 1]
    later = [r for r in rows if r["act_min"] > 1]
    print(f"open batch n={len(opens)}  later n={len(later)}  days={len({r['day'] for r in rows})}")

    def by_day(rs: list[dict]) -> dict[str, list[float]]:
        d: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            d[r["day"]].append(r["engine_reward"])
        return d

    o_day, l_day = by_day(opens), by_day(later)
    print("\n=== paired by day (effective n = days, not entries)")
    print(
        f"{'day':12s} {'n_open':>7s} {'open bp':>9s} {'n_later':>8s} {'later bp':>9s} {'diff bp':>9s}"
    )
    diffs = []
    for d in sorted(set(o_day) & set(l_day)):
        o, lt = np.mean(o_day[d]), np.mean(l_day[d])
        diffs.append(o - lt)
        print(
            f"{d:12s} {len(o_day[d]):>7d} {o * 1e4:>+9.0f} {len(l_day[d]):>8d}"
            f" {lt * 1e4:>+9.0f} {(o - lt) * 1e4:>+9.0f}"
        )
    a = np.array(diffs)
    se = a.std(ddof=1) / np.sqrt(len(a))
    print(
        f"{'MEAN':12s} {'':>7s} {'':>9s} {'':>8s} {'':>9s} {a.mean() * 1e4:>+9.0f}"
        f"   se {se * 1e4:.0f}  t={a.mean() / se:.2f} on {len(a) - 1} df"
        f"  days positive {int((a > 0).sum())}/{len(a)}"
    )

    lo, hi = block_bootstrap(o_day, args.reps, rng)
    lo2, hi2 = block_bootstrap(l_day, args.reps, rng)
    print("\nblock bootstrap over days (90% interval):")
    print(
        f"  open batch  mean {np.mean([v for vs in o_day.values() for v in vs]) * 1e4:+.0f} bp   [{lo:+.0f}, {hi:+.0f}]"
    )
    print(
        f"  later       mean {np.mean([v for vs in l_day.values() for v in vs]) * 1e4:+.0f} bp   [{lo2:+.0f}, {hi2:+.0f}]"
    )

    print("\n=== gap control: open batch split by overnight move (pre_entry_move)")
    g = np.array([r["pre_entry_move"] for r in opens])
    y = np.array([r["engine_reward"] for r in opens])
    edges = np.quantile(g, [1 / 3, 2 / 3])
    for lab, sel in (
        (f"gap <= {edges[0]:+.2%}", g <= edges[0]),
        (f"{edges[0]:+.2%} < gap <= {edges[1]:+.2%}", (g > edges[0]) & (g <= edges[1])),
        (f"gap > {edges[1]:+.2%}", g > edges[1]),
    ):
        v = y[sel]
        print(
            f"{lab:28s} n={int(sel.sum()):>3d} {v.mean() * 1e4:>+8.0f} bp"
            f"  sd {v.std() * 1e4:>5.0f}  win {float((v > 0).mean()):.3f}"
        )
    rx = np.argsort(np.argsort(g)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    print(
        f"spearman(overnight gap, reward) within the open batch: {np.corrcoef(rx, ry)[0, 1]:+.3f}"
    )

    top = sorted(opens, key=lambda r: -r["engine_reward"])[:3]
    tot = sum(r["engine_reward"] for r in opens)
    print(
        f"\nconcentration: the top 3 of {len(opens)} open-batch entries are"
        f" {sum(r['engine_reward'] for r in top) / tot:.0%} of its total reward"
    )
    for r in top:
        print(f"  {r['day']} {r['ticker']} {r['engine_reward'] * 1e4:+.0f} bp")


if __name__ == "__main__":
    main()
