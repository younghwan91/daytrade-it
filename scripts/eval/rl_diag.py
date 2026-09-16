"""Univariate, month-by-month diagnostics on the per-pair bandit reward dataset.

Answers one question before any model is fitted: does any decision-time feature
have a relationship to the per-pair reward whose *direction* survives a change
of month? Sign stability across disjoint months is the minimum a selection rule
needs; a feature that flips is noise being read as signal.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from daytrade_it.application.trading.bandit import FEATURES


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [r for r in rows if r["reward"] is not None and r["features"] is not None]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(set(x.tolist())) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float(rx @ ry / denom) if denom else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = load(args.dataset)
    months = sorted({r["day"][:7] for r in rows})
    y_all = np.array([r["reward"] for r in rows])

    print(f"rows={len(rows)} days={len({r['day'] for r in rows})} months={months}")
    print(
        f"reward mean={y_all.mean():+.5f} sd={y_all.std():.5f} win={float((y_all > 0).mean()):.3f}"
    )
    wins = y_all[y_all > 0]
    losses = y_all[y_all <= 0]
    print(
        f"  avg win={wins.mean():+.5f} (n={len(wins)})  avg loss={losses.mean():+.5f} (n={len(losses)})"
    )

    report: dict = {"months": months, "features": {}}
    print()
    print(
        "per-month Spearman(feature, reward)  [sign stability = share of months on the modal sign]"
    )
    hdr = f"{'feature':26s} " + " ".join(f"{m[-2:]:>7s}" for m in months) + "   all   stab"
    print(hdr)
    for f in FEATURES:
        per_month = []
        for m in months:
            sub = [r for r in rows if r["day"][:7] == m]
            x = np.array([r["features"][f] for r in sub])
            yy = np.array([r["reward"] for r in sub])
            per_month.append(spearman(x, yy))
        x_all = np.array([r["features"][f] for r in rows])
        rho_all = spearman(x_all, y_all)
        signs = [1 if v > 0 else -1 for v in per_month if v == v]
        stab = max(signs.count(1), signs.count(-1)) / len(signs) if signs else float("nan")
        report["features"][f] = {
            "per_month": per_month,
            "all": rho_all,
            "sign_stability": stab,
        }
        cells = " ".join(f"{v:+7.3f}" if v == v else "    n/a" for v in per_month)
        print(f"{f:26s} {cells} {rho_all:+6.3f}  {stab:.2f}")

    print()
    print("tercile mean reward by feature, per month (low / mid / high), in bp")
    for f in FEATURES:
        line = [f"{f:26s}"]
        for m in months:
            sub = [r for r in rows if r["day"][:7] == m]
            x = np.array([r["features"][f] for r in sub])
            yy = np.array([r["reward"] for r in sub])
            qs = np.quantile(x, [1 / 3, 2 / 3])
            lo = yy[x <= qs[0]]
            hi = yy[x > qs[1]]
            d = (hi.mean() - lo.mean()) * 1e4 if len(lo) and len(hi) else float("nan")
            line.append(f"{d:+7.0f}" if d == d else "    n/a")
        print(" ".join(line) + "   <- (high tercile - low tercile), bp")

    # Day-level: how much of the variance is the day, not the pair?
    by_day = defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r["reward"])
    day_means = [statistics.fmean(v) for v in by_day.values()]
    within = [r["reward"] - statistics.fmean(by_day[r["day"]]) for r in rows]
    print()
    print(
        f"day-mean sd={statistics.pstdev(day_means):.5f}  within-day sd={statistics.pstdev(within):.5f}"
        f"  -> day share of variance="
        f"{statistics.pvariance(day_means) / (statistics.pvariance(day_means) + statistics.pvariance(within)):.3f}"
    )
    print(f"positive days: {sum(m > 0 for m in day_means)}/{len(day_means)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
