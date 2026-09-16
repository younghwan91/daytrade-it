"""Was the sigma-scaled exit rejected by a noise-amplifying penalty?

The contextual exit failed adoption on `worst_day_candidate = -230,261 <
-214,007`. That is the one decision we know was made on a weight-discarding
statistic while the SELECTOR had optimised for the same statistic:

    objective(day_sums) = sum(day_sums) - 0.5 * max(0, -min(day_sums))

`sum(day_sums)` is trade-weighted and safe. `min(day_sums)` is not: a min over
days is maximally sensitive to a single unlucky day, and per-day noise is larger
on thin days -- 17 to 75 trades/day on this pool, a 4.4x spread.

Three penalties compared, selection only (the gate is untouched):

  raw       0.5 * max(0, -min(day_sum))                    -- as shipped
  weighted  0.5 * n_mean * max(0, -min(day_sum / n_day))   -- normalise each day
                                                              to a standard size
  robust    0.5 * max(0, -quantile(day_sums, 0.05))        -- a low quantile
                                                              instead of the min

If the rejection survives all three, the penalty's distortion was immaterial
there. If it flips, we wrongly rejected a candidate.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from exit_fast import reduce_path
from exit_lab import FLATTEN
from exit_lab import Path as EntryPath
from rl_exits import ORB_WIDTHS, SIGMA_STOP, SIGMA_TP, Stack, day_records, entry_sigma, grid, short

from daytrade_it.application.trading.auto_trader import TradingLimits


def penalties(days: np.ndarray, rewards: np.ndarray) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for d, r in zip(days.tolist(), rewards.tolist(), strict=True):
        sums[d] += r
        counts[d] += 1
    vals = np.array([sums[d] for d in sums])
    n = np.array([counts[d] for d in sums], dtype=float)
    total = float(vals.sum())
    n_mean = float(n.mean())
    return {
        "raw": total - 0.5 * max(0.0, -float(vals.min())),
        "weighted": total - 0.5 * n_mean * max(0.0, -float((vals / n).min())),
        "robust": total - 0.5 * max(0.0, -float(np.quantile(vals, 0.05))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=Path, required=True)
    args = ap.parse_args()

    from bayes_judge import judge

    rows = [json.loads(line) for line in args.paths.read_text().splitlines() if line]
    paths = [EntryPath.from_row(r) for r in rows]
    flat = FLATTEN.hour * 60 + FLATTEN.minute
    reduced, keep = [], []
    for p, r in zip(paths, rows, strict=True):
        red = reduce_path(p.bars, p.buy_minute, p.buy_price, flat, ORB_WIDTHS)
        if red is not None:
            reduced.append(red)
            keep.append((p, r))
    paths = [p for p, _ in keep]
    rows = [r for _, r in keep]
    days = np.array([r["day"] for r in rows])
    months = sorted({d[:7] for d in days.tolist()})
    stack = Stack(reduced)

    sigma = np.array([entry_sigma(p) for p in paths])
    sigma = np.where(np.isnan(sigma), float(np.nanmedian(sigma)), sigma)
    base = TradingLimits()
    fixed = grid()
    print(f"entries={len(rows)} fixed grid={len(fixed)} ctx grid={len(SIGMA_STOP) * len(SIGMA_TP)}")

    f_scores = {i: stack.rewards(c) for i, c in enumerate(fixed)}
    c_scores = {
        (a, b): stack.rewards(
            base, stop_pct=None if a is None else a * sigma, tp_pct=None if b is None else b * sigma
        )
        for a in SIGMA_STOP
        for b in SIGMA_TP
    }

    for pen in ("raw", "weighted", "robust"):
        ctx_days, fix_days = [], []
        picks = []
        for m in months[1:]:
            tr = np.array([d[:7] < m for d in days.tolist()])
            ev = np.array([d[:7] == m for d in days.tolist()])
            bf = max(f_scores, key=lambda i: penalties(days[tr], f_scores[i][tr])[pen])
            bc = max(c_scores, key=lambda k: penalties(days[tr], c_scores[k][tr])[pen])
            picks.append((m, short(fixed[bf]), bc))
            ctx_days += day_records(days[ev], c_scores[bc][ev])
            fix_days += day_records(days[ev], f_scores[bf][ev])
        v = judge({d["day"]: d for d in ctx_days}, {d["day"]: d for d in fix_days})
        print(
            f"\n[{pen}] ctx vs fixed: p(mu>0)={v['p_mu_gt_0']:.3f} mu={v['mu_mean']:+,.0f}"
            f" days>=0={v['days_ge_0']}/{v['n_days']}"
            f" worst {v['worst_day_candidate']:+,.0f} vs {v['worst_day_baseline']:+,.0f}"
            f" losers {v['loser_ratio_candidate']:.3f} vs {v['loser_ratio_baseline']:.3f}"
            f" adopt={v['adopt']}"
        )
        for r in v["reasons"]:
            print(f"    - {r}")
        print(f"    ctx picks (a,b) per fold: {[p[2] for p in picks]}")


if __name__ == "__main__":
    main()
