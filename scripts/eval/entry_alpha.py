"""Re-test the entry features against ALPHA, not raw return.

edge-hunt priced the whole top-300 cross-section at every acting minute and found
a monotone open-to-close headwind: a news-free position at 09:00 earns -56.8 bp
and at 13:00+ earns +5.6 bp. So a feature correlated with time of day picks up
62 bp of that curve before any signal exists, and my volume/turnover test scored
on raw realized return would have absorbed it.

This subtracts their matched null -- day x acting minute x volatility quartile x
liquidity quartile -- from MY live-parity reward, and re-runs the same univariate
and fold ranking on the residual. Their `null` column, my `engine_reward`: the
reward definition stays the one the replay engine produces, only the baseline
changes.

If the volume features showed nothing on raw return they should show nothing here
either; the point of running it is that the reverse is also possible -- a real
signal could have been masked by the curve rather than created by it.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rl_entry_volume import PRICE4, VOL, rank_eval, volume_features
from rl_nonlinear import day_records

from daytrade_it.application.trading import bandit

NOTIONAL = 100_000.0
TOP_K = 5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=Path, required=True)
    ap.add_argument("--edge-rows", type=Path, required=True)
    args = ap.parse_args()

    from bayes_judge import judge

    nulls: dict[str, float] = {}
    for line in args.edge_rows.read_text().splitlines():
        if line:
            r = json.loads(line)
            if r.get("null") is not None:
                nulls[r["key"]] = float(r["null"])

    rows = []
    for line in args.paths.read_text().splitlines():
        if not line:
            continue
        r = json.loads(line)
        if r.get("engine_reward") is None or r["key"] not in nulls:
            continue
        vf = volume_features(r)
        if vf is None:
            continue
        r["features"] = {**r["features"], **vf}
        r["alpha"] = r["engine_reward"] - nulls[r["key"]]
        rows.append(r)
    rows.sort(key=lambda r: (r["day"], r["decide_at"]))
    total = sum(1 for line in args.paths.read_text().splitlines() if line)
    print(f"joined {len(rows)} of {total} paths to a matched null ({len(rows) / total:.1%})")

    y_raw = np.array([r["engine_reward"] for r in rows])
    y_a = np.array([r["alpha"] for r in rows])
    months = sorted({r["day"][:7] for r in rows})
    print(
        f"raw reward {y_raw.mean() * 1e4:+.0f} bp   matched-null alpha {y_a.mean() * 1e4:+.0f} bp"
    )
    print(f"  the difference is the headwind: {(y_raw.mean() - y_a.mean()) * 1e4:+.0f} bp")

    print("\n=== univariate on ALPHA (raw-return value in brackets)")
    print(
        f"{'feature':18s} {'rho_alpha':>10s} {'rho_raw':>9s} {'top-Q alpha':>12s} {'bot-Q alpha':>12s}"
    )
    for f in (*VOL, "minutes_since_open", "premarket"):
        x = np.array([r["features"][f] for r in rows])
        rx = np.argsort(np.argsort(x)).astype(float)
        ra = np.argsort(np.argsort(y_a)).astype(float)
        rr = np.argsort(np.argsort(y_raw)).astype(float)
        hi = y_a[x >= np.quantile(x, 0.8)]
        lo = y_a[x <= np.quantile(x, 0.2)]
        print(
            f"{f:18s} {np.corrcoef(rx, ra)[0, 1]:>+10.3f} {np.corrcoef(rx, rr)[0, 1]:>+9.3f}"
            f" {hi.mean() * 1e4:>+12.0f} {lo.mean() * 1e4:>+12.0f}"
        )

    families = {
        "volume5": VOL,
        "volume5+price4": VOL + PRICE4,
        "all13+volume5": bandit.FEATURES + VOL,
        "all13": bandit.FEATURES,
    }
    by_day: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_day[r["day"]].append(i)
    base_first = {d: v[:TOP_K] for d, v in by_day.items() if d[:7] != months[0]}

    print("\n=== ranking trained AND scored on alpha, top-5/day, monthly folds")
    print(f"{'family':18s} {'pooled alpha KRW':>18s} {'p(mu>0)':>9s} {'days>=0':>9s}")
    for name, feats in families.items():
        picked, _cells = rank_eval(rows, y_a, feats, months)
        cand = day_records(picked, y_a)
        base = day_records({d: base_first[d] for d in picked}, y_a)
        v = judge({c["day"]: c for c in cand}, {b["day"]: b for b in base})
        print(
            f"{name:18s} {sum(c['net'] for c in cand):>+18,.0f}"
            f" {v['p_mu_gt_0']:>9.3f} {v['days_ge_0']:>4d}/{v['n_days']:<4d}"
        )
    print(f"{'BASE live order':18s} {sum(b['net'] for b in day_records(base_first, y_a)):>+18,.0f}")


if __name__ == "__main__":
    main()
