"""Side-by-side: neural exit arms against the table policy, same population, same cost.

    .venv/bin/python scripts/eval/seqexit_nn_report.py --dir data/eval/exitseq

Reads `seqexit_nn_<arm>.json` (from `seqexit_nn.py`) and `seqexit_table_fresh.json`
(from `seqexit_search.py --track track_d_fresh`). Both were scored on
`paths_fresh.jsonl` with the per-leg `costs.round_trip`, on the folds and the
confirm slice of `FRESH_CONFIRM_WINDOW.json`, so every row below is directly
comparable. Each neural arm's confirm series is its MEDIAN seed of three, and it
is judged against live, hold, the fixed grid and the table policy on the
identical confirm days.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/eval")

from bayes_judge import judge  # noqa: E402

ARMS = ("lin", "mlp8", "mlp8_novol", "mlp8_noma", "mlp8_pooled")


def keyed(days: list[dict]) -> dict[str, dict]:
    return {d["day"]: d for d in days}


def per_trade_bp(days: dict[str, dict]) -> float:
    net = sum(v["net"] for v in days.values())
    n = sum(v["trades"] for v in days.values())
    return net / n / 100_000 * 1e4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/eval/exitseq"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    table = json.loads((args.dir / "seqexit_table_fresh.json").read_text())
    base = {}
    for name in ("novol", "vol"):
        blk = table["arms"].get(name)
        if blk and "confirm" in blk:
            base[f"table_{name}"] = keyed(blk["confirm"]["daily"]["seq"])
    ref = table["arms"]["novol"]["confirm"]["daily"]
    for k in ("fixed", "live", "hold"):
        base[k] = keyed(ref[k])

    print("=== search window, pooled out-of-sample over time-ordered monthly folds (bp/trade)")
    for name in ("novol", "vol"):
        blk = table["arms"].get(name)
        if blk:
            p = blk["search"]["pooled"]
            print(
                f"  table_{name:6s} {p['seq']['mean_bp']:+7.1f}   "
                f"(fixed {p['fixed']['mean_bp']:+.1f}, live {p['live']['mean_bp']:+.1f},"
                f" hold {p['hold']['mean_bp']:+.1f})"
            )
    nn = {}
    for arm in ARMS:
        f = args.dir / f"seqexit_nn_{arm}.json"
        if not f.exists():
            print(f"  {arm:12s} (missing)")
            continue
        blk = json.loads(f.read_text())["arms"][arm]
        nn[arm] = blk
        print(f"  {arm:12s} {blk['pooled']['mean_bp']:+7.1f}   params {blk['n_params']}")

    print("\n=== shuffled-feature null, inner holdout (bp/trade)")
    for name in ("novol", "vol"):
        s = table["arms"].get(name, {}).get("shuffle", {})
        if s.get("n"):
            print(
                f"  table_{name:6s} real test {s['real_test_bp']:+7.2f}  noise mean/p90/max "
                f"{s['shuffled_test']['mean']:+.2f}/{s['shuffled_test']['p90']:+.2f}/"
                f"{s['shuffled_test']['max']:+.2f}"
            )
    for arm, blk in nn.items():
        s = blk["shuffle"]
        if s["noise_test"]["mean"] is not None:
            nt = s["noise_test"]
            verdict = (
                "beats max"
                if s["real_test_bp"] > nt["max"]
                else "beats p90 only"
                if s["real_test_bp"] > nt["p90"]
                else "inside noise"
            )
            print(
                f"  {arm:12s} real test {s['real_test_bp']:+7.2f}  noise mean/p90/max "
                f"{nt['mean']:+.2f}/{nt['p90']:+.2f}/{nt['max']:+.2f}  -> {verdict}"
            )

    print("\n=== frozen confirm slice, touched once (bp/trade; neural = median of 3 seeds)")
    for k, v in base.items():
        print(f"  {k:12s} {per_trade_bp(v):+7.1f}")
    rows = {}
    for arm, blk in nn.items():
        c = blk.get("confirm")
        if not c:
            continue
        seeds = ", ".join(f"{s['confirm_bp']:+.1f}" for s in c["seeds"])
        print(f"  {arm:12s} {c['median_bp']:+7.1f}   seeds [{seeds}]")
        cand = keyed(c["daily"])
        rows[arm] = {}
        for b in ("live", "hold", "fixed", "table_novol"):
            if b in base:
                rows[arm][b] = judge(cand, base[b])

    print("\n=== bayes_judge on the confirm slice (median seed)")
    print(f"  {'arm':12s} {'vs':12s} {'p(mu>0)':>8s} {'mu KRW/d':>10s} {'days>=0':>8s}  adopt")
    for arm, vs in rows.items():
        for b, v in vs.items():
            print(
                f"  {arm:12s} {b:12s} {v['p_mu_gt_0']:8.3f} {v['mu_mean']:+10,.0f}"
                f" {v['days_ge_0']:>3d}/{v['n_days']:<3d}  {v['adopt']}"
            )

    if args.out:
        args.out.write_text(json.dumps({"verdicts": rows}, indent=1, default=str))


if __name__ == "__main__":
    main()
