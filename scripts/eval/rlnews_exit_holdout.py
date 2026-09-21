"""The sigma-scaled contextual exit, out-of-sample on the strictly-prior holdout.

Stages 1, 4 and 5 of `scripts/eval/rl_exits.py` re-run on `paths_prior.jsonl`
(149 days, 2025-08-01..2026-03-16, never used to fit anything), priced both at
the legacy `replay_baseline` literal and at `costs.round_trip` via
`exit_worst_day.recost`, and judged with `bayes_judge.judge` plus the three
worst-day penalties of `exit_worst_day`.

Stages 2 (engine parity) and 3 (MFE/MAE) are skipped: the holdout has no
recorded engine rewards to compare against, and MFE/MAE are already in
`labels_prior.jsonl`.

    .venv/bin/python data/eval/rlnews/prior_holdout.py \
        --paths data/eval/rlnews/paths_prior.jsonl \
        --out data/eval/rlnews/prior_holdout.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path as FsPath

import numpy as np

sys.path.insert(0, "scripts/eval")

from bayes_judge import judge  # noqa: E402
from exit_fast import first_exit, reduce_path  # noqa: E402
from exit_lab import FLATTEN, Path, simulate  # noqa: E402
from exit_worst_day import (  # noqa: E402
    PENALTIES,
    bootstrap_gate,
    gate,
    real_cost,
    recost,
    stock_markets,
)
from replay_baseline import BUY_COST, SELL_COST  # noqa: E402
from rl_exits import (  # noqa: E402
    ORB_WIDTHS,
    SIGMA_BARS,
    SIGMA_STOP,
    SIGMA_TP,
    Stack,
    day_records,
    describe,
    entry_sigma,
    grid,
    objective,
    short,
)

from daytrade_it.application.trading.auto_trader import TradingLimits  # noqa: E402


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=FsPath, required=True)
    ap.add_argument("--out", type=FsPath, required=True)
    ap.add_argument("--verify-configs", type=int, default=12)
    ap.add_argument("--verify-paths", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.paths.read_text().splitlines() if line]
    parsed = [Path.from_row(r) for r in rows]
    flat_minute = FLATTEN.hour * 60 + FLATTEN.minute
    reduced, kept = [], []
    for p, row in zip(parsed, rows, strict=True):
        r = reduce_path(p.bars, p.buy_minute, p.buy_price, flat_minute, ORB_WIDTHS)
        if r is not None:
            reduced.append(r)
            kept.append((p, row))
    print(f"paths={len(rows)}  usable={len(reduced)}  dropped={len(rows) - len(reduced)}")
    parsed = [p for p, _ in kept]
    rows = [r for _, r in kept]
    days = np.array([r["day"] for r in rows])
    months = sorted({d[:7] for d in days.tolist()})
    stack = Stack(reduced)
    configs = grid()
    print(f"days={len(set(days.tolist()))} months={months} grid={len(configs)} configs")

    # -- 1. twin verification (rl_exits stage 1, verbatim) -------------------
    rng = random.Random(args.seed)
    sample = [TradingLimits()] + rng.sample(configs, min(args.verify_configs, len(configs)))
    idx = rng.sample(range(len(parsed)), min(args.verify_paths, len(parsed)))
    vpaths = [(parsed[i], reduced[i]) for i in idx]
    bad = 0
    for lim in sample:
        for p, r in vpaths:
            _, _, ref_minute = simulate(p, lim)
            _, _, fast_minute = first_exit(r, lim, ORB_WIDTHS)
            if ref_minute != fast_minute:
                bad += 1
                if bad <= 5:
                    print(
                        f"  MISMATCH {p.day} {p.ticker} {short(lim)}: {ref_minute} vs {fast_minute}"
                    )
    print(f"twin verification: {len(sample)} configs x {len(vpaths)} paths -> {bad} mismatches")
    if bad:
        raise SystemExit("twin does not match the live predicate; refusing to search")

    # -- scoring -------------------------------------------------------------
    live = TradingLimits()
    lab0 = stack.rewards(live)
    hold0 = stack.rewards(dataclasses.replace(live, vwap_exit_gap=None))
    print(f"scoring {len(configs)} fixed configs ...", flush=True)
    scores0 = np.empty((len(configs), stack.n))
    for i, c in enumerate(configs):
        scores0[i] = stack.rewards(c)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(configs)}", flush=True)

    sigma = np.array([entry_sigma(p) for p in parsed])
    n_nan = int(np.isnan(sigma).sum())
    median_sigma = float(np.nanmedian(sigma))
    sigma = np.where(np.isnan(sigma), median_sigma, sigma)
    print(
        f"entry sigma ({SIGMA_BARS} bars before the fill): median {median_sigma * 1e4:.0f} bp,"
        f" p10 {np.quantile(sigma, 0.1) * 1e4:.0f}, p90 {np.quantile(sigma, 0.9) * 1e4:.0f} bp,"
        f" nan->median on {n_nan} paths"
    )
    ctx_grid = [(a, b) for a in SIGMA_STOP for b in SIGMA_TP]
    ctx0 = {
        (a, b): stack.rewards(
            live,
            stop_pct=None if a is None else a * sigma,
            tp_pct=None if b is None else b * sigma,
        )
        for a, b in ctx_grid
    }

    cost = real_cost(rows, stock_markets())
    print(
        f"real round trip: mean {cost.mean() * 1e4:.1f} bp,"
        f" min {cost.min() * 1e4:.1f}, max {cost.max() * 1e4:.1f};"
        f" legacy literal {(float(SELL_COST) + float(BUY_COST)) * 1e4:.1f} bp"
    )

    result: dict = {"n_paths": len(rows), "n_days": len(set(days.tolist())), "modes": {}}

    for mode in ("legacy", "real"):
        fix = (lambda r: r) if mode == "legacy" else (lambda r: recost(r, cost))
        lab = fix(lab0)
        hold = fix(hold0)
        scores = {i: fix(scores0[i]) for i in range(len(configs))}
        ctx_scores = {k: fix(v) for k, v in ctx0.items()}

        print(f"\n################ cost = {mode}")
        print(
            f"pooled per-trade: live exits {lab.mean() * 1e4:+.1f} bp,"
            f" hold-to-close {hold.mean() * 1e4:+.1f} bp"
        )
        print(
            f"{'fold':9s} {'n':>5s} {'a':>5s} {'b':>5s} {'live bp':>9s} {'hold bp':>9s}"
            f" {'fixed bp':>9s} {'ctx bp':>9s}  chosen fixed config"
        )
        fold_rows = []
        live_days, fixed_days, ctx_days, hold_days = [], [], [], []
        for m in months[1:]:
            tr = np.array([d[:7] < m for d in days.tolist()])
            ev = np.array([d[:7] == m for d in days.tolist()])
            fbest = max(scores, key=lambda i: objective(days[tr], scores[i][tr]))
            cbest = max(ctx_scores, key=lambda k: objective(days[tr], ctx_scores[k][tr]))
            lim = configs[fbest]
            ev_fixed, ev_ctx, ev_live, ev_hold = (
                scores[fbest][ev],
                ctx_scores[cbest][ev],
                lab[ev],
                hold[ev],
            )
            live_days += day_records(days[ev], ev_live)
            hold_days += day_records(days[ev], ev_hold)
            fixed_days += day_records(days[ev], ev_fixed)
            ctx_days += day_records(days[ev], ev_ctx)
            fold_rows.append(
                {
                    "fold": m,
                    "n": int(ev.sum()),
                    "n_days": len(set(days[ev].tolist())),
                    "fixed_config": describe(lim),
                    "fixed_short": short(lim),
                    "a": cbest[0],
                    "b": cbest[1],
                    "live_bp": float(ev_live.mean() * 1e4),
                    "hold_bp": float(ev_hold.mean() * 1e4),
                    "fixed_bp": float(ev_fixed.mean() * 1e4),
                    "ctx_bp": float(ev_ctx.mean() * 1e4),
                }
            )
            print(
                f"{m:9s} {int(ev.sum()):>5d} {str(cbest[0]):>5s} {str(cbest[1]):>5s}"
                f" {ev_live.mean() * 1e4:>+9.1f} {ev_hold.mean() * 1e4:>+9.1f}"
                f" {ev_fixed.mean() * 1e4:>+9.1f} {ev_ctx.mean() * 1e4:>+9.1f}  {short(lim)}",
                flush=True,
            )

        key = lambda recs: {d["day"]: d for d in recs}  # noqa: E731
        series = {
            "live": key(live_days),
            "hold": key(hold_days),
            "fixed": key(fixed_days),
            "ctx": key(ctx_days),
        }
        oos = np.array([d[:7] in months[1:] for d in days.tolist()])
        pooled = {
            "live_bp": float(lab[oos].mean() * 1e4),
            "hold_bp": float(hold[oos].mean() * 1e4),
            "fixed_bp": float(
                sum(d["net"] for d in fixed_days) / sum(d["trades"] for d in fixed_days) / 10.0
            ),
            "ctx_bp": float(
                sum(d["net"] for d in ctx_days) / sum(d["trades"] for d in ctx_days) / 10.0
            ),
            "oos_trades": int(oos.sum()),
        }
        print(
            f"pooled OOS per-trade ({pooled['oos_trades']} trades):"
            f" live {pooled['live_bp']:+.1f} bp, hold {pooled['hold_bp']:+.1f} bp,"
            f" fixed {pooled['fixed_bp']:+.1f} bp, ctx {pooled['ctx_bp']:+.1f} bp"
        )

        verdicts = {}
        for label, cand, base in (
            ("ctx_vs_live", "ctx", "live"),
            ("ctx_vs_fixed", "ctx", "fixed"),
            ("fixed_vs_live", "fixed", "live"),
        ):
            v = judge(series[cand], series[base])
            verdicts[label] = v
            print(
                f"{label:14s} p(mu>0)={v['p_mu_gt_0']:.3f} mu={v['mu_mean']:+,.0f}"
                f" days>=0={v['days_ge_0']}/{v['n_days']}"
                f" worst {v['worst_day_candidate']:+,.0f} vs {v['worst_day_baseline']:+,.0f}"
                f" losers {v['loser_ratio_candidate']:.4f} vs {v['loser_ratio_baseline']:.4f}"
                f" adopt={v['adopt']}"
            )
            for r in v["reasons"]:
                print(f"    - {r}")

        comparisons = {}
        for label, cand, base in (("ctx_vs_fixed", "ctx", "fixed"), ("ctx_vs_live", "ctx", "live")):
            g = gate(series[cand], series[base])
            bs = bootstrap_gate(series[cand], series[base])
            comparisons[label] = {"gate": g, "bootstrap_pass_rate": bs}
            print(f"\n=== {mode} / {label}")
            print(
                f"{'penalty':10s} {'candidate':>14s} {'baseline':>14s}"
                f" {'margin':>12s} {'passes':>7s} {'boot':>6s}"
            )
            for name in PENALTIES:
                r = g[name]
                print(
                    f"{name:10s} {r['candidate']:>14,.1f} {r['baseline']:>14,.1f}"
                    f" {r['margin']:>+12,.1f} {str(r['passes']):>7s} {bs[name]:>6.3f}"
                )

        result["modes"][mode] = {
            "folds": fold_rows,
            "pooled": pooled,
            "verdicts": verdicts,
            "comparisons": comparisons,
            "daily": {k: sorted(v.values(), key=lambda d: d["day"]) for k, v in series.items()},
        }

    # parameter stability
    for mode in ("legacy", "real"):
        fr = result["modes"][mode]["folds"]
        a_counts: dict = {}
        b_counts: dict = {}
        for r in fr:
            a_counts[str(r["a"])] = a_counts.get(str(r["a"]), 0) + 1
            b_counts[str(r["b"])] = b_counts.get(str(r["b"]), 0) + 1
        distinct_fixed = len({r["fixed_short"] for r in fr})
        result["modes"][mode]["stability"] = {
            "a_counts": a_counts,
            "b_counts": b_counts,
            "distinct_fixed_configs": distinct_fixed,
            "n_folds": len(fr),
        }
        print(
            f"\nstability [{mode}]: a {a_counts}  b {b_counts}"
            f"  distinct fixed configs {distinct_fixed}/{len(fr)}"
        )

    result["twin_mismatches"] = bad
    result["real_cost_mean_bp"] = float(cost.mean() * 1e4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
