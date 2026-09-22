"""Put the three exit attempts on one table, and price the selection defect.

Two jobs, both about making the new result comparable rather than merely new.

**1. The same protocol as the published exit holdout.** `rlnews_exit_holdout.py`
scored the fixed grid and the sigma-scaled contextual exit on monthly
time-ordered folds over all 149 days and reported pooled per-trade net. That is
reproduced here at the broker's own round trip so the sequential policy can be
read against -35.4 (fixed), -40.5 (sigma-scaled) and -42.2 (live) rather than
against numbers computed some other way.

**2. What the worst-day term in the selection statistic is worth.**
`rl_exits.objective` is ``sum(day_nets) - 0.5 * max(0, -min(day_nets))`` and
`bayes_judge` gate 2 is ``min(net) over days``: the policy is chosen partly for
clearing the gate it is about to face. The composition was found and fixed once
tonight in `optimize_policy.py`; it is still here. Rather than assert that it
matters, both selections are run over the identical fold structure and
configuration grid -- once under `rl_exits.objective`, once under plain mean
realised KRW -- and the difference is reported. If the two pick the same
configurations the defect is inert on this book, which is worth knowing as
plainly as the reverse.

    .venv/bin/python scripts/eval/seqexit_compare.py \
        --paths data/eval/rlnews/paths_prior.jsonl \
        --out data/eval/rlnews/seqexit_compare.json                # simnode
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path as FsPath

import numpy as np

sys.path.insert(0, "scripts/eval")

from exit_fast import reduce_path  # noqa: E402
from exit_lab import FLATTEN, Path  # noqa: E402
from exit_worst_day import real_cost, recost, stock_markets  # noqa: E402
from rl_exits import (  # noqa: E402
    ORB_WIDTHS,
    SIGMA_STOP,
    SIGMA_TP,
    Stack,
    entry_sigma,
    grid,
    short,
)
from rl_exits import objective as worstday_objective  # noqa: E402
from seqexit_core import NOTIONAL, build  # noqa: E402

from daytrade_it.application.trading.auto_trader import TradingLimits  # noqa: E402


def mean_objective(_days: np.ndarray, net: np.ndarray) -> float:
    """Mean realised KRW per trade -- no day aggregation, no tail term."""
    return float(net.mean())


def worstday(days: np.ndarray, net: np.ndarray) -> float:
    """`rl_exits.objective`, which the published runs selected on.

    Homogeneous of degree one, so feeding it KRW rather than fractional returns
    scales it without changing which configuration wins.
    """
    return worstday_objective(days, net)


SELECTORS = {"mean_krw": mean_objective, "worst_day_penalised": worstday}


def pooled(net: np.ndarray, days: np.ndarray) -> dict:
    per_day = defaultdict(float)
    for d, r in zip(days.tolist(), net.tolist(), strict=True):
        per_day[d] += r
    nets = np.array(list(per_day.values()))
    return {
        "trades": int(net.size),
        "n_days": int(nets.size),
        "mean_bp": float(net.mean() / NOTIONAL * 1e4),
        "total_krw": float(net.sum()),
        "worst_day_krw": float(nets.min()),
        "loser_ratio": float((net < 0).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=FsPath, required=True)
    ap.add_argument("--out", type=FsPath, required=True)
    ap.add_argument(
        "--fixed-cache", type=FsPath, default=FsPath("data/eval/rlnews/seqexit_fixed.npy")
    )
    ap.add_argument("--ctx-cache", type=FsPath, default=FsPath("data/eval/rlnews/seqexit_ctx.npy"))
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.paths.read_text().splitlines() if line]
    book, keep_idx = build(rows)
    keep_rows = [rows[i] for i in keep_idx]
    cost = real_cost(keep_rows, stock_markets())
    days = book.day
    months = sorted({d[:7] for d in days.tolist()})
    print(f"usable {book.n_paths} paths, {len(set(days.tolist()))} days, months {months}")

    flat = FLATTEN.hour * 60 + FLATTEN.minute
    parsed = [Path.from_row(r) for r in keep_rows]
    reduced = [reduce_path(p.bars, p.buy_minute, p.buy_price, flat, ORB_WIDTHS) for p in parsed]
    stack = Stack(reduced)
    configs = grid()

    if args.fixed_cache.exists():
        fixed = np.load(args.fixed_cache)
    else:
        fixed = np.empty((len(configs), book.n_paths), dtype=np.float32)
        for i, c in enumerate(configs):
            fixed[i] = NOTIONAL * recost(stack.rewards(c), cost)
        np.save(args.fixed_cache, fixed)
    print(f"fixed grid {fixed.shape}")

    # -- the sigma-scaled contextual family, the second failed attempt -------
    sigma = np.array([entry_sigma(p) for p in parsed])
    sigma = np.where(np.isnan(sigma), float(np.nanmedian(sigma)), sigma)
    ctx_keys = [(a, b) for a in SIGMA_STOP for b in SIGMA_TP]
    if args.ctx_cache.exists():
        ctx = np.load(args.ctx_cache)
    else:
        live = TradingLimits()
        ctx = np.empty((len(ctx_keys), book.n_paths), dtype=np.float32)
        t0 = time.time()
        for i, (a, b) in enumerate(ctx_keys):
            ctx[i] = NOTIONAL * recost(
                stack.rewards(
                    live,
                    stop_pct=None if a is None else a * sigma,
                    tp_pct=None if b is None else b * sigma,
                ),
                cost,
            )
        np.save(args.ctx_cache, ctx)
        print(f"ctx grid {ctx.shape} in {time.time() - t0:.0f}s")
    print(f"ctx grid {ctx.shape}")

    live_idx = next(i for i, c in enumerate(configs) if c == TradingLimits())
    import dataclasses  # noqa: PLC0415

    hold_idx = next(
        i
        for i, c in enumerate(configs)
        if c == dataclasses.replace(TradingLimits(), vwap_exit_gap=None)
    )

    result: dict = {"months": months, "selectors": {}}
    for sel_name, sel in SELECTORS.items():
        picks_fixed, picks_ctx = [], []
        net_fixed, net_ctx, oos_mask = [], [], []
        for m in months[1:]:
            tr = np.array([d[:7] < m for d in days.tolist()])
            ev = np.array([d[:7] == m for d in days.tolist()])
            bf = max(range(len(configs)), key=lambda i: sel(days[tr], fixed[i][tr]))
            bc = max(range(len(ctx_keys)), key=lambda i: sel(days[tr], ctx[i][tr]))
            picks_fixed.append(short(configs[bf]))
            picks_ctx.append(f"a={ctx_keys[bc][0]} b={ctx_keys[bc][1]}")
            net_fixed.append(fixed[bf][ev].astype(float))
            net_ctx.append(ctx[bc][ev].astype(float))
            oos_mask.append(ev)
        oos = np.any(oos_mask, axis=0)
        d_oos = days[oos]
        block = {
            "fixed_picks": picks_fixed,
            "ctx_picks": picks_ctx,
            "distinct_fixed": len(set(picks_fixed)),
            "distinct_ctx": len(set(picks_ctx)),
            "pooled": {
                "fixed": pooled(np.concatenate(net_fixed), d_oos),
                "ctx": pooled(np.concatenate(net_ctx), d_oos),
                "live": pooled(fixed[live_idx][oos].astype(float), d_oos),
                "hold": pooled(fixed[hold_idx][oos].astype(float), d_oos),
            },
        }
        result["selectors"][sel_name] = block
        print(f"\n=== selector = {sel_name}")
        for k, v in block["pooled"].items():
            print(
                f"  {k:6s} {v['mean_bp']:+8.1f} bp over {v['trades']} trades,"
                f" {v['n_days']} days, worst day {v['worst_day_krw']:+,.0f}"
            )
        print(
            f"  distinct configs chosen: fixed {block['distinct_fixed']}/{len(picks_fixed)},"
            f" ctx {block['distinct_ctx']}/{len(picks_ctx)}"
        )
        print(f"  fixed picks: {picks_fixed}")
        print(f"  ctx picks:   {picks_ctx}")

    a = result["selectors"]["mean_krw"]["pooled"]
    b = result["selectors"]["worst_day_penalised"]["pooled"]
    result["selection_defect_bp"] = {
        k: float(b[k]["mean_bp"] - a[k]["mean_bp"]) for k in ("fixed", "ctx")
    }
    print(
        "\nselection defect (worst-day-penalised minus clean mean), per trade:"
        f" fixed {result['selection_defect_bp']['fixed']:+.1f} bp,"
        f" ctx {result['selection_defect_bp']['ctx']:+.1f} bp"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
