"""Search the sequential hold/add/cut policy, and judge it where it cannot cheat.

    .venv/bin/python scripts/eval/seqexit_search.py \
        --paths data/eval/rlnews/paths_prior.jsonl \
        --out data/eval/rlnews/seqexit.json                       # simnode

The population is `paths_prior.jsonl`: 149 judgment-free days, 2025-08-01 ..
2026-03-16, the same book the sigma-scaled contextual exit lost on, so the
three exit attempts are comparable on the same trades.

The split is NOT chosen here. `CONFIRM_WINDOWS.json` froze `track_a_prior` at
02:52 (commit f9376cf), before any exit searching had been dispatched: search
2025-08-01..2026-01-27 (119 days), confirm 2026-01-28..2026-03-16 (30 days).
This script reads that file and refuses to run if the two do not partition the
book. The confirm slice is scored ONCE, at the end, by `--confirm`.

Inside the search window there are two further time-ordered layers, and no
leave-one-day-out anywhere:

  * **monthly folds** -- fit on the months strictly before m, score on m, for m
    in 2025-09..2026-01. This is the out-of-sample number the search reports.
  * **an inner holdout** -- fit on 2025-08..2025-12, score on 2026-01. This is
    what the shuffled-feature null is measured against, so that establishing
    the noise floor does not cost a look at the real confirm slice.

Arms, all searched identically so the comparison is about the state and not
about the search:

    novol    pnl x slope                  15 states   the ablation
    vol      pnl x slope x vsurge         45 states   the hypothesis
    volplus  pnl x slope x vsurge x vtrend 135 states  the over-parameterised check

Baselines: the live exit rules, hold-to-close, the live pyramiding rule, and
the best fixed configuration from `rl_exits.grid()` chosen on the same folds by
the same objective. The fixed grid is the arm that beat both previous learned
exits, so it is the one to beat.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path as FsPath

import numpy as np

sys.path.insert(0, "scripts/eval")

from bayes_judge import judge  # noqa: E402
from exit_lab import FLATTEN, Path  # noqa: E402
from exit_worst_day import real_cost, recost, stock_markets  # noqa: E402
from rl_exits import ORB_WIDTHS, Stack, describe, grid, short  # noqa: E402
from seqexit_core import (  # noqa: E402
    ADD_BUCKETS,
    CUT_BUCKETS,
    LATTICES,
    NOTIONAL,
    PNL_EDGES,
    SLOPE_EDGES,
    VSURGE_EDGES,
    VTREND_EDGES,
    Book,
    Sim,
    build,
    day_records,
    objective,
    scramble,
)

from daytrade_it.application.trading.auto_trader import TradingLimits  # noqa: E402

# ------------------------------------------------------------------ the search


def ascend(
    sim: Sim,
    mask: np.ndarray,
    rng: np.random.Generator,
    restarts: int,
    max_sweeps: int = 12,
) -> tuple[np.ndarray, float, int]:
    """Coordinate ascent on the binary action table. (policy, train objective, evals).

    The policy is one bit per state -- below the entry that bit means "cut
    here", at or above it "add here" -- so the search space is a hypercube and
    every move is a single flip, evaluated EXACTLY by re-rolling every path
    rather than through a fitted value function. There is no bootstrapping and
    no discount factor to misspecify; the number being maximised is the number
    that will be reported.

    `mask` is the training population; the roll covers the whole book and only
    the objective is restricted, so the fold, its scrambled channels and its
    per-path costs cannot drift out of alignment with one another.

    Restart 0 starts from all-hold (hold to the close, the honest null). The
    rest start random, so a basin that only all-hold can reach is not mistaken
    for the global optimum.
    """
    states = sim.actionable
    best_act, best_val, evals = None, -np.inf, 0
    for r in range(restarts):
        act = np.zeros(sim.lat.n_states, dtype=np.int8)
        if r:
            act[states] = (rng.random(states.size) < 0.3).astype(np.int8)
        val = objective(sim.run(act)[0], mask)
        evals += 1
        for _ in range(max_sweeps):
            improved = False
            for s in rng.permutation(states):
                act[s] ^= 1
                trial = objective(sim.run(act)[0], mask)
                evals += 1
                if trial > val:
                    val, improved = trial, True
                else:
                    act[s] ^= 1
            if not improved:
                break
        if val > best_val:
            best_act, best_val = act.copy(), val
    return best_act, best_val, evals


def evaluate(sim_full: Sim, act: np.ndarray, keep: np.ndarray) -> dict:
    """Roll the whole book under `act`, then report only the paths in `keep`."""
    net, exit_at, add_at = sim_full.run(act)
    book = sim_full.book
    cut = (exit_at < book.last_idx)[keep]
    return {
        "net": net[keep],
        "days": book.day[keep],
        "cut_rate": float(cut.mean()),
        "add_rate": float((add_at >= 0)[keep].mean()),
        "mean_bp": float(net[keep].mean() / NOTIONAL * 1e4),
    }


# ------------------------------------------------------------------ baselines


def fixed_grid_scores(book: Book, rows: list[dict], cost: np.ndarray, cache: FsPath) -> np.ndarray:
    """(n_configs, n_paths) net KRW for every configuration in `rl_exits.grid()`.

    Scored through `rl_exits.Stack` -- the verified twin -- not re-implemented
    here, and re-priced from the legacy literal to the broker's own round trip
    with `exit_worst_day.recost`. ~4,300 configurations over 29,800 paths is
    twenty minutes, so it is cached: the fixed grid is a fixed baseline and
    nothing in the policy search changes it.
    """
    if cache.exists():
        out = np.load(cache)
        print(f"fixed grid: {out.shape} from cache {cache}")
        return out
    flat = FLATTEN.hour * 60 + FLATTEN.minute
    from exit_fast import reduce_path  # noqa: PLC0415

    reduced = []
    for row in rows:
        p = Path.from_row(row)
        reduced.append(reduce_path(p.bars, p.buy_minute, p.buy_price, flat, ORB_WIDTHS))
    stack = Stack(reduced)
    configs = grid()
    out = np.empty((len(configs), book.n_paths), dtype=np.float32)
    t0 = time.time()
    for i, c in enumerate(configs):
        out[i] = NOTIONAL * recost(stack.rewards(c), cost)
        if (i + 1) % 400 == 0:
            print(f"  fixed grid {i + 1}/{len(configs)}  {time.time() - t0:.0f}s", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def live_add_policy(sim: Sim) -> np.ndarray:
    """The live pyramiding rule as a table: add wherever `add_on_hit` would.

    `add_on_hit` fires at +1% or better, which is the whole of pnl bucket 4
    (>= +1.5%) and part of bucket 3; the table can only express the bucket, so
    this arm adds across both legal buckets and is a slight over-statement of
    what live does. It is a reference point, not a claim about live.
    """
    act = np.zeros(sim.lat.n_states, dtype=np.int8)
    act[sim.add_states] = 1
    return act


# ------------------------------------------------------------------ reporting


def policy_table(sim: Sim, act: np.ndarray, lat) -> list[dict]:
    steps, paths = sim.visits()
    rows = []
    for s in range(lat.n_states):
        if not steps[s]:
            continue
        d = lat.decode(s)
        legal = "cut" if d["pnl"] in CUT_BUCKETS else "add"
        rows.append(
            {
                **d,
                "legal_action": legal,
                "action": legal if act[s] else "hold",
                "visits_steps": int(steps[s]),
                "visits_paths": int(paths[s]),
            }
        )
    return rows


def describe_lattice() -> dict:
    return {
        "pnl_edges": list(PNL_EDGES),
        "slope_edges": list(SLOPE_EDGES),
        "vsurge_edges": list(VSURGE_EDGES),
        "vtrend_edges": list(VTREND_EDGES),
        "cut_buckets": list(CUT_BUCKETS),
        "add_buckets": list(ADD_BUCKETS),
    }


def from_days(series: dict[str, dict]) -> dict:
    """Pooled figures from `bayes_judge`-shaped day records.

    Worst day and loser ratio appear here and in `summarise` as DIAGNOSTICS.
    They are what `bayes_judge` gates on, which is exactly why nothing upstream
    of them is allowed to select on them.
    """
    nets = np.array([v["net"] for v in series.values()])
    trades = sum(v["trades"] for v in series.values())
    return {
        "trades": int(trades),
        "n_days": len(series),
        "mean_bp": float(nets.sum() / trades / NOTIONAL * 1e4),
        "total_krw": float(nets.sum()),
        "worst_day_krw": float(nets.min()),
        "q05_day_krw": float(np.quantile(nets, 0.05)),
        "loser_ratio": float(sum(v["losers"] for v in series.values()) / trades),
        "days_ge_0": int((nets >= 0).sum()),
    }


def summarise(net: np.ndarray, days: np.ndarray) -> dict:
    recs = day_records(days, net)
    nets = np.array([r["net"] for r in recs])
    return {
        "trades": int(net.size),
        "n_days": len(recs),
        "mean_bp": float(net.mean() / NOTIONAL * 1e4),
        "total_krw": float(net.sum()),
        # DIAGNOSTICS ONLY -- nothing below this line ever selected a policy.
        "worst_day_krw": float(nets.min()),
        "q05_day_krw": float(np.quantile(nets, 0.05)),
        "loser_ratio": float((net < 0).mean()),
        "days_ge_0": int((nets >= 0).sum()),
    }


# ------------------------------------------------------------------ main


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=FsPath, required=True)
    ap.add_argument(
        "--windows", type=FsPath, default=FsPath("data/eval/rlnews/CONFIRM_WINDOWS.json")
    )
    ap.add_argument("--out", type=FsPath, required=True)
    ap.add_argument("--cache", type=FsPath, default=FsPath("data/eval/rlnews/seqexit_fixed.npy"))
    ap.add_argument("--arms", default="novol,vol,volplus")
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--shuffles", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (smoke runs)")
    ap.add_argument(
        "--confirm",
        action="store_true",
        help="score the frozen confirm slice. Left off until the search is finished.",
    )
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.paths.read_text().splitlines() if line]
    if args.limit:
        rows = rows[: args.limit]
    book, keep_idx = build(rows)
    keep_rows = [rows[i] for i in keep_idx]
    print(f"paths={len(rows)} usable={book.n_paths} steps={book.n_steps:,}")

    cost = real_cost(keep_rows, stock_markets())
    print(
        f"round trip: mean {cost.mean() * 1e4:.1f} bp, min {cost.min() * 1e4:.1f},"
        f" max {cost.max() * 1e4:.1f}"
    )

    # -- the split, read rather than chosen ---------------------------------
    win = json.loads(args.windows.read_text())["track_a_prior"]
    confirm_days = set(win["confirm"])
    lo, hi = win["search"]
    all_days = sorted(set(book.day.tolist()))
    search_days = {d for d in all_days if lo <= d <= hi}
    if search_days & confirm_days:
        raise SystemExit("frozen windows overlap")
    stray = set(all_days) - search_days - confirm_days
    if stray:
        raise SystemExit(f"days in neither window: {sorted(stray)[:5]}")
    is_search = np.isin(book.day, sorted(search_days))
    is_confirm = np.isin(book.day, sorted(confirm_days))
    print(
        f"search {len(search_days)} days / {int(is_search.sum())} trades"
        f"   confirm {len(confirm_days)} days / {int(is_confirm.sum())} trades"
        f"   [frozen {json.loads(args.windows.read_text())['frozen_at']}]"
    )

    months = sorted({d[:7] for d in sorted(search_days)})
    fold_months = months[1:]
    inner_test_month = months[-1]
    print(f"months {months}; folds {fold_months}; inner holdout {inner_test_month}")

    # -- baselines ----------------------------------------------------------
    fixed = fixed_grid_scores(book, keep_rows, cost, args.cache)
    configs = grid()
    live_idx = next(i for i, c in enumerate(configs) if c == TradingLimits())
    hold_idx = next(
        i
        for i, c in enumerate(configs)
        if c == dataclasses.replace(TradingLimits(), vwap_exit_gap=None)
    )
    live_net, hold_net = fixed[live_idx].astype(float), fixed[hold_idx].astype(float)

    result: dict = {
        "population": {
            "paths": len(rows),
            "usable": book.n_paths,
            "steps": int(book.n_steps),
            "search_days": len(search_days),
            "confirm_days": len(confirm_days),
            "frozen_at": json.loads(args.windows.read_text())["frozen_at"],
            "cost_mean_bp": float(cost.mean() * 1e4),
        },
        "lattice": describe_lattice(),
        "selection_statistic": "mean realised KRW per trade (seqexit_core.objective)",
        "arms": {},
    }

    for arm in args.arms.split(","):
        lat = LATTICES[arm]
        t0 = time.time()
        sim_full = Sim(book, lat, cost)
        dof = int(sim_full.actionable.size)
        print(f"\n################ arm={arm}  states={lat.n_states}  actionable={dof}")
        print(
            f"  effective dof {dof} binary params vs {int(is_search.sum())} training trades"
            f" = {int(is_search.sum()) / max(dof, 1):.0f} trades/param"
        )

        # ---- monthly time-ordered folds inside the search window ----------
        fold_rows, cand_net, cand_days_arr, fixed_net_oos = [], [], [], []
        for m in fold_months:
            tr = is_search & np.array([d[:7] < m for d in book.day.tolist()])
            ev = is_search & np.array([d[:7] == m for d in book.day.tolist()])
            act, tr_val, n_ev = ascend(
                sim_full, tr, np.random.default_rng(args.seed), args.restarts
            )
            e = evaluate(sim_full, act, ev)
            best_fixed = int(np.argmax(fixed[:, tr].mean(axis=1)))
            fold_rows.append(
                {
                    "fold": m,
                    "n": int(ev.sum()),
                    "train_bp": float(tr_val / NOTIONAL * 1e4),
                    "seq_bp": e["mean_bp"],
                    "live_bp": float(live_net[ev].mean() / NOTIONAL * 1e4),
                    "hold_bp": float(hold_net[ev].mean() / NOTIONAL * 1e4),
                    "fixed_bp": float(fixed[best_fixed, ev].mean() / NOTIONAL * 1e4),
                    "fixed_config": short(configs[best_fixed]),
                    "cut_rate": e["cut_rate"],
                    "add_rate": e["add_rate"],
                    "n_cut_states": int(act[sim_full.cut_states].sum()),
                    "n_add_states": int(act[sim_full.add_states].sum()),
                    "evals": n_ev,
                }
            )
            cand_net.append(e["net"])
            cand_days_arr.append(e["days"])
            fixed_net_oos.append(fixed[best_fixed, ev].astype(float))
            r = fold_rows[-1]
            print(
                f"  {m}  n={r['n']:>5d}  seq {r['seq_bp']:+7.1f}  fixed {r['fixed_bp']:+7.1f}"
                f"  live {r['live_bp']:+7.1f}  hold {r['hold_bp']:+7.1f}"
                f"  cut {r['cut_rate']:.3f} add {r['add_rate']:.3f}  ({r['evals']} evals)",
                flush=True,
            )

        oos = is_search & np.array([d[:7] in fold_months for d in book.day.tolist()])
        seq_oos = np.concatenate(cand_net)
        seq_oos_days = np.concatenate(cand_days_arr)
        fixed_oos = np.concatenate(fixed_net_oos)

        search_block = {
            "folds": fold_rows,
            "pooled": {
                "seq": summarise(seq_oos, seq_oos_days),
                "fixed": summarise(fixed_oos, book.day[oos]),
                "live": summarise(live_net[oos], book.day[oos]),
                "hold": summarise(hold_net[oos], book.day[oos]),
            },
        }
        print(
            f"  pooled search-OOS ({int(oos.sum())} trades):"
            f" seq {search_block['pooled']['seq']['mean_bp']:+.1f} bp,"
            f" fixed {search_block['pooled']['fixed']['mean_bp']:+.1f},"
            f" live {search_block['pooled']['live']['mean_bp']:+.1f},"
            f" hold {search_block['pooled']['hold']['mean_bp']:+.1f}"
        )

        # ---- shuffled-feature null, on the inner holdout ------------------
        inner_tr = is_search & np.array([d[:7] < inner_test_month for d in book.day.tolist()])
        inner_ev = is_search & np.array([d[:7] == inner_test_month for d in book.day.tolist()])
        act_real, real_train, _ = ascend(
            sim_full, inner_tr, np.random.default_rng(args.seed), args.restarts
        )
        real_inner = evaluate(sim_full, act_real, inner_ev)["mean_bp"]
        print(
            f"  inner holdout {inner_test_month}: real train {real_train / NOTIONAL * 1e4:+.1f} bp,"
            f" test {real_inner:+.1f} bp;  {args.shuffles} scrambles ...",
            flush=True,
        )

        sh_train, sh_test = [], []
        for rep in range(args.shuffles):
            # ONE scramble of the whole book per repeat, so the policy is fit
            # and tested against the same scrambled world -- the coherent null.
            # Scrambling train and test separately would test a policy on
            # features it was never fit against, which is a harder task than
            # the real arm faces and would understate the noise floor.
            sc = scramble(book, np.random.default_rng(args.seed + 1000 + rep))
            s = Sim(book, lat, cost, scramble=sc)
            a, v, _ = ascend(s, inner_tr, np.random.default_rng(args.seed), args.restarts)
            sh_train.append(v / NOTIONAL * 1e4)
            sh_test.append(float(s.run(a)[0][inner_ev].mean() / NOTIONAL * 1e4))
            if (rep + 1) % 5 == 0:
                print(
                    f"    scramble {rep + 1}/{args.shuffles}  {time.time() - t0:.0f}s", flush=True
                )
        shuffle_block = {
            "n": args.shuffles,
            "real_train_bp": float(real_train / NOTIONAL * 1e4),
            "real_test_bp": real_inner,
            "shuffled_train": {
                "mean": float(np.mean(sh_train)),
                "p90": float(np.quantile(sh_train, 0.9)),
                "max": float(np.max(sh_train)),
            },
            "shuffled_test": {
                "mean": float(np.mean(sh_test)),
                "p90": float(np.quantile(sh_test, 0.9)),
                "max": float(np.max(sh_test)),
            },
            "real_beats_shuffled_max_train": bool(real_train / NOTIONAL * 1e4 > np.max(sh_train)),
            "real_beats_shuffled_p90_test": bool(real_inner > np.quantile(sh_test, 0.9)),
        }
        print(
            f"  shuffle premium: real train {shuffle_block['real_train_bp']:+.1f} vs"
            f" noise mean/p90/max {shuffle_block['shuffled_train']['mean']:+.1f}/"
            f"{shuffle_block['shuffled_train']['p90']:+.1f}/"
            f"{shuffle_block['shuffled_train']['max']:+.1f};"
            f"  real test {real_inner:+.1f} vs"
            f" {shuffle_block['shuffled_test']['mean']:+.1f}/"
            f"{shuffle_block['shuffled_test']['p90']:+.1f}/"
            f"{shuffle_block['shuffled_test']['max']:+.1f}"
        )

        arm_block = {
            "n_states": lat.n_states,
            "actionable_states": dof,
            "trades_per_param": float(int(is_search.sum()) / max(dof, 1)),
            "search": search_block,
            "shuffle": shuffle_block,
            "seconds": time.time() - t0,
        }

        # ---- the frozen confirm slice, fit once on the whole search window --
        if args.confirm:
            act_final, final_train, _ = ascend(
                sim_full, is_search, np.random.default_rng(args.seed), args.restarts
            )
            e = evaluate(sim_full, act_final, is_confirm)
            best_fixed = int(np.argmax(fixed[:, is_search].mean(axis=1)))
            cdays = book.day[is_confirm]
            series = {
                "seq": {d["day"]: d for d in day_records(cdays, e["net"])},
                "fixed": {
                    d["day"]: d
                    for d in day_records(cdays, fixed[best_fixed, is_confirm].astype(float))
                },
                "live": {d["day"]: d for d in day_records(cdays, live_net[is_confirm])},
                "hold": {d["day"]: d for d in day_records(cdays, hold_net[is_confirm])},
                "liveadd": {
                    d["day"]: d
                    for d in day_records(
                        cdays, evaluate(sim_full, live_add_policy(sim_full), is_confirm)["net"]
                    )
                },
            }
            verdicts = {
                f"{c}_vs_{b}": judge(series[c], series[b])
                for c, b in (
                    ("seq", "live"),
                    ("seq", "hold"),
                    ("seq", "fixed"),
                    ("fixed", "live"),
                )
            }
            arm_block["confirm"] = {
                "train_bp": float(final_train / NOTIONAL * 1e4),
                "policy": policy_table(sim_full, act_final, lat),
                "n_cut_states": int(act_final[sim_full.cut_states].sum()),
                "n_add_states": int(act_final[sim_full.add_states].sum()),
                "cut_rate": e["cut_rate"],
                "add_rate": e["add_rate"],
                "fixed_config": short(configs[best_fixed]),
                "fixed_describe": describe(configs[best_fixed]),
                "summary": {k: from_days(v) for k, v in series.items()},
                "verdicts": verdicts,
                "daily": {k: sorted(v.values(), key=lambda d: d["day"]) for k, v in series.items()},
            }
            print(f"\n  === CONFIRM {len(confirm_days)} days, touched once")
            for k, s in arm_block["confirm"]["summary"].items():
                print(
                    f"    {k:9s} {s['mean_bp']:+8.1f} bp  total {s['total_krw']:+12,.0f}"
                    f"  worst day {s['worst_day_krw']:+10,.0f}  losers {s['loser_ratio']:.3f}"
                )
            for k, v in verdicts.items():
                print(
                    f"    {k:14s} p(mu>0)={v['p_mu_gt_0']:.3f} mu={v['mu_mean']:+,.0f}"
                    f" days>=0={v['days_ge_0']}/{v['n_days']} adopt={v['adopt']}"
                )
                for r in v["reasons"]:
                    print(f"        - {r}")

        result["arms"][arm] = arm_block
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        print(f"  -> {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
