"""Would a gate chosen from the past have worked on the next month?

The univariate scan in `edge_cond.py` is in-sample by construction: with 22
axes and a few hundred levels, some level is positive at any sample size. This
runs the only test that answers the question honestly.

For each month m from the second onward:
  * fit  -- every (axis, level) with at least `--min-pairs` pairs in the months
            strictly before m is ranked by its training alpha
  * pick -- the best one, and separately the best `--k` combined as an OR
  * eval -- its alpha and net return on month m alone, which the fit never saw

No LODO: month m's training set is a strict prefix, so no fold shares a day
with its own evaluation and no fold sees the future. Reported per fold and
pooled, with the day-clustered spread, the loser ratio and the worst day --
because the objective is asymmetric in loss size, not in expectancy.

`--frozen` then applies the gate fitted on ALL sweep months to a pool it has
never touched (the dashboard days), which is the only transfer test that has
ever held up in this repo.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from edge_cond import day_mean
from edge_feat import AXES

BP = 10_000


def gate_rows(rows: list[dict], gate: tuple[str, str]) -> list[dict]:
    axis, level = gate
    return [r for r in rows if str(r["f"][axis]) == level]


def rank_gates(train: list[dict], min_pairs: int, min_days: int) -> list[tuple]:
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in train:
        for axis in AXES:
            by[(axis, str(r["f"][axis]))].append(r)
    out = []
    for gate, rs in by.items():
        if len(rs) < min_pairs or len({r["day"] for r in rs}) < min_days:
            continue
        mu, _, _, _ = day_mean([(r["day"], r["alpha"]) for r in rs])
        out.append((mu, gate, len(rs)))
    return sorted(out, reverse=True)


def report(name: str, rs: list[dict]) -> str:
    if not rs:
        return f"{name:<30} (no pairs)"
    a_mu, a_se, nd, a_ge = day_mean([(r["day"], r["alpha"]) for r in rs])
    n_mu, _, _, n_ge = day_mean([(r["day"], r["net"]) for r in rs])
    per: dict[str, list[float]] = defaultdict(list)
    for r in rs:
        per[r["day"]].append(r["net"])
    worst = min(statistics.fmean(v) for v in per.values())
    losers = sum(1 for r in rs if r["net"] < 0) / len(rs)
    return (
        f"{name:<30} n={len(rs):>6} d={nd:>3} alpha={a_mu * BP:>+7.1f}(se{a_se * BP:>5.1f}) "
        f"net={n_mu * BP:>+7.1f} d>=0={n_ge:>3}/{nd:<3} worst={worst * BP:>+8.1f} "
        f"losers={losers:.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="jsonl from edge_cond.py --dump")
    ap.add_argument("--frozen", help="a second --dump to apply the all-months gate to")
    ap.add_argument("--min-pairs", type=int, default=500)
    ap.add_argument("--min-days", type=int, default=20)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines()]
    months = sorted({r["month"] for r in rows})
    print(f"\n{len(rows)} pairs over {months[0]}..{months[-1]}\n")
    print(report("BASE all pairs", rows))
    print()

    picked_all: list[dict] = []
    picked_k: list[dict] = []
    for i, m in enumerate(months):
        if i == 0:
            continue
        train = [r for r in rows if r["month"] < m]
        test = [r for r in rows if r["month"] == m]
        ranked = rank_gates(train, args.min_pairs, args.min_days)
        if not ranked:
            continue
        mu, gate, n_tr = ranked[0]
        sel = gate_rows(test, gate)
        picked_all += sel
        top = [g for _mu, g, _n in ranked[: args.k]]
        keys = {id(r) for g in top for r in gate_rows(test, g)}
        sel_k = [r for r in test if id(r) in keys]
        picked_k += sel_k
        base, _, _, _ = day_mean([(r["day"], r["alpha"]) for r in test])
        got, _, _, _ = day_mean([(r["day"], r["alpha"]) for r in sel]) if sel else (0, 0, 0, 0)
        print(
            f"{m}  fit[{gate[0]}={gate[1]}] train_alpha={mu * BP:+7.1f} (n={n_tr}) "
            f"-> test_alpha={got * BP:+7.1f} vs month base {base * BP:+7.1f}  (n={len(sel)})"
        )

    print()
    print(report("OOS best-1 gate", picked_all))
    print(report(f"OOS best-{args.k} union", picked_k))

    if args.frozen:
        frozen = [json.loads(line) for line in Path(args.frozen).read_text().splitlines()]
        ranked = rank_gates(rows, args.min_pairs, args.min_days)
        print(f"\n--- frozen on {args.frozen} ({len(frozen)} pairs)")
        print(report("BASE frozen pool", frozen))
        for mu, gate, _n_tr in ranked[: args.k]:
            sel = gate_rows(frozen, gate)
            print(report(f"{gate[0]}={gate[1]} (fit{mu * BP:+.0f})", sel))


if __name__ == "__main__":
    main()
