"""The twin-verification gate, run hard enough that sampling luck cannot pass it.

`rl_exits.main` stage 1 samples 12 configurations x 400 paths. That is what let
the Decimal-tie defect (`exit_fast.barrier`) survive: on the 2026-04..08 book the
6 affected paths are indices [2228, 2576, 2966, 3624, 3930, 3982] and seed 0's
400-path draw contains none of them, so the gate passed on a twin that was not
one.

This runs the same assertion -- `exit_lab.simulate` (the live `exit_rule_hit`)
against `exit_fast.first_exit` -- over:

  * EVERY path carrying an exact Decimal tie on any grid threshold, found by
    construction rather than by sampling, and
  * a large random draw on top,

crossed with a configuration set that deliberately includes the far-target arms
(+8%/+10% take-profit, +8% trail) where the tie lives.

    .venv/bin/python scripts/eval/verify_twin.py --paths <book.jsonl>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from decimal import Decimal
from multiprocessing import Pool
from pathlib import Path as FsPath

import numpy as np

sys.path.insert(0, "scripts/eval")

from exit_fast import first_exit, reduce_path  # noqa: E402
from exit_lab import FLATTEN, Path, simulate  # noqa: E402
from rl_exits import ORB_WIDTHS, STOP_LOSS, TAKE_PROFIT, TRAIL, grid, short  # noqa: E402

from daytrade_it.application.trading.auto_trader import TradingLimits  # noqa: E402

_PATHS: list = []
_REDUCED: list = []
_CONFIGS: list = []


def tie_affected(rows: list[dict]) -> list[int]:
    """Indices of paths where some bar sits EXACTLY on a Decimal grid barrier that
    the float barrier would have placed on the other side. Found by construction."""
    levels = [(p, +1) for p in TAKE_PROFIT if p is not None]
    levels += [(p, -1) for p in STOP_LOSS if p is not None]
    levels += [(a, +1) for a, _ in TRAIL if a is not None]
    levels += [(g, -1) for _, g in TRAIL]
    out = []
    for i, r in enumerate(rows):
        e = float(r["buy_price"])
        ed = Decimal(str(e))
        bm = r["buy_minute"]
        arr = np.array([b for b in r["bars"] if b[0] > bm], dtype=float)
        if arr.size == 0:
            continue
        series = np.concatenate([arr[:, 4], arr[:, 2]])  # closes and highs
        for p, sign in levels:
            exact = float(ed * (Decimal(1) + (1 if sign > 0 else -1) * Decimal(str(p))))
            approx = e * (1 + float(p)) if sign > 0 else e * (1 - float(p))
            if exact != approx and (series == exact).any():
                out.append(i)
                break
    return out


def _check(i: int) -> list[str]:
    bad = []
    p, r = _PATHS[i], _REDUCED[i]
    for lim in _CONFIGS:
        _, _, ref = simulate(p, lim)
        _, _, fast = first_exit(r, lim, ORB_WIDTHS)
        if ref != fast:
            bad.append(f"{p.day} {p.ticker} {short(lim)}: live {ref} vs twin {fast}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=FsPath, required=True)
    ap.add_argument("--random-paths", type=int, default=1500)
    ap.add_argument("--random-configs", type=int, default=24)
    ap.add_argument("--far-configs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=14)
    args = ap.parse_args()

    # Rows are read and the verification pool chosen BEFORE any `Path` is built:
    # a full book of `MinuteBar` objects is ~10 GB, and forking workers over that
    # is what turns a verification into an OOM.
    rows = [json.loads(line) for line in args.paths.read_text().splitlines() if line]
    usable = [i for i, r in enumerate(rows) if any(b[0] > r["buy_minute"] for b in r["bars"])]
    print(f"{args.paths}: rows={len(rows)} usable={len(usable)}")

    usable_set = set(usable)
    affected = [i for i in tie_affected(rows) if i in usable_set]
    print(f"tie-affected paths found by construction: {len(affected)}")

    configs = grid()
    rng = random.Random(args.seed)
    far = [
        c
        for c in configs
        if (c.take_profit_pct is not None and float(c.take_profit_pct) >= 0.08)
        or (c.trail_after_gain is not None and float(c.trail_after_gain) >= 0.08)
    ]
    chosen = [TradingLimits()]
    chosen += rng.sample(configs, min(args.random_configs, len(configs)))
    chosen += rng.sample(far, min(args.far_configs, len(far)))

    pool_idx = sorted(set(affected) | set(rng.sample(usable, min(args.random_paths, len(usable)))))
    print(
        f"verifying {len(chosen)} configs x {len(pool_idx)} paths"
        f" ({len(affected)} of them tie-affected) = {len(chosen) * len(pool_idx)} comparisons",
        flush=True,
    )

    flat = FLATTEN.hour * 60 + FLATTEN.minute
    keep_paths, keep_reduced = [], []
    for i in pool_idx:
        p = Path.from_row(rows[i])
        keep_paths.append(p)
        keep_reduced.append(reduce_path(p.bars, p.buy_minute, p.buy_price, flat, ORB_WIDTHS))
    rows.clear()
    keep = [i for i, r in enumerate(keep_reduced) if r is not None]

    global _PATHS, _REDUCED, _CONFIGS  # noqa: PLW0603
    _PATHS, _REDUCED, _CONFIGS = keep_paths, keep_reduced, chosen
    bad: list[str] = []
    with Pool(args.procs) as pool:
        for res in pool.imap_unordered(_check, keep, chunksize=8):
            bad += res
    for line in bad[:10]:
        print(f"  MISMATCH {line}")
    print(f"twin verification -> {len(bad)} mismatches")
    if bad:
        raise SystemExit("twin does not match the live predicate")


if __name__ == "__main__":
    main()
