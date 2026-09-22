"""The exactness gate the sequential simulator must pass before it may search.

`exit_fast.py` earned its place by being held against the live predicate on
every path, and the one time that gate was sampled rather than exhaustive it
passed a twin that was not one (the Decimal-tie defect, research doc 5.2).
`seqexit_core.Sim` is a THIRD implementation of the same walk, so it gets the
same treatment and does not get to skip it for being slow to build.

Two stages, both exhaustive over the book:

1. **Reduction identity.** Every field `seqexit_core.reduce_row` shares with
   `exit_fast.reduce_path` -- decision minutes, sell fills, last-completed
   close, VWAP validity, flatten index -- must agree elementwise on all 29,800
   paths. If the two disagree about WHEN a decision happens or WHAT it
   transacts at, no later comparison between them means anything.

2. **Driven identity.** Driven with a plain stop-loss, the sequential walk must
   return the exit minute `exit_fast.first_exit` returns, at every level in
   `rl_exits.STOP_LOSS`, on every path. Thresholds go through
   `exit_fast.barrier` on both sides, so the Decimal tie is inside the test
   rather than underneath it.

`first_exit` is held against the live `exit_rule_hit` by `verify_twin.py`, so
passing 1 and 2 places this simulator against the live function transitively.

    .venv/bin/python scripts/eval/seqexit_gate.py \
        --paths data/eval/rlnews/paths_prior.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path as FsPath

import numpy as np

sys.path.insert(0, "scripts/eval")

from exit_fast import barrier, first_exit, reduce_path  # noqa: E402
from exit_lab import FLATTEN, Path  # noqa: E402
from rl_exits import ORB_WIDTHS, STOP_LOSS  # noqa: E402
from seqexit_core import LATTICES, Sim, build, reduce_row  # noqa: E402

from daytrade_it.application.trading.auto_trader import TradingLimits  # noqa: E402

#: Everything off, so the only thing that can fire is the stop-loss under test.
BARE = TradingLimits(
    stop_loss_pct=None,
    stop_vwap_gap=None,
    stop_time=None,
    stop_orb=None,
    take_profit_pct=None,
    trail_after_gain=None,
    vwap_exit_gap=None,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=FsPath, required=True)
    ap.add_argument("--out", type=FsPath)
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (smoke runs)")
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.paths.read_text().splitlines() if line]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{args.paths}: {len(rows)} rows", flush=True)

    flat = FLATTEN.hour * 60 + FLATTEN.minute
    mine, theirs, kept_rows = [], [], []
    for row in rows:
        a = reduce_row(row)
        p = Path.from_row(row)
        b = reduce_path(p.bars, p.buy_minute, p.buy_price, flat, ORB_WIDTHS)
        if (a is None) != (b is None):
            raise SystemExit(f"usability disagrees on {row['day']} {row['ticker']}")
        if a is None:
            continue
        mine.append(a)
        theirs.append(b)
        kept_rows.append(row)
    print(f"usable {len(mine)} (dropped {len(rows) - len(mine)}) -- agrees with exit_fast")

    # -- stage 1: reduction identity ----------------------------------------
    bad = {"minute": 0, "fill": 0, "close": 0, "valid": 0, "length": 0}
    for a, b in zip(mine, theirs, strict=True):
        if a["minute"].size != b.minute.size:
            bad["length"] += 1
            continue
        bad["minute"] += int((a["minute"] != b.minute).sum())
        bad["fill"] += int((a["fill"] != b.fill).sum())
        bad["close"] += int((a["close"] != b.close).sum())
        bad["valid"] += int((a["valid"] != np.isfinite(b.vwap)).sum())
    print(f"stage 1 reduction identity: {bad}")
    if any(bad.values()):
        raise SystemExit("reduction differs from exit_fast; refusing to search")

    # -- stage 2: driven identity -------------------------------------------
    book, _ = build(kept_rows)
    sim = Sim(book, LATTICES["vol"], np.zeros(book.n_paths))
    entry_per_step = np.repeat(book.entry, book.stop - book.starts)
    mismatches = []
    for pct in [p for p in STOP_LOSS if p is not None]:
        lim = dataclasses.replace(BARE, stop_loss_pct=pct)
        ref = np.array([first_exit(b, lim, ORB_WIDTHS)[2] for b in theirs])
        got_idx = sim.run_fixed_stop(barrier(entry_per_step, pct, -1))
        got = book.minute[got_idx]
        n_bad = int((ref != got).sum())
        print(f"  stop_loss={float(pct):.3f}: {n_bad} mismatches over {len(ref)} paths")
        if n_bad:
            where = np.flatnonzero(ref != got)[:5]
            for i in where:
                mismatches.append(
                    f"{book.day[i]} {book.ticker[i]} sl={float(pct)}: "
                    f"exit_fast {ref[i]} vs seq {got[i]}"
                )
    for m in mismatches[:10]:
        print(f"  MISMATCH {m}")
    print(f"stage 2 driven identity -> {len(mismatches)} mismatching paths")
    if mismatches:
        raise SystemExit("sequential walk differs from the verified twin; refusing to search")

    # A stop-loss that never fires must equal hold-to-flatten, and the
    # asymmetry must actually hold: no cut above entry, no add below it.
    hold_idx = sim.run_fixed_stop(np.zeros(book.n_steps))
    if not np.array_equal(hold_idx, book.last_idx):
        raise SystemExit("an unreachable stop still moved the exit")
    assert not (sim.cut_legal & np.isin(book.pnl_b, (3, 4))).any()
    assert not (sim.add_legal & np.isin(book.pnl_b, (0, 1, 2))).any()
    print("asymmetry holds: cut only below entry, add only at or above it")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "rows": len(rows),
                    "usable": len(mine),
                    "reduction_mismatches": bad,
                    "driven_levels": [float(p) for p in STOP_LOSS if p is not None],
                    "driven_mismatches": len(mismatches),
                },
                indent=1,
            )
        )
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
