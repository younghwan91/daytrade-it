"""Evaluate gates stated in advance, month by month, instead of searched for.

`edge_folds.py` searches the whole (axis, level) grid and its winner does not
transfer -- which is what a search over a few hundred levels on a pool with no
edge should do. This file exists for the opposite case: a hypothesis with a
shape, written down before the months are looked at, so that sign stability
across months means something.

The one such hypothesis this pool offers is monotone in the pre-entry move:
alpha falls from +16 bp at -1..-0.3% to -21 bp above +3%, in that order, across
all seven buckets. A monotone response over an ordered variable is a much
stronger prior than "this level was highest". The gates below therefore all
say the same thing at different cut points -- do not buy what has already run.

Each gate reports the pool it keeps and, separately, what it drops, because a
filter is only worth having if the part it throws away is the bad part.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from edge_cond import day_mean

BP = 10_000

GATES: dict[str, tuple[float, float]] = {
    "pre_move <= +3%": (-9.9, 0.03),
    "pre_move <= +1%": (-9.9, 0.01),
    "pre_move <= +0.3%": (-9.9, 0.003),
    "pre_move <= 0%": (-9.9, 0.0),
    "pre_move in [-3%, +1%]": (-0.03, 0.01),
    "pre_move in [-3%, 0%]": (-0.03, 0.0),
}


def summarise(name: str, rs: list[dict], months: list[str]) -> str:
    if len(rs) < 50:
        return f"{name:<26} (too few)"
    a_mu, a_se, nd, _ = day_mean([(r["day"], r["alpha"]) for r in rs])
    n_mu, _, _, n_ge = day_mean([(r["day"], r["net"]) for r in rs])
    per: dict[str, list[float]] = defaultdict(list)
    for r in rs:
        per[r["day"]].append(r["net"])
    worst = min(statistics.fmean(v) for v in per.values())
    losers = sum(1 for r in rs if r["net"] < 0) / len(rs)
    marks = ""
    for m in months:
        sel = [r for r in rs if r["month"] == m]
        if len(sel) < 50:
            marks += "."
            continue
        mu, _, _, _ = day_mean([(r["day"], r["alpha"]) for r in sel])
        marks += "+" if mu > 0 else "-"
    return (
        f"{name:<26} n={len(rs):>6} alpha={a_mu * BP:>+7.1f}(se{a_se * BP:>5.1f}) "
        f"net={n_mu * BP:>+7.1f} d>=0={n_ge:>3}/{nd:<3} worst={worst * BP:>+8.1f} "
        f"losers={losers:.3f}  {marks}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--label", default="pool")
    ap.add_argument("--days", help="comma-separated subset of days")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines()]
    if args.days:
        keep_days = set(args.days.split(","))
        rows = [r for r in rows if r["day"] in keep_days]
    known = [r for r in rows if r["f"]["pre_move"] != "na"]
    months = sorted({r["month"] for r in known})
    print(
        f"\n=== {args.label}: {len(known)} pairs with a known previous close "
        f"(of {len(rows)}), {months[0]}..{months[-1]}\n"
    )
    print(summarise("ALL (pre_move known)", known, months))
    print()

    raw = {r["key"]: r for r in known}
    for name, (lo, hi) in GATES.items():
        keep, drop = [], []
        for r in known:
            v = r.get("pre_move_raw")
            (keep if v is not None and lo <= v <= hi else drop).append(r)
        if not raw:
            break
        print(summarise(f"KEEP {name}", keep, months))
        print(summarise("  drop", drop, months))
    print()


if __name__ == "__main__":
    main()
