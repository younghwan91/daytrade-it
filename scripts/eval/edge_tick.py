"""Does moving to wider-priced names throw away the signal it saves in spread?

scalp-it's point: tick width is a function of price, price correlates with
company size, and small caps react harder to news. So a universe chosen to
save spread may be quietly selling the signal. Our measured selection alpha is
+1.6 +/- 3.4 bp, so there is probably nothing to sell -- but that is an
assumption, and this is one line of analysis to test it.

Bucket boundaries are re-cut on OUR pool's own distribution rather than
inherited: our mean tick is 12.5 bp, which sits mid-bucket in a scheme built
for a different price mix, so the within-bucket spread would be large.
Quintiles of our own tick/price put the boundaries where our mass actually is.

Read the result as "is there less room to win where the tick is wide", not as
"trade the high-priced bucket". The spread saving is realisable; any alpha
difference here is measured on the same +/-3 bp noise floor as everything else
in this pool.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from edge_cond import day_mean

BP = 10_000
TICKS = ((2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50), (200_000, 100), (500_000, 500))
TOP_TICK = 1_000


def tick_bp(price: float) -> float:
    tick = next((t for bound, t in TICKS if price < bound), TOP_TICK)
    return tick / price


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="data/eval/edge/rows_sweep.jsonl")
    ap.add_argument("--labels", default="data/eval/prompt_v3/labels.jsonl")
    ap.add_argument("--nq", type=int, default=5)
    args = ap.parse_args()

    entry: dict[str, float] = {}
    with Path(args.labels).open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("entry"):
                entry[r["key"]] = float(r["entry"])

    rows = []
    with Path(args.rows).open() as f:
        for line in f:
            r = json.loads(line)
            e = entry.get(r["key"])
            if e and e > 0:
                r["tick"] = tick_bp(e)
                rows.append(r)

    ticks = sorted(r["tick"] for r in rows)
    edges = [ticks[int(len(ticks) * k / args.nq)] for k in range(1, args.nq)]
    print(f"\n{len(rows):,} pairs   mean tick {statistics.fmean(ticks) * BP:.1f} bp")
    print("quintile edges (bp): " + ", ".join(f"{e * BP:.1f}" for e in edges) + "\n")

    def bucket(t: float) -> int:
        return sum(t >= e for e in edges)

    by: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by[bucket(r["tick"])].append(r)

    print(
        f"{'tick bucket':<18}{'n':>8}{'tick bp':>9}{'alpha':>9}{'se':>7}"
        f"{'net':>9}{'days>=0':>10}  months"
    )
    months = sorted({r["month"] for r in rows})
    lab = (
        [f"<{edges[0] * BP:.0f}"]
        + [f"{edges[i] * BP:.0f}-{edges[i + 1] * BP:.0f}" for i in range(len(edges) - 1)]
        + [f">{edges[-1] * BP:.0f}"]
    )
    for b in sorted(by):
        rs = by[b]
        a_mu, a_se, nd, a_ge = day_mean([(r["day"], r["alpha"]) for r in rs])
        n_mu, _, _, _ = day_mean([(r["day"], r["net"]) for r in rs])
        marks = ""
        for m in months:
            sel = [r for r in rs if r["month"] == m]
            if len(sel) < 50:
                marks += "."
                continue
            mu, _, _, _ = day_mean([(r["day"], r["alpha"]) for r in sel])
            marks += "+" if mu > 0 else "-"
        print(
            f"{lab[b]:<18}{len(rs):>8,}{statistics.fmean(r['tick'] for r in rs) * BP:>9.1f}"
            f"{a_mu * BP:>+9.1f}{a_se * BP:>7.1f}{n_mu * BP:>+9.1f}{a_ge:>6}/{nd:<3}  {marks}"
        )
    print(
        "\nalpha is selection excess over a matched null; `net` still carries the "
        "legacy 21.5 bp\ncost constant from rows_sweep.jsonl, so read the alpha "
        "column for the signal question.\n"
    )


if __name__ == "__main__":
    main()
