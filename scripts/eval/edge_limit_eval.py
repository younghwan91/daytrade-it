"""Was the spread saving real, or paid back by adverse selection?

Reads `edge_limit.py`'s fills and answers the one question that decides the
limit-entry proposal: **what did the names we would have missed go on to do?**

For each fill window and horizon:

    market       every pair at the crossed price -- today's policy
    filled       what a resting bid caught, priced at the limit (so the tick
                 saving is inside the number)
    unfilled     what it missed, priced the way we actually bought it
    selection    alpha(filled at the crossed price) - alpha(market)
                 i.e. the fill's adverse selection with the tick saving
                 removed, so the two effects do not hide inside one another
    saving       the tick itself, in bp

A limit entry is worth having only if `saving + selection > 0` AND the
opportunity cost of the unfilled days does not exceed it. Both are printed.

Errors are moving-block bootstrapped at the holding length, because
consecutive windows of a multi-day hold overlap.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from edge_persist import day_stats

BP = 10_000
COST = 0.00367
HORIZONS = ("0", "1", "2", "5", "10", "20", "40")


def alpha_of(rows: list[dict], h: str, rescale: bool) -> tuple[float, float, int] | None:
    """Day-clustered alpha at horizon h; `rescale` prices the entry at the limit."""
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if h not in r["r"] or h not in r["nc"]:
            continue
        ret = r["r"][h]
        if rescale:
            ret = (1 + ret) * r["entry"] / r["limit"] - 1
        by[r["day"]].append(ret - r["nc"][h])
    if sum(len(v) for v in by.values()) < 200:
        return None
    mu, se, n, _, _ = day_stats(by, int(h) + 1)
    return mu, se, sum(len(v) for v in by.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fills", default="data/eval/edge/limit_fills.jsonl")
    ap.add_argument("--windows", default="5,30,390")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.fills).read_text().splitlines()]
    ok = {r["day"] for r in rows if all(h in r["r"] for h in HORIZONS)}
    rows = [r for r in rows if r["day"] in ok]
    saving = statistics.fmean(r["tick_bp"] for r in rows)
    print(
        f"\n{len(rows):,} pairs over {len(ok)} common days   "
        f"mean tick (the saving) {saving * BP:.1f} bp   cost {COST * BP:.1f} bp"
    )

    for w in args.windows.split(","):
        filled = [r for r in rows if r["fills"].get(w)]
        unfilled = [r for r in rows if not r["fills"].get(w)]
        rate = len(filled) / len(rows)
        print(
            f"\n=== resting bid, {w}-minute window   fill rate {rate:.1%} "
            f"({len(filled):,} filled / {len(unfilled):,} missed)"
        )
        print(
            f"{'hold':<8}{'market':>9}{'se':>6}{'filled@lim':>12}{'se':>6}"
            f"{'missed':>9}{'selection':>11}{'net@lim':>9}"
        )
        for h in HORIZONS:
            m = alpha_of(rows, h, False)
            fl = alpha_of(filled, h, True)
            fx = alpha_of(filled, h, False)
            un = alpha_of(unfilled, h, False)
            if not (m and fl and fx):
                continue
            sel = fx[0] - m[0]
            label = "same day" if h == "0" else f"+{h}d"
            print(
                f"{label:<8}{m[0] * BP:>+9.1f}{m[1] * BP:>6.1f}{fl[0] * BP:>+12.1f}{fl[1] * BP:>6.1f}"
                f"{(un[0] * BP if un else float('nan')):>+9.1f}{sel * BP:>+11.1f}"
                f"{(fl[0] - COST) * BP:>+9.1f}"
            )
    print(
        "\n`selection` is alpha(filled, priced as if crossed) - alpha(market): the "
        "adverse\nselection alone, with the tick saving removed so the two do not "
        "hide inside\neach other. The proposal needs saving + selection > 0 and the "
        "missed names\nnot to outrun the filled ones.\n"
    )


if __name__ == "__main__":
    main()
