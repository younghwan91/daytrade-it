"""Does the no-chase filter throw away the only part of the pool that works?

The hypothesis: the pool's alpha lives in names that gap and run, and
`generate_no_chase_confirmation` refuses to buy anything already up more than
`MAX_PRE_ENTRY_MOVE` versus the prior close -- so the filter may be removing
exactly the part that works. That rule was fitted on 209 pairs over 10 days in
the 2026-09-13 eval, the same eval whose founding baseline did not replicate,
and has never been re-examined.

**There is a conflation in the hypothesis and this tests it first.** Two
different variables are being run together:

    pre_move      entry price vs the PRIOR close -- what the filter sees,
                  known before we act
    never dipped  the name never traded a tick below our entry AFTER the
                  decision minute -- what the limit-fill split found carries
                  +246 bp

A name can be flat on the day and still never dip after we look at it. If those
two are only weakly related, "the alpha is in the gappers" does not follow from
"the alpha is in the names a resting bid misses", and the filter is not
implicated. So the first table is the cross.

Then, on the filter itself: alpha across the whole pre_move range including the
region live refuses, monthly sign stability, how much of the pool's positive
alpha sits above each line, and -- the crux -- whether that region is buyable
at all, since a name up 3% and still running may be precisely the one a resting
bid never catches.

Note live's constant is 0.03, not the 0.02 it was fitted at; it was raised on
2026-09-15 on six days of leave-one-day-out. Both lines are reported.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from edge_persist import day_stats

BP = 10_000
HORIZONS = ("0", "1", "2", "5", "10", "20", "40")
#: the fitted line, the live line, and the range either side
CUTS = (0.0, 0.01, 0.02, 0.03, 0.05)
BUCKETS = (
    (-9.9, -0.03),
    (-0.03, -0.01),
    (-0.01, 0.003),
    (0.003, 0.01),
    (0.01, 0.02),
    (0.02, 0.03),
    (0.03, 0.05),
    (0.05, 9.9),
)


def label(lo: float, hi: float) -> str:
    f = lambda x: "-inf" if x < -9 else "+inf" if x > 9 else f"{x * 100:+.1f}%"  # noqa: E731
    return f"{f(lo)} .. {f(hi)}"


def stat(rows: list[dict], h: str) -> tuple[float, float, int] | None:
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if h in r["r"] and h in r["nc"]:
            by[r["day"]].append(r["r"][h] - r["nc"][h])
    n = sum(len(v) for v in by.values())
    if n < 150:
        return None
    mu, se, _, _, _ = day_stats(by, int(h) + 1)
    return mu, se, n


def months_sign(rows: list[dict], h: str) -> str:
    out = ""
    for m in sorted({r["day"][:7] for r in rows}):
        sel = [r for r in rows if r["day"][:7] == m]
        s = stat(sel, h)
        out += "." if s is None else ("+" if s[0] > 0 else "-")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/eval/edge/horizon_pairs.jsonl")
    ap.add_argument("--fills", default="data/eval/edge/limit_fills.jsonl")
    args = ap.parse_args()

    rows = []
    with Path(args.pairs).open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("pre_move") is not None and r.get("nc"):
                rows.append(r)
    ok = {r["day"] for r in rows if all(h in r["r"] for h in HORIZONS)}
    rows = [r for r in rows if r["day"] in ok]

    fills: dict[str, dict] = {}
    if Path(args.fills).exists():
        with Path(args.fills).open() as f:
            for line in f:
                x = json.loads(line)
                fills[x["key"]] = x
    for r in rows:
        x = fills.get(r["key"])
        r["missed"] = (not x["fills"].get("390")) if x else None

    print(f"\n{len(rows):,} pairs with a known prior close, {len(ok)} common days\n")

    # --- 0. the conflation check
    print("=== 0. are 'gapped up' and 'never dipped after entry' the same names?")
    have = [r for r in rows if r["missed"] is not None]
    print(f"{'pre_move bucket':<20}{'n':>8}{'never dipped':>14}{'alpha same-day':>16}")
    for lo, hi in BUCKETS:
        sel = [r for r in have if lo <= r["pre_move"] < hi]
        if len(sel) < 150:
            continue
        s = stat(sel, "0")
        print(
            f"{label(lo, hi):<20}{len(sel):>8,}{statistics.fmean(r['missed'] for r in sel):>13.1%}"
            f"{(s[0] * BP if s else float('nan')):>+16.1f}"
        )
    base = statistics.fmean(r["missed"] for r in have)
    print(f"{'ALL':<20}{len(have):>8,}{base:>13.1%}\n")

    # --- 1. alpha across the whole range, by horizon
    print("=== 1. alpha by pre_move bucket and horizon (monthly signs for same-day)")
    print(
        f"{'bucket':<20}{'n':>8}"
        + "".join(f"{('d0' if h == '0' else '+' + h):>9}" for h in HORIZONS)
        + "  months"
    )
    for lo, hi in BUCKETS:
        sel = [r for r in rows if lo <= r["pre_move"] < hi]
        if len(sel) < 150:
            continue
        cells = ""
        for h in HORIZONS:
            s = stat(sel, h)
            cells += f"{s[0] * BP:>+9.0f}" if s else f"{'.':>9}"
        print(f"{label(lo, hi):<20}{len(sel):>8,}{cells}  {months_sign(sel, '0')}")

    # --- 2. what each line removes
    print("\n=== 2. what a no-chase line at each cut removes")
    print(
        f"{'cut':<10}{'kept n':>9}{'kept %':>8}{'kept alpha d0':>15}{'removed alpha d0':>18}"
        f"{'kept alpha +10d':>17}{'removed +10d':>14}"
    )
    for c in CUTS:
        keep = [r for r in rows if r["pre_move"] <= c]
        drop = [r for r in rows if r["pre_move"] > c]
        k0, d0 = stat(keep, "0"), stat(drop, "0")
        k10, d10 = stat(keep, "10"), stat(drop, "10")
        fmt = lambda s: f"{s[0] * BP:>+.1f}" if s else "."  # noqa: E731
        print(
            f"{'<= ' + f'{c * 100:.0f}%':<10}{len(keep):>9,}{len(keep) / len(rows):>8.1%}"
            f"{fmt(k0):>15}{fmt(d0):>18}{fmt(k10):>17}{fmt(d10):>14}"
        )

    # --- 3. is the rejected region buyable?
    print("\n=== 3. is the region live refuses actually buyable?")
    print(f"{'bucket':<20}{'n':>8}{'limit fill rate':>17}{'never dipped':>14}")
    for lo, hi in ((0.02, 0.03), (0.03, 0.05), (0.05, 9.9)):
        sel = [r for r in have if lo <= r["pre_move"] < hi]
        if len(sel) < 150:
            continue
        miss = statistics.fmean(r["missed"] for r in sel)
        print(f"{label(lo, hi):<20}{len(sel):>8,}{1 - miss:>16.1%}{miss:>14.1%}")
    print()


if __name__ == "__main__":
    main()
