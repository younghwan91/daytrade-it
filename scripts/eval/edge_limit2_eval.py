"""What retrying buys, and whether through-fills are a different animal.

Three questions, kept apart:

  1. How much of the "missed winner" problem survives retrying? The first pass
     found the unfilled 3.3% carried the pool's entire alpha; if a retry
     acquires almost all of them, that opportunity cost largely evaporates.
  2. Do the two kinds of day-0 miss behave differently -- price gone (A) versus
     price there but not quite reached (B)?
  3. **Through-fills**, where price traded past our limit rather than resting
     on it, are the maximum-adverse-selection subset: we bought because it
     fell. Whether that subset still carries alpha at +10d is the most direct
     answer available to "does the spread/adverse-selection cancellation decay
     with horizon".

Returns are rescaled exactly rather than re-simulated. The exit date is fixed,
so filling later at price P gives `(1 + ret_market) * entry / P - 1` -- a slow
acquisition eats its own holding period, which is the honest treatment.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from edge_persist import day_stats

BP = 10_000
COST = 0.00367
HORIZONS = ("0", "1", "2", "5", "10", "20", "40")


def alpha(rows: list[dict], h: str, at_fill: bool) -> tuple[float, float, int] | None:
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if h not in r["r"] or h not in r["nc"]:
            continue
        ret = r["r"][h]
        if at_fill:
            px = r.get("fill_px")
            if not px:
                continue
            ret = (1 + ret) * r["entry"] / px - 1
        by[r["day"]].append(ret - r["nc"][h])
    n = sum(len(v) for v in by.values())
    if n < 150:
        return None
    mu, se, _, _, _ = day_stats(by, int(h) + 1)
    return mu, se, n


def line(label: str, s, extra: str = "") -> str:
    if s is None:
        return f"  {label:<26} (too few)"
    return f"  {label:<26}{s[2]:>8,}{s[0] * BP:>+10.1f}{s[1] * BP:>7.1f}{extra}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="data/eval/edge/limit_retry.jsonl")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines()]
    ok = {r["day"] for r in rows if all(h in r["r"] for h in HORIZONS)}
    rows = [r for r in rows if r["day"] in ok]
    fd = Counter(r["fill_day"] for r in rows)
    tot = len(rows)
    print(f"\n{tot:,} pairs over {len(ok)} common days")
    print("acquisition by retry day:")
    cum = 0
    for k in sorted(x for x in fd if x is not None):
        cum += fd[k]
        print(f"  day {k}: {fd[k]:>7,}  cumulative {cum / tot:>6.1%}")
    print(f"  never : {fd.get(None, 0):>7,}  ({fd.get(None, 0) / tot:.1%})")
    mk = Counter(r["miss_kind"] for r in rows if r["miss_kind"])
    print(f"day-0 misses: {dict(mk)}")
    thr = sum(1 for r in rows if r.get("through"))
    print(
        f"through-fills: {thr:,} of {sum(1 for r in rows if r['fill_px']):,} fills "
        f"({thr / max(1, sum(1 for r in rows if r['fill_px'])):.1%})\n"
    )

    same_day = [r for r in rows if r["fill_day"] == 0]
    retried = [r for r in rows if r["fill_day"] not in (0, None)]
    through = [r for r in rows if r.get("through") and r["fill_px"]]
    rested = [r for r in rows if r["fill_px"] and not r.get("through")]
    filled = [r for r in rows if r["fill_px"]]

    print(f"{'':<28}{'n':>8}{'alpha':>10}{'se':>7}")
    for h in HORIZONS:
        print(f"--- {'same day' if h == '0' else f'+{h}d'}")
        print(line("market (today's policy)", alpha(rows, h, False)))
        print(line("limit, any retry", alpha(filled, h, True)))
        print(line("  filled day 0", alpha(same_day, h, True)))
        print(line("  filled on a retry", alpha(retried, h, True)))
        print(line("  through-fill", alpha(through, h, True)))
        print(line("  rested-fill", alpha(rested, h, True)))
    print(
        "\n`limit, any retry` is priced at whatever it actually filled at; the exit "
        "date is\nunchanged, so a later fill has less holding time. Alpha is excess "
        "over the same\nmatched null throughout.\n"
    )


if __name__ == "__main__":
    main()
