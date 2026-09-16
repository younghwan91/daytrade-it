"""The decomposition ladder: index -> universe -> news-tagged -> our pick.

For every pair we already know what our selection returned. `edge_null.py` adds
what the same day, the same minute and the same universe returned without any
news signal, and what the index returned over the identical window. Stacking
those gives four rungs and three deltas:

    R0 index        069500 held from the decision minute to the close
    R1 universe     the equal-weight top-300 cross-section, same minute
    R2 pool         the news-tagged names we actually judged
    R3 buy          the subset our score would have bought

    D1 = R1 - R0    universe vs index          (cross-sectional drift)
    D2 = R2 - R1    news tag vs no news        <- is the NEWS the edge
    D3 = R3 - R2    our score within the tag   <- is our CHOOSING the edge

Every delta is formed pair by pair against its own minute's null, so the day
and the minute are differenced out exactly. Inference is day-clustered -- a
day is the unit that repeats, not a pair -- with a bootstrap over days.

Costs cancel in every delta (both legs are one round trip), so rungs are gross
and deltas are cost-free; the net column exists only to tie R2/R3 back to the
published -52 bp / +18 bp / +39 bp figures.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

COST = 0.00215
BUY_SCORE = 0.15
BP = 10_000


def load_null(path: Path) -> dict[tuple[str, int], dict]:
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out[(r["day"], r["act_min"])] = r
    return out


def load_pool(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def day_stats(by_day: dict[str, list[float]], reps: int, seed: int) -> dict:
    """Mean of pair-level values, clustered on the day."""
    days = sorted(by_day)
    per_day = [statistics.fmean(by_day[d]) for d in days]
    flat = [v for d in days for v in by_day[d]]
    n = len(days)
    rng = random.Random(seed)
    boots = []
    for _ in range(reps):
        pick = [per_day[rng.randrange(n)] for _ in range(n)]
        boots.append(statistics.fmean(pick))
    boots.sort()
    return {
        "n_pairs": len(flat),
        "n_days": n,
        "pair_mean": statistics.fmean(flat),
        "day_mean": statistics.fmean(per_day),
        "day_se": statistics.stdev(per_day) / (n**0.5) if n > 1 else float("nan"),
        "lo": boots[int(0.025 * reps)],
        "hi": boots[int(0.975 * reps)],
        "p_gt0": sum(b > 0 for b in boots) / reps,
        "days_ge0": sum(v >= 0 for v in per_day),
        "worst_day": min(per_day),
        "worst_day_on": days[per_day.index(min(per_day))],
    }


def bucket(rows: list[tuple[str, float]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for day, v in rows:
        out[day].append(v)
    return out


def show(name: str, rows: list[tuple[str, float]], reps: int, seed: int, net: bool = False) -> None:
    if not rows:
        print(f"{name:<34} (empty)")
        return
    s = day_stats(bucket(rows), reps, seed)
    adj = -COST if net else 0.0
    print(
        f"{name:<34} n={s['n_pairs']:>6d} d={s['n_days']:>3d} "
        f"pair={(s['pair_mean'] + adj) * BP:+8.1f}  day={(s['day_mean'] + adj) * BP:+8.1f} "
        f"se={s['day_se'] * BP:6.1f}  "
        f"CI[{(s['lo'] + adj) * BP:+7.1f},{(s['hi'] + adj) * BP:+7.1f}] "
        f"p>0={s['p_gt0']:.3f}  days>=0={s['days_ge0']:>3d}/{s['n_days']:<3d} "
        f"worst={(s['worst_day'] + adj) * BP:+8.1f}({s['worst_day_on']})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--null", required=True)
    ap.add_argument("--label", default="pool")
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--days", help="comma-separated subset of days")
    ap.add_argument(
        "--strata",
        help="strata.jsonl from edge_strata.py; matches each pair to the null "
        "inside its own volatility x liquidity cell instead of the whole universe",
    )
    args = ap.parse_args()

    nulls = load_null(Path(args.null))
    pool = load_pool(Path(args.pool))
    cell_of: dict[tuple[str, str], str] = {}
    if args.strata:
        with Path(args.strata).open() as f:
            for line in f:
                s = json.loads(line)
                cell_of[(s["day"], s["code"])] = f"{s['sig_q']}|{s['val_q']}"
    if args.days:
        keep = set(args.days.split(","))
        pool = [r for r in pool if r["day"] in keep]

    r0: list[tuple[str, float]] = []
    r1: list[tuple[str, float]] = []
    r2: list[tuple[str, float]] = []
    r3: list[tuple[str, float]] = []
    d1: list[tuple[str, float]] = []
    d2: list[tuple[str, float]] = []
    d2_buy: list[tuple[str, float]] = []
    d3: list[tuple[str, float]] = []
    dropped = 0

    thin = 0
    for r in pool:
        n = nulls.get((r["day"], r["act_min"]))
        if n is None or n.get("mkt_ret") is None or r.get("ret_close") is None:
            dropped += 1
            continue
        day, ret = r["day"], r["ret_close"]
        null = n["null_mean"]
        if cell_of:
            code = r.get("code") or r["key"].split("|")[1]
            cell = (n.get("cells") or {}).get(cell_of.get((day, code), ""))
            # A cell thinner than 5 names is its own noise, not a null.
            if not cell or cell["n"] < 5:
                thin += 1
                continue
            null = cell["mean"]
        r0.append((day, n["mkt_ret"]))
        r1.append((day, null))
        r2.append((day, ret))
        d1.append((day, null - n["mkt_ret"]))
        d2.append((day, ret - null))
        score = r.get("score")
        if score is None:
            score = (r.get("v2") or {}).get("score")
        if score is not None and score > BUY_SCORE:
            r3.append((day, ret))
            d2_buy.append((day, ret - n["null_mean"]))

    pool_day_mean = {d: statistics.fmean(v) for d, v in bucket(r2).items()}
    for day, ret in r3:
        d3.append((day, ret - pool_day_mean[day]))

    match = "sigma x value cell" if cell_of else "whole top-300"
    print(
        f"\n=== {args.label}  ({len(pool)} pairs, {dropped} without a matched null, "
        f"{thin} in a cell under 5 names)\n    null = {match}\n"
    )
    print("--- rungs (gross, hold to close)")
    show("R0 index 069500", r0, args.reps, args.seed)
    show(f"R1 null ({match})", r1, args.reps, args.seed)
    show("R2 pool (news-tagged)", r2, args.reps, args.seed)
    show("R3 buy (score > 0.15)", r3, args.reps, args.seed)
    print("\n--- rungs (net of 21.5 bp round trip, for comparison with published figures)")
    show("R2 pool net", r2, args.reps, args.seed, net=True)
    show("R3 buy net", r3, args.reps, args.seed, net=True)
    print("\n--- deltas (cost cancels; each pair differenced against its own minute)")
    show("D1 universe - index", d1, args.reps, args.seed)
    show("D2 pool - universe", d2, args.reps, args.seed)
    show("D2b buy - universe", d2_buy, args.reps, args.seed)
    show("D3 buy - pool (same day)", d3, args.reps, args.seed)
    print()


if __name__ == "__main__":
    main()
