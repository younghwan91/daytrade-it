"""D1/D2 on the hourly ladder, sliced by day subset -- reuses edge_ladder.

Pairs are matched to the null inside their own 4x4 sigma x value cell, exactly
as `edge_ladder.py --strata` does; this file only adds the day slicing (search
vs confirm window, by year, reproduction overlap) and prints one line per slice.

Cost cancels in D1 and D2 -- both legs are one round trip over the identical
window -- so the deltas are gross-equals-net and the 36.7 bp toll is quoted
only against the R2 rung, never subtracted from a delta.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_EVAL = Path("/home/young/git/daytrade-it-rlnews/scripts/eval")
sys.path.insert(0, str(_EVAL))
from edge_ladder import BP, bucket, day_stats  # noqa: E402

MIN_CELL = 5


def load(labels: Path, null: Path, strata: Path):
    nulls = {}
    with null.open() as f:
        for line in f:
            r = json.loads(line)
            nulls[(r["day"], r["act_min"])] = r
    cell_of = {}
    with strata.open() as f:
        for line in f:
            s = json.loads(line)
            cell_of[(s["day"], s["code"])] = f"{s['sig_q']}|{s['val_q']}"
    rows = []
    thin = dropped = 0
    with labels.open() as f:
        for line in f:
            r = json.loads(line)
            n = nulls.get((r["day"], r["act_min"]))
            if n is None or n.get("mkt_ret") is None or r.get("ret_close") is None:
                dropped += 1
                continue
            cell = (n.get("cells") or {}).get(cell_of.get((r["day"], r["code"]), ""))
            if not cell or cell["n"] < MIN_CELL:
                thin += 1
                continue
            rows.append(
                {
                    "day": r["day"],
                    "act_min": r["act_min"],
                    "ret": r["ret_close"],
                    "null": cell["mean"],
                    "univ": n["null_mean"],
                    "mkt": n["mkt_ret"],
                }
            )
    return rows, dropped, thin


def line(name: str, vals: list[tuple[str, float]], reps: int, seed: int) -> None:
    if not vals:
        print(f"{name:<40} (empty)")
        return
    s = day_stats(bucket(vals), reps, seed)
    print(
        f"{name:<40} n={s['n_pairs']:>7d} d={s['n_days']:>3d} "
        f"pair={s['pair_mean'] * BP:+7.2f}  day={s['day_mean'] * BP:+7.2f} "
        f"se={s['day_se'] * BP:5.2f} CI[{s['lo'] * BP:+6.2f},{s['hi'] * BP:+6.2f}] "
        f"p>0={s['p_gt0']:.3f} days>0={s['days_ge0']:>3d}/{s['n_days']:<3d}"
    )


def slab(tag: str, rows: list[dict], reps: int, seed: int) -> None:
    if not rows:
        print(f"\n### {tag}: no pairs\n")
        return
    d2 = [(r["day"], r["ret"] - r["null"]) for r in rows]
    d2u = [(r["day"], r["ret"] - r["univ"]) for r in rows]
    d1 = [(r["day"], r["univ"] - r["mkt"]) for r in rows]
    r2 = [(r["day"], r["ret"]) for r in rows]
    print(f"\n### {tag}")
    line("R2 pool gross (hold to close)", r2, reps, seed)
    line("D1 universe - broad cross-section", d1, reps, seed)
    line("D2 pool - universe (whole top-300)", d2u, reps, seed)
    line("D2 pool - universe (sigma x value cell)", d2, reps, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    base = Path("/home/young/git/daytrade-it-rlnews/data/eval/rlnews")
    ap.add_argument("--labels", default=str(base / "labels_hourly.jsonl"))
    ap.add_argument("--null", default=str(base / "null_hourly_strat.jsonl"))
    ap.add_argument("--strata", default=str(base / "strata_hourly.jsonl"))
    ap.add_argument("--windows", default=str(base / "CONFIRM_WINDOWS.json"))
    ap.add_argument("--track", default="track_b_hourly")
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only-days", help="comma-separated day whitelist")
    ap.add_argument("--tag", default="all")
    ap.add_argument("--by-year", action="store_true")
    ap.add_argument("--split-confirm", action="store_true")
    args = ap.parse_args()

    rows, dropped, thin = load(Path(args.labels), Path(args.null), Path(args.strata))
    if args.only_days:
        keep = set(args.only_days.split(","))
        rows = [r for r in rows if r["day"] in keep]
    days = sorted({r["day"] for r in rows})
    print(
        f"loaded {len(rows)} pairs / {len(days)} days  "
        f"({dropped} without a matched null, {thin} in a cell under {MIN_CELL} names)"
    )
    if days:
        print(f"span {days[0]} .. {days[-1]}")

    if args.split_confirm:
        w = json.loads(Path(args.windows).read_text())[args.track]
        confirm = set(w["confirm"])
        lo, hi = w["search"]
        search = [r for r in rows if lo <= r["day"] <= hi and r["day"] not in confirm]
        conf = [r for r in rows if r["day"] in confirm]
        other = [r for r in rows if r not in search and r not in conf] if False else None
        slab(f"SEARCH {lo}..{hi} (frozen in advance)", search, args.reps, args.seed)
        if args.by_year:
            byy = defaultdict(list)
            for r in search:
                byy[r["day"][:4]].append(r)
            for y in sorted(byy):
                slab(f"SEARCH year {y}", byy[y], args.reps, args.seed)
        slab(
            f"CONFIRM {w['confirm'][0]}..{w['confirm'][-1]} ({len(confirm)} days, frozen in advance)",
            conf,
            args.reps,
            args.seed,
        )
        _ = other
    else:
        slab(args.tag, rows, args.reps, args.seed)
        if args.by_year:
            byy = defaultdict(list)
            for r in rows:
                byy[r["day"][:4]].append(r)
            for y in sorted(byy):
                slab(f"year {y}", byy[y], args.reps, args.seed)


if __name__ == "__main__":
    main()
