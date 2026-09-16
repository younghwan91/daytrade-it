"""Scoping the prior-day |move| filter: cut point, how much it removes, and refill.

Three independent routes have arrived at the same line -- the ten-year universe
table, the news-pool horizon split, and the dashboard's own profile -- and the
ten-year cells are monotone with the top 1% of prior-day movers at -94 to -147
bp. It needs no new data, it is computable from the previous day's bar at
decision time, and unlike everything else in this investigation it has a decade
behind it.

What a filter needs before it can be proposed:

  cut point   where to draw the line, and what the decade says about each
  removal     what fraction of candidates it takes out
  REFILL      and this is the one that has bitten this project twice: the live
              book is capped at a fixed number of slots per day, so removing
              candidates does not shrink the book, it changes which names fill
              it. A filter that removes bad names and lets worse ones in is a
              loss. This measures the book before and after, at the cap.

Refill is modelled the way the live daemon fills: earliest acting minute first,
ties broken by the higher score, capped at `--max-positions` per day.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from edge_cond import day_mean

BP = 10_000
COST = 0.00367
BUY_SCORE = 0.15
CUTS = (0.99, 0.95, 0.90, 0.80, 0.67, 0.50)


def book(rows: list[dict], cap: int) -> list[dict]:
    """The names that actually get bought, under the live ordering and cap."""
    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r)
    out = []
    for day in sorted(by_day):
        picks = sorted(by_day[day], key=lambda r: (r["act_min"], -(r.get("score_raw") or 0.0)))
        out += picks[:cap]
    return out


def describe(label: str, rows: list[dict], cap: int) -> str:
    if len(rows) < 50:
        return f"{label:<26} (too few)"
    bk = book(rows, cap)
    a_mu, a_se, nd, _ = day_mean([(r["day"], r["alpha"]) for r in bk])
    n_mu, _, _, n_ge = day_mean([(r["day"], r["net"]) for r in bk])
    per: dict[str, list[float]] = defaultdict(list)
    for r in bk:
        per[r["day"]].append(r["net"])
    worst = min(statistics.fmean(v) for v in per.values())
    losers = sum(1 for r in bk if r["net"] < 0) / len(bk)
    return (
        f"{label:<26}{len(rows):>8,}{len(bk):>7,}{a_mu * BP:>+9.1f}{a_se * BP:>6.1f}"
        f"{n_mu * BP:>+9.1f}{worst * BP:>+10.1f}{losers:>8.3f}{n_ge:>5}/{nd:<3}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="data/eval/edge/rows_sweep.jsonl")
    ap.add_argument("--movers", default="data/eval/edge/movers.jsonl")
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--buys-only", action="store_true", help="restrict to the live BUY gate")
    ap.add_argument("--bayes-cut", type=float, help="write bayes_judge day records for this cut")
    ap.add_argument("--out-dir", default="data/eval/edge")
    ap.add_argument("--days", help="comma-separated subset of days")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    pct: dict[tuple[str, str], float] = {}
    with Path(args.movers).open() as f:
        for line in f:
            m = json.loads(line)
            pct[(m["day"], m["code"])] = abs(m["prev_move"])

    keep_days = set(args.days.split(",")) if args.days else None
    rows = []
    with Path(args.rows).open() as f:
        for line in f:
            r = json.loads(line)
            v = pct.get((r["day"], r["code"]))
            if v is None:
                continue
            if keep_days and r["day"] not in keep_days:
                continue
            r["absmove"] = v
            rows.append(r)
    if args.buys_only:
        rows = [r for r in rows if (r.get("score_raw") or 0.0) > BUY_SCORE]

    # percentile of |prev move| within each day's candidates
    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r)
    for rs in by_day.values():
        rs.sort(key=lambda r: r["absmove"])
        for i, r in enumerate(rs):
            r["mpct"] = (i + 1) / len(rs)

    cap = args.max_positions
    print(
        f"\n{args.label}{' — ' if args.label else ''}{len(rows):,} candidates over {len(by_day)} days, "
        f"book capped at {cap}/day{'  (BUY gate only)' if args.buys_only else ''}\n"
    )
    print(
        f"{'filter':<26}{'cands':>8}{'book':>7}{'alpha':>9}{'se':>6}"
        f"{'net':>9}{'worst day':>10}{'losers':>8}{'d>=0':>9}"
    )
    print(describe("none (baseline)", rows, cap))
    for c in CUTS:
        keep = [r for r in rows if r["mpct"] <= c]
        print(describe(f"drop top {(1 - c) * 100:.0f}% movers", keep, cap))

    print("\n--- what the filter removes, and what replaces it in the book")
    base = {id(r) for r in book(rows, cap)}
    for c in CUTS:
        keep = [r for r in rows if r["mpct"] <= c]
        new = book(keep, cap)
        added = [r for r in new if id(r) not in base]
        dropped_cands = len(rows) - len(keep)
        if not added:
            print(
                f"  drop top {(1 - c) * 100:>2.0f}%:  candidates removed {dropped_cands:>6,}"
                f"   book unchanged"
            )
            continue
        a_mu, _, _, _ = day_mean([(r["day"], r["alpha"]) for r in added])
        n_mu, _, _, _ = day_mean([(r["day"], r["net"]) for r in added])
        print(
            f"  drop top {(1 - c) * 100:>2.0f}%:  candidates removed {dropped_cands:>6,}"
            f"   slots refilled {len(added):>5,}"
            f"   refill alpha {a_mu * BP:>+7.1f}  refill net {n_mu * BP:>+7.1f}"
        )
    if args.bayes_cut:
        keep = [r for r in rows if r["mpct"] <= args.bayes_cut]
        out = Path(args.out_dir)
        for name, bk in (
            ("filter_candidate", book(keep, cap)),
            ("filter_baseline", book(rows, cap)),
        ):
            per: dict[str, list[float]] = defaultdict(list)
            for r in bk:
                per[r["day"]].append(r["net"] * 100_000.0)
            recs = [
                {
                    "day": d,
                    "net": sum(v),
                    "trades": len(v),
                    "losers": sum(1 for x in v if x < 0),
                    "worst_trade": min(v),
                    "errors": 0,
                    "leftovers": 0,
                }
                for d, v in sorted(per.items())
            ]
            (out / f"{name}.json").write_text(json.dumps(recs, indent=1))
        print(f"\nbayes_judge records for cut {args.bayes_cut} -> {out}/filter_*.json")
    print()


if __name__ == "__main__":
    main()
