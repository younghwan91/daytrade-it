"""Spend the pre-move gate through the live capital budget, for `bayes_judge.py`.

The pair-level tables say the gate is worth 17.5 bp per pair. That is not the
question the adoption rule asks. This turns each day into the record shape
`bayes_judge.py` wants -- `{day, net, trades, losers, worst_trade, errors,
leftovers}` -- by spending a fixed notional on at most `--max-positions` BUY
signals per day, which is how the live daemon spends it.

Selection inside a day is the live order: earliest acting minute first, ties
broken by the higher score, which is what the daemon does when a batch of
judgments lands together. That ordering is known to be 35 bp/trade worse than
random, and it is used here anyway, because the candidate and the baseline
must differ ONLY in the gate under test.

This is a budget model, not the replay engine: no fill model, no VWAP exit, no
add-on, no position contention. It is comparable to itself across the two
gates and should not be read against replay KRW figures.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

NOTIONAL = 100_000.0
BUY_SCORE = 0.15


def day_records(rows: list[dict], cap: float, max_pos: int) -> list[dict]:
    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        score = r.get("score_raw")
        pm = r.get("pre_move_raw")
        if score is None or score <= BUY_SCORE or pm is None or pm > cap:
            continue
        by_day[r["day"]].append(r)

    out = []
    for day in sorted(by_day):
        # live order: earliest acting minute first, the higher score breaking ties
        picks = sorted(by_day[day], key=lambda r: (r["act_min"], -r["score_raw"]))[:max_pos]
        pnl = [r["net"] * NOTIONAL for r in picks]
        out.append(
            {
                "day": day,
                "net": sum(pnl),
                "trades": len(pnl),
                "losers": sum(1 for p in pnl if p < 0),
                "worst_trade": min(pnl) if pnl else 0.0,
                "errors": 0,
                "leftovers": 0,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--candidate-cap", type=float, default=0.01)
    ap.add_argument("--baseline-cap", type=float, default=0.03)
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--out-dir", default="data/eval/edge")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines()]
    cand = day_records(rows, args.candidate_cap, args.max_positions)
    base = day_records(rows, args.baseline_cap, args.max_positions)

    days = {r["day"] for r in cand} & {r["day"] for r in base}
    cand = [r for r in cand if r["day"] in days]
    base = [r for r in base if r["day"] in days]

    out = Path(args.out_dir)
    (out / "budget_candidate.json").write_text(json.dumps(cand, indent=1))
    (out / "budget_baseline.json").write_text(json.dumps(base, indent=1))
    print(
        f"{len(days)} common days\n"
        f"  candidate (cap {args.candidate_cap:+.0%}): net {sum(r['net'] for r in cand):+,.0f} "
        f"trades {sum(r['trades'] for r in cand)} losers {sum(r['losers'] for r in cand)}\n"
        f"  baseline  (cap {args.baseline_cap:+.0%}): net {sum(r['net'] for r in base):+,.0f} "
        f"trades {sum(r['trades'] for r in base)} losers {sum(r['losers'] for r in base)}\n"
        f"-> {out}/budget_candidate.json, {out}/budget_baseline.json"
    )


if __name__ == "__main__":
    main()
