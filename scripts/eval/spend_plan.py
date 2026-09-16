#!/usr/bin/env python3
"""Cross `universe_readiness`'s usable days with the per-day judging cost
summary, producing a recommended spend plan: usable days ordered
cheapest-first (not newest-first, unlike `judge_days.py`'s contiguous
newest-first window), with cumulative cost at each budget threshold.

`judge_days.py` spends newest-first and has no idea which days are
minute-bar usable (A8 concern #6) -- a $10 wave can land entirely in a
not-yet-backfilled window and buy judged days the replay cannot use. This
script is the missing cross: usable days only, cheapest first, so a top-up
buys the most usable whole days per dollar.

Usage (simnode, a-plan worktree):
    uv run python scripts/eval/spend_plan.py \
        --usable-days data/eval/universe_readiness_a8/usable_days.txt \
        --summary data/eval/universe_pairs_n300_summary.csv \
        --out data/eval/spend_plan_a8.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_plan(usable: set[str], rows: list[dict[str, str]], budgets: list[float]) -> dict:
    """Usable days ordered cheapest-first, with cumulative-cost thresholds."""
    usable_rows = [r for r in rows if r["day"] in usable]
    usable_rows.sort(key=lambda r: float(r["est_cost_usd"]))

    plan = []
    cum = 0.0
    for r in usable_rows:
        cum += float(r["est_cost_usd"])
        plan.append(
            {
                "day": r["day"],
                "est_cost_usd": round(float(r["est_cost_usd"]), 6),
                "missing": int(r["missing"]),
                "cumulative_usd": round(cum, 6),
            }
        )

    thresholds = {}
    for budget in sorted(budgets):
        days = [p for p in plan if p["cumulative_usd"] <= budget]
        thresholds[str(budget)] = {
            "n_days": len(days),
            "actual_cost": round(days[-1]["cumulative_usd"], 2) if days else 0.0,
            "oldest_day": min((d["day"] for d in days), default=None),
            "newest_day": max((d["day"] for d in days), default=None),
            "days": [d["day"] for d in days],
        }

    return {
        "n_usable_days_total": len(usable_rows),
        "n_days_total_in_summary": len(rows),
        "thresholds": thresholds,
        "plan_cheapest_first": plan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usable-days", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budgets", type=float, nargs="+", default=[10, 30, 60, 112])
    args = ap.parse_args()

    usable = {line.strip() for line in args.usable_days.read_text().splitlines() if line.strip()}
    with args.summary.open() as f:
        rows = list(csv.DictReader(f))

    out = build_plan(usable, rows, args.budgets)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for b, t in out["thresholds"].items():
        print(
            f"${b}: {t['n_days']} usable day(s), ${t['actual_cost']}, {t['oldest_day']}..{t['newest_day']}"
        )


if __name__ == "__main__":
    main()
