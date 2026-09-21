"""What would it cost to buy each additional year of judged regime coverage?

The binding limit on Track B is no longer prices -- hourly bars reach
2023-09-19 -- it is that LLM judgments exist for 125 days only. So the question
is concrete: pairs per year, times the prefilter keep rate, times the measured
unit cost.

Unit cost is MEASURED, not quoted: 1,127 in / 68 out tokens per pair =
$0.000734/pair, confirmed by the 136,574-pair / $100.25 estimate the A7-prep
run produced.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path.home() / "git" / "daytrade-it-rlnews"

#: Measured per-pair judging cost (1,127 in / 68 out tokens).
UNIT = 0.000734
#: The adopted prefilter kept 279,354 of 331,308 built pairs on the judged pool.
KEEP = 279354 / 331308

master = ROOT / "data/eval/rlnews/master_daily_full.jsonl"
by_year: collections.Counter = collections.Counter()
days_by_year: dict[str, set] = collections.defaultdict(set)
with master.open() as fh:
    for line in fh:
        r = json.loads(line)
        y = r["day"][:4]
        by_year[y] += 1
        days_by_year[y].add(r["day"])

judged_days = {
    json.loads(line)["day"] for line in (ROOT / "data/eval/prompt_v3/labels.jsonl").open()
}

print(f"unit cost ${UNIT}/pair, prefilter keeps {KEEP:.1%}\n")
print(
    f"{'year':6s} {'days':>5s} {'judged':>7s} {'unjudged':>9s} {'pairs':>9s} "
    f"{'pairs/day':>10s} {'kept':>9s} {'USD':>8s} {'USD/day':>8s}"
)
tot_pairs = tot_usd = 0.0
for y in sorted(by_year):
    days = sorted(days_by_year[y])
    nj = len([d for d in days if d in judged_days])
    un = len(days) - nj
    pairs = by_year[y]
    # cost to judge only the days that are NOT already judged
    unj_pairs = sum(1 for _ in ()) or pairs * (un / len(days))
    kept = unj_pairs * KEEP
    usd = kept * UNIT
    tot_pairs += unj_pairs
    tot_usd += usd
    print(
        f"{y:6s} {len(days):>5d} {nj:>7d} {un:>9d} {pairs:>9,d} "
        f"{pairs / len(days):>10.0f} {kept:>9,.0f} {usd:>8.2f} {usd / max(un, 1):>8.2f}"
    )
print(f"\nALL unjudged days in 2023-03..2026-09: {tot_pairs:,.0f} kept pairs, ${tot_usd:,.2f}")
print("NOTE: only days from 2023-09-19 have hourly prices; earlier days are")
print("      unusable at intraday resolution however much judging is bought.")
usable = [y for y in sorted(by_year) if y >= "2023"]
