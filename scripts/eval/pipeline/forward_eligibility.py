"""How many forward-holdout days are eligible for stage 4 yet.

A day on or after the holdout start is eligible when both hold:
  * the collection ledger records it COMPLETE -- every code of that day's
    universe fetched in a successful run that began after the day's sweep
    window closed (toss_incremental.py);
  * top-300 minute-bar coverage >= 0.95 -- the share of the day's universe with
    >= 300 regular-session bars (the relaxed usable rule used throughout).

Weekday holidays drop out on the second test: they have no bars.

    .venv/bin/python scripts/eval/pipeline/forward_eligibility.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import edge_db  # noqa: E402
from toss_incremental import LEDGER, WORK, universes  # noqa: E402

MIN_BARS = 300
MIN_COVERAGE = 0.95


def complete_days() -> list[str]:
    out = set()
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "ok":
                out |= {d for d, v in r["days"].items() if v["complete"]}
    return sorted(d for d in out if config.is_holdout(d))


def coverage(days: list[str]) -> dict[str, float]:
    if not days:
        return {}
    uni = universes(days[0], days[-1])
    cur = edge_db.connect().cursor()
    cur.execute(
        "SELECT ts::date, code, count(*) FROM minute_bars "
        "WHERE ts::date = ANY(%s) AND ts::time >= '09:00' AND ts::time <= '15:30' "
        "GROUP BY 1, 2",
        ([dt.date.fromisoformat(d) for d in days],),
    )
    bars = {(str(d), c): n for d, c, n in cur.fetchall()}
    return {
        d: sum(bars.get((d, c), 0) >= MIN_BARS for c in uni[d]) / len(uni[d])
        for d in days
        if uni.get(d)
    }


def main() -> None:
    days = complete_days()
    cov = coverage(days)
    rows = [
        {
            "day": d,
            "news_complete": True,
            "bar_coverage": round(cov.get(d, 0.0), 4),
            "eligible": cov.get(d, 0.0) >= MIN_COVERAGE,
        }
        for d in days
    ]
    status = config.stage4_status([r["day"] for r in rows if r["eligible"]])
    for r in rows:
        print(f"{r['day']}  bars {r['bar_coverage']:.3f}  {'ELIGIBLE' if r['eligible'] else '-'}")
    print(json.dumps(status))
    (WORK / "forward_eligibility.json").write_text(
        json.dumps(
            {
                "checked_at": dt.datetime.now().isoformat(timespec="seconds"),
                "days": rows,
                "stage4": status,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
