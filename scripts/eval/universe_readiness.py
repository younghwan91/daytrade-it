"""A6 prerequisites: which days have the minute bars and the C1 news the universe backtest needs.

    uv run python scripts/eval/universe_readiness.py --start 2026-03-16 --end 2026-09-15 \\
        --c1-dir data/toss_universe --out-dir data/eval/universe_readiness   # simnode

For trading day d (universe = `application/news/universe.select_universe` of the
previous trading day's traded value -- the pool C3 and C5 use):
- minute bars: every universe code plus 069500 has >= `COVERAGE_THRESHOLD`
  regular-session bars (same grouped monthly count as `universe_coverage.py`);
- news: every universe code has a C1 `.done`, and its coverage start (since, or
  the earliest row for a page-capped code) is <= d's window start (previous
  trading day 15:20), exactly the C3 rule (`build_universe_events.covered`).

status: `ready` (everything present), `partial` (some codes lack bars or news
coverage -- C3 drops those pairs), `not_ready` (069500 bars missing or some
universe code has no `.done`, which C3 refuses without --allow-incomplete).
A6 uses only `ready` days (`ready_days.txt`). Read-only DB, simnode only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_universe_events import (  # noqa: E402
    CUTOFF,
    TRUNCATED,
    covered,
    load_coverage,
)
from universe_coverage import (  # noqa: E402
    ALWAYS_INCLUDE,
    COVERAGE_THRESHOLD,
    connect,
    fetch_daily_rows,
    fetch_month_bar_counts,
    fetch_trading_days,
    month_bounds,
)

from daytrade_it.application.news.universe import select_universe  # noqa: E402
from daytrade_it.infrastructure.brokers.base import KST  # noqa: E402


def day_readiness(
    day: dt.date,
    prev_day: dt.date,
    universe: list[str],
    bar_counts: dict[tuple[dt.date, str], int],
    coverage: dict[str, dict],
    threshold: int = COVERAGE_THRESHOLD,
) -> dict:
    lo = dt.datetime.combine(prev_day, CUTOFF, tzinfo=KST)
    bar_codes = [*universe, *(c for c in ALWAYS_INCLUDE if c not in universe)]
    bars_missing = [c for c in bar_codes if bar_counts.get((day, c), 0) < threshold]
    news_missing = [c for c in universe if c not in coverage]
    news_uncovered = [c for c in universe if c in coverage and not covered(c, coverage, lo)]
    if any(c in bars_missing for c in ALWAYS_INCLUDE) or news_missing or not universe:
        status = "not_ready"
    elif bars_missing or news_uncovered:
        status = "partial"
    else:
        status = "ready"
    return {
        "day": str(day),
        "status": status,
        "universe": len(universe),
        "bars_missing": bars_missing,
        "news_missing": news_missing,
        "news_uncovered": news_uncovered,
        "news_truncated": [c for c in universe if coverage.get(c, {}).get("status") in TRUNCATED],
    }


def readiness_report(
    universes: dict[dt.date, list[str]],
    prev_days: dict[dt.date, dt.date],
    bar_counts: dict[tuple[dt.date, str], int],
    coverage: dict[str, dict],
) -> dict:
    days = [
        day_readiness(d, prev_days[d], universes[d], bar_counts, coverage)
        for d in sorted(universes)
    ]
    counts = {"ready": 0, "partial": 0, "not_ready": 0}
    for d in days:
        counts[d["status"]] += 1
    return {
        "counts": counts,
        "ready_days": [d["day"] for d in days if d["status"] == "ready"],
        "days": days,
    }


def main() -> None:
    from daytrade_it.shared.runtime_host import require_backtest_host

    require_backtest_host()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--c1-dir", type=Path, default=Path("data/toss_universe"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/eval/universe_readiness"))
    ap.add_argument("--dsn", default=None)
    args = ap.parse_args()
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)

    conn = connect(args.dsn)
    try:
        buffer_start = start - dt.timedelta(days=20)
        trading = fetch_trading_days(conn, buffer_start, end)
        targets = [d for d in trading if start <= d <= end]
        if not trading or not targets or trading[0] >= targets[0]:
            raise SystemExit(f"no trading day before {start} in daily_bars_adjusted")
        rows_by_day = fetch_daily_rows(conn, buffer_start, end)
        prev_days = {d: trading[trading.index(d) - 1] for d in targets}
        universes = {d: select_universe(dict(rows_by_day.get(prev_days[d], []))) for d in targets}
        bar_counts: dict[tuple[dt.date, str], int] = {}
        for month_start, month_end in month_bounds(targets):
            bar_counts.update(fetch_month_bar_counts(conn, month_start, month_end))
    finally:
        conn.close()

    report = readiness_report(universes, prev_days, bar_counts, load_coverage(args.c1_dir))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "readiness.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (args.out_dir / "ready_days.txt").write_text("".join(f"{d}\n" for d in report["ready_days"]))
    for d in report["days"]:
        print(
            f"{d['day']} {d['status']:9s} universe {d['universe']} bars_missing "
            f"{len(d['bars_missing'])} news_missing {len(d['news_missing'])} "
            f"news_uncovered {len(d['news_uncovered'])} truncated {len(d['news_truncated'])}"
        )
    print(json.dumps(report["counts"]), f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
