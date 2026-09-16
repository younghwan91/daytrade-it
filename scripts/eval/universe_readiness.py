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

Relaxed view (A7 controller ruling). No day reaches `ready`: every universe of
600 codes holds some code with fewer than 300 bars (halted, newly listed or
thin), so the strict rule can never select a backtest day. With
`--event-tickers` (a C4 pairs file or a built events directory) each day is
also judged on the codes it actually trades: a day is `usable` when at least
`--event-bar-frac` (0.95) of its *distinct* event tickers have bars, and the
events on the rest are dropped and counted (`dropped_events`). 069500 still
has to be there, since the replay needs the index. Both views are reported
side by side -- `ready_days.txt` and `usable_days.txt`.
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

#: Relaxed rule: share of a day's distinct event tickers that must have bars.
EVENT_BAR_FRAC = 0.95


def load_event_tickers(source: Path) -> dict[dt.date, list[str]]:
    """day -> the ticker of every event (or pair) of that day, in file order.

    `source` is either a C4 pairs file (JSONL rows with `day` and `code`, what
    `build_universe_events.py --pairs-out` writes) or a directory of built
    event files (`YYYY-MM-DD.jsonl` with `ticker`, what it writes by default;
    `meta.json` and anything else that is not a day file is ignored). Rows are
    kept one per event, not deduped, so the caller can count dropped events as
    well as distinct tickers.
    """
    out: dict[dt.date, list[str]] = {}
    if source.is_dir():
        for path in sorted(source.glob("*.jsonl")):
            try:
                day = dt.date.fromisoformat(path.stem)
            except ValueError:
                continue
            rows = out.setdefault(day, [])
            with path.open() as f:
                rows.extend(json.loads(line)["ticker"] for line in f if line.strip())
        return out
    with source.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.setdefault(dt.date.fromisoformat(r["day"]), []).append(r["code"])
    return out


def day_readiness(
    day: dt.date,
    prev_day: dt.date,
    universe: list[str],
    bar_counts: dict[tuple[dt.date, str], int],
    coverage: dict[str, dict],
    threshold: int = COVERAGE_THRESHOLD,
    event_tickers: dict[dt.date, list[str]] | None = None,
    event_bar_frac: float = EVENT_BAR_FRAC,
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
    out = {
        "day": str(day),
        "status": status,
        "universe": len(universe),
        "bars_missing": bars_missing,
        "news_missing": news_missing,
        "news_uncovered": news_uncovered,
        "news_truncated": [c for c in universe if coverage.get(c, {}).get("status") in TRUNCATED],
    }
    if event_tickers is None:
        return out
    rows = event_tickers.get(day, [])
    distinct = sorted(set(rows))
    no_bars = {c for c in distinct if bar_counts.get((day, c), 0) < threshold}
    frac = (len(distinct) - len(no_bars)) / len(distinct) if distinct else 0.0
    dropped = sum(c in no_bars for c in rows)
    out.update(
        {
            "event_tickers": len(distinct),
            "event_rows": len(rows),
            "event_bars_missing": sorted(no_bars),
            "event_bar_frac": frac,
            "event_row_frac": (len(rows) - dropped) / len(rows) if rows else 0.0,
            "dropped_events": dropped,
            "usable": bool(distinct)
            and frac >= event_bar_frac
            and not any(c in bars_missing for c in ALWAYS_INCLUDE)
            and bool(universe),
        }
    )
    return out


def readiness_report(
    universes: dict[dt.date, list[str]],
    prev_days: dict[dt.date, dt.date],
    bar_counts: dict[tuple[dt.date, str], int],
    coverage: dict[str, dict],
    event_tickers: dict[dt.date, list[str]] | None = None,
    event_bar_frac: float = EVENT_BAR_FRAC,
) -> dict:
    days = [
        day_readiness(
            d,
            prev_days[d],
            universes[d],
            bar_counts,
            coverage,
            event_tickers=event_tickers,
            event_bar_frac=event_bar_frac,
        )
        for d in sorted(universes)
    ]
    counts = {"ready": 0, "partial": 0, "not_ready": 0}
    for d in days:
        counts[d["status"]] += 1
    report = {
        "counts": counts,
        "ready_days": [d["day"] for d in days if d["status"] == "ready"],
        "days": days,
    }
    if event_tickers is not None:
        counts["usable"] = sum(d["usable"] for d in days)
        report["usable_days"] = [d["day"] for d in days if d["usable"]]
        report["dropped_events"] = sum(d["dropped_events"] for d in days)
        report["event_bar_frac_rule"] = event_bar_frac
    return report


def main() -> None:
    from daytrade_it.shared.runtime_host import require_backtest_host

    require_backtest_host()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--c1-dir", type=Path, default=Path("data/toss_universe"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/eval/universe_readiness"))
    ap.add_argument("--dsn", default=None)
    ap.add_argument(
        "--event-tickers",
        type=Path,
        default=None,
        help="C4 pairs file or built events directory; adds the relaxed `usable` view "
        "(>= --event-bar-frac of a day's distinct event tickers have bars)",
    )
    ap.add_argument("--event-bar-frac", type=float, default=EVENT_BAR_FRAC)
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

    events = load_event_tickers(args.event_tickers) if args.event_tickers else None
    report = readiness_report(
        universes,
        prev_days,
        bar_counts,
        load_coverage(args.c1_dir),
        event_tickers=events,
        event_bar_frac=args.event_bar_frac,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "readiness.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (args.out_dir / "ready_days.txt").write_text("".join(f"{d}\n" for d in report["ready_days"]))
    if events is not None:
        (args.out_dir / "usable_days.txt").write_text(
            "".join(f"{d}\n" for d in report["usable_days"])
        )
    for d in report["days"]:
        line = (
            f"{d['day']} {d['status']:9s} universe {d['universe']} bars_missing "
            f"{len(d['bars_missing'])} news_missing {len(d['news_missing'])} "
            f"news_uncovered {len(d['news_uncovered'])} truncated {len(d['news_truncated'])}"
        )
        if events is not None:
            line += (
                f" | events {d['event_rows']} tickers {d['event_tickers']} "
                f"event_bar_frac {d['event_bar_frac']:.3f} dropped {d['dropped_events']} "
                f"{'USABLE' if d['usable'] else '-'}"
            )
        print(line)
    print(json.dumps(report["counts"]), f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
