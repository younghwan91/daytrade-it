"""A4 — 6개월 뉴스 유니버스 분봉 커버리지 점검.

    uv run python scripts/eval/universe_coverage.py \\
        --start 2026-03-16 --end 2026-09-15 \\
        --out-dir data/eval/universe_coverage   # simnode

날짜 d 의 유니버스 = 전 거래일 거래대금(``daily_bars_adjusted.trade_value``,
close*volume 과 동치인 기존 컬럼) 내림차순 상위 600 종목, 동률은 코드 오름차순
+ 항상 069500(코스피200 ETF). 각 (day, code) 쌍의 커버리지는 그날 정규장
09:00~15:30 ``minute_bars`` 행 수 >= 300 인지로 판정한다(390분 세션의 약 77%
이상 — 상장폐지·거래정지일도 있어 300을 완전 커버리지 기준으로 쓴다).

DB 호출은 두 번뿐이다: (1) 6개월치 ``daily_bars_adjusted`` 를 code/date/
trade_value 만 한 번에 읽어 유니버스를 파이썬에서 계산하고, (2) 월별로 한 번씩
``minute_bars`` 를 code, ts::date 로 그룹핑해 센다(하루 단위·종목 단위 질의는
하지 않는다). 읽기 전용: `SET default_transaction_read_only` 로 세션을 잠그고
쓰기 쿼리는 아예 없다. simnode 전용(``require_backtest_host``).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from collections import defaultdict
from pathlib import Path

import psycopg2

from daytrade_it.application.news.universe import UNIVERSE_SIZE as _UNIVERSE_SIZE
from daytrade_it.application.news.universe import universe_size
from daytrade_it.shared.runtime_host import require_backtest_host

#: One number for live sweep, C3 events and the analysis scripts -- see
#: ``application/news/universe.UNIVERSE_SIZE`` (300; Addendum C).
UNIVERSE_SIZE = _UNIVERSE_SIZE
ALWAYS_INCLUDE = ("069500",)
COVERAGE_THRESHOLD = 300  # >=300 규정장 분봉이면 "커버됨"
SESSION_START = "09:00"
SESSION_END = "15:30"


# --------------------------------------------------------------------------
# Pure functions (unit-testable without a DB) --
# --------------------------------------------------------------------------


def select_universe(
    day_rows: list[tuple[str, float]],
    size: int | None = None,
    always: tuple[str, ...] = ALWAYS_INCLUDE,
) -> list[str]:
    """Top ``size`` codes by trade value (desc), tie-break code asc, plus ``always``.

    ``day_rows`` is ``[(code, trade_value), ...]`` for one (the previous
    trading) day. Deterministic. ``always`` codes are appended even if they
    fall outside the top ``size`` (they never count against the size limit).
    """
    size = universe_size(size)
    ranked = sorted(day_rows, key=lambda r: (-r[1], r[0]))
    top = [code for code, _ in ranked[:size]]
    seen = set(top)
    for code in always:
        if code not in seen:
            top.append(code)
            seen.add(code)
    return top


def compute_universes(
    rows_by_day: dict[dt.date, list[tuple[str, float]]],
    ordered_days: list[dt.date],
    target_days: list[dt.date],
    size: int | None = None,
    always: tuple[str, ...] = ALWAYS_INCLUDE,
) -> dict[dt.date, list[str]]:
    """Universe per target day, keyed off the previous entry in ``ordered_days``.

    ``ordered_days`` must be sorted and contain, for every day in
    ``target_days``, the trading day immediately before it (i.e. it should
    start at least one trading day earlier than ``min(target_days)``).
    Raises ``KeyError`` if a target day has no predecessor in ``ordered_days``
    or its predecessor has no rows.
    """
    index = {day: i for i, day in enumerate(ordered_days)}
    universes: dict[dt.date, list[str]] = {}
    for day in target_days:
        i = index[day]
        if i == 0:
            raise KeyError(f"{day} has no predecessor in ordered_days")
        prev_day = ordered_days[i - 1]
        universes[day] = select_universe(rows_by_day.get(prev_day, []), size=size, always=always)
    return universes


def coverage_rows(
    universes: dict[dt.date, list[str]],
    bar_counts: dict[tuple[dt.date, str], int],
) -> list[tuple[dt.date, str, int]]:
    """Flatten universes into (day, code, bars) rows; missing pairs get bars=0."""
    rows: list[tuple[dt.date, str, int]] = []
    for day in sorted(universes):
        for code in universes[day]:
            rows.append((day, code, bar_counts.get((day, code), 0)))
    return rows


def month_key(day: dt.date) -> str:
    return f"{day.year:04d}-{day.month:02d}"


def summarize(
    rows: list[tuple[dt.date, str, int]],
    threshold: int = COVERAGE_THRESHOLD,
    top_n: int = 20,
) -> dict:
    """Per-month universe/covered/missing counts plus top missing codes."""
    per_month: dict[str, dict] = defaultdict(
        lambda: {"universe_pairs": 0, "covered": 0, "missing": 0}
    )
    missing_by_code: dict[str, int] = defaultdict(int)
    for day, code, bars in rows:
        m = per_month[month_key(day)]
        m["universe_pairs"] += 1
        if bars >= threshold:
            m["covered"] += 1
        else:
            m["missing"] += 1
            missing_by_code[code] += 1

    top_missing = sorted(missing_by_code.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    total_universe = sum(m["universe_pairs"] for m in per_month.values())
    total_missing = sum(m["missing"] for m in per_month.values())
    return {
        "threshold_bars": threshold,
        "total_universe_pairs": total_universe,
        "total_missing_pairs": total_missing,
        "total_covered_pairs": total_universe - total_missing,
        "coverage_pct": round(100.0 * (total_universe - total_missing) / total_universe, 3)
        if total_universe
        else None,
        "by_month": dict(sorted(per_month.items())),
        "top_missing_codes": top_missing,
    }


def missing_pairs(
    rows: list[tuple[dt.date, str, int]], threshold: int = COVERAGE_THRESHOLD
) -> list[tuple[dt.date, str, int]]:
    return [(d, c, b) for d, c, b in rows if b < threshold]


# --------------------------------------------------------------------------
# DB-touching code (simnode only, read-only)
# --------------------------------------------------------------------------


def connect(dsn: str | None = None) -> psycopg2.extensions.connection:
    conn = psycopg2.connect(dsn or os.environ["KR_QUANT_DB"], connect_timeout=5)
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    cur.execute("SET max_parallel_workers_per_gather = 0")
    cur.close()
    return conn


def fetch_trading_days(conn, buffer_start: dt.date, end: dt.date) -> list[dt.date]:
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT date FROM daily_bars_adjusted WHERE date >= %s AND date <= %s "
        "ORDER BY date",
        (buffer_start, end),
    )
    days = [r[0] for r in cur.fetchall()]
    cur.close()
    return days


def fetch_daily_rows(
    conn, buffer_start: dt.date, end: dt.date
) -> dict[dt.date, list[tuple[str, float]]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT code, date, trade_value FROM daily_bars_adjusted WHERE date >= %s AND date <= %s",
        (buffer_start, end),
    )
    rows_by_day: dict[dt.date, list[tuple[str, float]]] = defaultdict(list)
    for code, date, trade_value in cur.fetchall():
        rows_by_day[date].append((code, float(trade_value or 0)))
    cur.close()
    return dict(rows_by_day)


def fetch_month_bar_counts(
    conn, month_start: dt.date, month_end_exclusive: dt.date
) -> dict[tuple[dt.date, str], int]:
    """One grouped query for a whole month: {(day, code): regular-session bar count}."""
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT code, ts::date AS day, count(*) FROM minute_bars
        WHERE ts >= %s AND ts < %s
          AND ts::time >= '{SESSION_START}' AND ts::time <= '{SESSION_END}'
        GROUP BY code, ts::date
        """,
        (month_start, month_end_exclusive),
    )
    counts = {(day, code): n for code, day, n in cur.fetchall()}
    cur.close()
    return counts


def month_bounds(target_days: list[dt.date]) -> list[tuple[dt.date, dt.date]]:
    """Distinct (month_start, next_month_start) pairs covering target_days."""
    months = sorted({(d.year, d.month) for d in target_days})
    bounds = []
    for y, m in months:
        start = dt.date(y, m, 1)
        end = dt.date(y + 1, 1, 1) if m == 12 else dt.date(y, m + 1, 1)
        bounds.append((start, end))
    return bounds


def run(
    start: dt.date,
    end: dt.date,
    out_dir: Path,
    dsn: str | None = None,
    size: int | None = None,
) -> dict:
    conn = connect(dsn)
    try:
        buffer_start = start - dt.timedelta(days=20)
        trading_days = fetch_trading_days(conn, buffer_start, end)
        if not trading_days or trading_days[0] >= start:
            raise RuntimeError(
                f"no trading day before {start} found in daily_bars_adjusted "
                f"(buffer_start={buffer_start}); widen the buffer"
            )
        target_days = [d for d in trading_days if start <= d <= end]
        rows_by_day = fetch_daily_rows(conn, buffer_start, end)
        universes = compute_universes(rows_by_day, trading_days, target_days, size=size)

        bar_counts: dict[tuple[dt.date, str], int] = {}
        for month_start, month_end in month_bounds(target_days):
            bar_counts.update(fetch_month_bar_counts(conn, month_start, month_end))
    finally:
        conn.close()

    rows = coverage_rows(universes, bar_counts)
    summary = summarize(rows)
    missing = missing_pairs(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "coverage.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "code", "bars"])
        for day, code, bars in rows:
            w.writerow([day.isoformat(), code, bars])

    with (out_dir / "missing.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "code", "bars"])
        for day, code, bars in missing:
            w.writerow([day.isoformat(), code, bars])

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main() -> None:
    require_backtest_host()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out-dir", default="data/eval/universe_coverage")
    ap.add_argument("--dsn", default=None, help="생략 시 KR_QUANT_DB")
    ap.add_argument(
        "--universe-size",
        type=int,
        default=None,
        help=f"전일 거래대금 상위 N (기본: DAYTRADE_NEWS_UNIVERSE_SIZE, 없으면 {UNIVERSE_SIZE})",
    )
    args = ap.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    summary = run(start, end, Path(args.out_dir), dsn=args.dsn, size=args.universe_size)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
