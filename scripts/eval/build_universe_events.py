"""Universe-pool replay events (A-plan contract C3).

    # events for trading days (DB read-only; run on simnode)
    uv run python scripts/eval/build_universe_events.py --days 2026-03-16..2026-09-15
    # pairs file for judge_batch.py (C4) over the same range
    uv run python scripts/eval/build_universe_events.py --days 2026-03-16..2026-09-15 \\
        --pairs-out data/eval/universe_pairs.jsonl

For trading day d:
- universe = `select_universe(previous trading day's traded value)` (kr_quant
  `daily_bars_adjusted.trade_value`, read-only);
- pairs = (article, code) with createdAt in [prev 15:20, d 15:20) and code in
  (related codes from the C1 backfill files) ∩ universe;
- decide_at = next_visit(code, universe, d, max(createdAt, d 08:00)) + judge_lag,
  batch_at = decide_at + 1s; features = judgment cache v2 (missing -> excluded,
  count printed).

judge_lag: `--judge-lag-s` if given; otherwise measured from the live archive
(`data/news_judgments` + daemon log): per poll, first judged_at minus the
"Toss news poll complete" time -- one article's detect-to-judgment latency
without queueing (the sweep finds articles one code at a time). If no archive
is available the fallback is 60 s (noted in output).

Category: the per-code Toss feed has no dashboard category, so events and
pairs carry the C1 row's `category` if present, else None (the prompt renders
an empty 분류). The live sweep runner (C5) must do the same.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import statistics
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from daytrade_it.application.news.universe import (
    live_article_id,
    next_visit,
    related_codes,
    select_universe,
)
from daytrade_it.application.replay.events import ReplayEvent, dump_events
from daytrade_it.infrastructure.brokers.base import KST

C1_DIR = Path("data/toss_universe")
CACHE = Path("data/eval/cache/universe_v2.jsonl")
OUT = Path("data/eval/replay_events_universe")
ARCHIVE = Path("data/news_judgments")
DAEMON_LOG = Path("data/news_signal_daemon.log")

SWEEP_START = dt.time(8, 0)
CUTOFF = dt.time(15, 20)
OPEN = dt.time(9, 0)
FALLBACK_JUDGE_LAG = dt.timedelta(seconds=60)


def _parse_created(value: str) -> dt.datetime:
    ts = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return ts.replace(tzinfo=KST) if ts.tzinfo is None else ts.astimezone(KST)


def load_articles(root: Path) -> dict[str, dict]:
    """C1 files -> article id -> {title, summary, category, published_at, codes}."""
    arts: dict[str, dict] = {}
    codes: dict[str, set[str]] = {}
    for path in sorted(root.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                news_id = str(r["id"])
                codes.setdefault(news_id, set()).add(r.get("code") or path.stem)
                if news_id not in arts:
                    arts[news_id] = {
                        "news_id": news_id,
                        "title": r.get("title") or "",
                        "summary": r.get("summary") or "",
                        "category": r.get("category"),
                        "published_at": _parse_created(r["createdAt"]),
                    }
    for news_id, related in related_codes(codes).items():
        arts[news_id]["codes"] = related
    return arts


def day_pairs(
    articles: dict[str, dict], day: dt.date, prev_day: dt.date, universe: list[str]
) -> list[dict]:
    """(article, code) pairs the sweep would discover on `day`, with the visit time."""
    lo = dt.datetime.combine(prev_day, CUTOFF, tzinfo=KST)
    hi = dt.datetime.combine(day, CUTOFF, tzinfo=KST)
    start = dt.datetime.combine(day, SWEEP_START, tzinfo=KST)
    open_ = dt.datetime.combine(day, OPEN, tzinfo=KST)
    members = set(universe)
    out = []
    for a in articles.values():
        pub = a["published_at"]
        if not lo <= pub < hi:
            continue
        for code in a["codes"]:
            if code not in members:
                continue
            visit = next_visit(code, universe, day, max(pub, start))
            if visit is None:
                continue
            aid = live_article_id(a["news_id"])
            out.append(
                {
                    **{
                        k: a[k] for k in ("news_id", "title", "summary", "category", "published_at")
                    },
                    "key": f"{aid}|{code}",
                    "article_id": aid,
                    "code": code,
                    "day": day,
                    "visit_at": visit,
                    "session": "pre_market" if pub < open_ else "intraday",
                }
            )
    out.sort(key=lambda p: (p["visit_at"], p["key"]))
    return out


def build_day_events(
    pairs: list[dict], stocks: dict, cache: dict[str, dict], judge_lag: dt.timedelta
) -> tuple[list[ReplayEvent], int]:
    events = []
    excluded = 0
    for p in pairs:
        features = cache.get(p["key"])
        if features is None:
            excluded += 1
            continue
        s = stocks.get(p["code"])
        decide = p["visit_at"] + judge_lag
        events.append(
            ReplayEvent(
                key=p["key"],
                article_id=p["article_id"],
                ticker=p["code"],
                name=s.name if s else p["code"],
                market=s.market if s else None,
                sector=s.sector if s else None,
                title=p["title"],
                category=p["category"],
                published_at=p["published_at"],
                decide_at=decide,
                batch_at=decide + dt.timedelta(seconds=1),
                features=features,
                source="universe",
            )
        )
    events.sort(key=lambda e: (e.decide_at, e.key))
    return events, excluded


def pair_rows(pairs: list[dict], stocks: dict) -> list[dict]:
    """C4 input rows (contract fields + month/session/day/published_at for stratifying)."""
    rows = []
    for p in pairs:
        s = stocks.get(p["code"])
        rows.append(
            {
                "key": p["key"],
                "code": p["code"],
                "name": s.name if s else p["code"],
                "market": s.market if s else None,
                "sector": s.sector if s else None,
                "title": p["title"],
                "body": p["summary"],
                "category": p["category"],
                "day": str(p["day"]),
                "month": str(p["day"])[:7],
                "session": p["session"],
                "published_at": p["published_at"].isoformat(),
            }
        )
    return rows


def run_day(
    articles: dict[str, dict],
    day: dt.date,
    prev_day: dt.date,
    daily_values: Callable[[dt.date], dict[str, float]],
    stocks: dict,
    cache: dict[str, dict],
    judge_lag: dt.timedelta,
    out_dir: Path,
) -> tuple[int, int]:
    universe = select_universe(daily_values(prev_day))
    events, excluded = build_day_events(
        day_pairs(articles, day, prev_day, universe), stocks, cache, judge_lag
    )
    dump_events(out_dir / f"{day}.jsonl", events)
    return len(events), excluded


def measure_judge_lag(rows: Iterable[dict], polls: list[dt.datetime]) -> dt.timedelta:
    """Median over polls of (first judged_at after the poll) - poll time."""
    polls = sorted(polls)
    first: dict[int, dt.datetime] = {}
    for r in rows:
        ts = dt.datetime.fromisoformat(r["judged_at"])
        i = bisect.bisect_right(polls, ts) - 1
        if i >= 0 and (i not in first or ts < first[i]):
            first[i] = ts
    lags = [(ts - polls[i]).total_seconds() for i, ts in first.items()]
    if not lags:
        raise ValueError("no judgments after any poll")
    return dt.timedelta(seconds=statistics.median(lags))


def parse_days(spec: str, trading_days: list[dt.date]) -> list[dt.date]:
    """`A..B` (inclusive) or comma list, restricted to trading days."""
    valid = set(trading_days)
    if ".." in spec:
        a, b = (dt.date.fromisoformat(x) for x in spec.split(".."))
        return sorted(d for d in valid if a <= d <= b)
    return sorted(d for x in spec.split(",") if x and (d := dt.date.fromisoformat(x)) in valid)


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.open():
        if line.strip():
            r = json.loads(line)
            out[r["key"]] = r["v2"]
    return out


def archive_judge_lag() -> dt.timedelta | None:
    from build_replay_events import poll_times  # same log parsing as stage-0 events

    rows = []
    polls: list[dt.datetime] = []
    if not DAEMON_LOG.exists():
        return None
    for path in sorted(ARCHIVE.glob("*.jsonl")):
        rows += [json.loads(line) for line in path.open() if line.strip()]
        polls += poll_times(dt.date.fromisoformat(path.stem))
    try:
        return measure_judge_lag(rows, polls)
    except ValueError:
        return None


def _db_sources() -> tuple[list[dt.date], Callable[[dt.date], dict[str, float]], dict]:
    from daytrade_it.domain.value_objects import Ticker
    from daytrade_it.infrastructure.external.quant_airflow_db_client import QuantAirflowDBClient

    db = QuantAirflowDBClient()
    bars = db.get_adjusted_daily_bars(Ticker(value="005930"))
    trading = [b.date for b in bars]

    def daily_values(day: dt.date) -> dict[str, float]:
        rows = db._query(
            "SELECT code, trade_value FROM daily_bars_adjusted WHERE date = %s", (day,)
        )
        return {code: float(v) for code, v in rows if v is not None}

    return trading, daily_values, db.get_stock_master()


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--days", required=True, help="A..B (inclusive) or comma list; trading days only"
    )
    ap.add_argument("--c1-dir", type=Path, default=C1_DIR)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--judge-lag-s", type=float, default=None)
    ap.add_argument(
        "--pairs-out", type=Path, default=None, help="write C4 pairs file instead of events"
    )
    args = ap.parse_args()

    trading, daily_values, stocks = _db_sources()
    days = parse_days(args.days, trading)
    articles = load_articles(args.c1_dir)
    print(f"{len(articles)} articles from {args.c1_dir}; {len(days)} trading days")

    if args.pairs_out:
        seen: set[str] = set()
        args.pairs_out.parent.mkdir(parents=True, exist_ok=True)
        with args.pairs_out.open("w") as f:
            for day in days:
                i = bisect.bisect_left(trading, day)
                if i == 0:
                    print(f"{day}: no previous trading day in bars -- skipped")
                    continue
                universe = select_universe(daily_values(trading[i - 1]))
                rows = [
                    r
                    for r in pair_rows(day_pairs(articles, day, trading[i - 1], universe), stocks)
                    if r["key"] not in seen
                ]
                seen.update(r["key"] for r in rows)
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(day, "pairs", len(rows))
        print(f"wrote {len(seen)} pairs -> {args.pairs_out}")
        return

    if args.judge_lag_s is not None:
        lag = dt.timedelta(seconds=args.judge_lag_s)
    elif (measured := archive_judge_lag()) is not None:
        lag = measured
        print(f"judge_lag measured from live archive: {lag.total_seconds():.2f}s")
    else:
        lag = FALLBACK_JUDGE_LAG
        print(
            f"NOTE: no live archive to measure judge_lag -- using fallback {lag.total_seconds():.0f}s"
        )

    cache = load_cache(args.cache)
    for day in days:
        i = bisect.bisect_left(trading, day)
        if i == 0:
            print(f"{day}: no previous trading day in bars -- skipped")
            continue
        n, excluded = run_day(
            articles, day, trading[i - 1], daily_values, stocks, cache, lag, args.out
        )
        print(day, "universe events", n, "excluded (no cached judgment)", excluded)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "meta.json").write_text(
        json.dumps({"judge_lag_s": lag.total_seconds(), "days": [str(d) for d in days]}, indent=1)
    )


if __name__ == "__main__":
    main()
