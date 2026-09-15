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
- decide_at = the article's discovery time in `simulate_sweep` (the live sweep
  runner's exact request order: universe order, paging, one detail call per new
  article, 1 request/s from 08:00 until 15:20) + judge_lag -- the same for every
  pair of the article (live sweep judges all pairs at discovery),
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

from judge_batch import cache_row_valid

from daytrade_it.application.news.universe import (
    live_article_id,
    related_codes,
    select_universe,
    simulate_sweep,
)
from daytrade_it.application.replay.events import ReplayEvent, dump_events
from daytrade_it.infrastructure.brokers.base import KST
from daytrade_it.infrastructure.external.anthropic_client import DEFAULT_MODEL

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


#: C1 `.done` statuses whose file stops before `since` (Toss's 100-page cap or
#: the collector's `--max-pages`): coverage starts at the earliest row instead.
TRUNCATED = frozenset({"page_cap", "max_pages"})


class IncompleteDay(RuntimeError):
    """A day's universe has codes the C1 backfill hasn't finished (no `.done`)."""


def load_c1(root: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """C1 files of finished codes -> (articles, coverage).

    Only `{code}.jsonl` with a `{code}.done` next to it is read: a code still
    being backfilled (or failed) has no `.done` and its partial rows would
    silently under-count related codes. articles: news id -> {news_id, title,
    summary, category, published_at, codes}. coverage: code -> {status, start}
    where start is `since` 00:00 KST for a complete code and the earliest
    createdAt in the file for a truncated one (S12).
    """
    arts: dict[str, dict] = {}
    codes: dict[str, set[str]] = {}
    coverage: dict[str, dict] = {}
    for done in sorted(root.glob("*.done")):
        code = done.stem
        info = json.loads(done.read_text())
        earliest: dt.datetime | None = None
        path = root / f"{code}.jsonl"
        if path.exists():
            with path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    news_id = str(r["id"])
                    created = _parse_created(r["createdAt"])
                    earliest = created if earliest is None else min(earliest, created)
                    codes.setdefault(news_id, set()).add(code)
                    if news_id not in arts:
                        arts[news_id] = {
                            "news_id": news_id,
                            "title": r.get("title") or "",
                            "summary": r.get("summary") or "",
                            "category": r.get("category"),
                            "published_at": created,
                        }
        coverage[code] = _coverage_entry(info, earliest)
    for news_id, related in related_codes(codes).items():
        arts[news_id]["codes"] = related
    return arts, coverage


def _coverage_entry(info: dict, earliest: dt.datetime | None) -> dict:
    status = info.get("status")
    start = dt.datetime.combine(dt.date.fromisoformat(info["since"]), dt.time(0), tzinfo=KST)
    if status in TRUNCATED and earliest is not None:
        start = max(start, earliest)
    return {"status": status, "start": start}


def load_coverage(root: Path) -> dict[str, dict]:
    """`load_c1`'s coverage without loading articles (reads rows of truncated codes only)."""
    coverage: dict[str, dict] = {}
    for done in sorted(root.glob("*.done")):
        code = done.stem
        info = json.loads(done.read_text())
        earliest: dt.datetime | None = None
        path = root / f"{code}.jsonl"
        if info.get("status") in TRUNCATED and path.exists():
            with path.open() as f:
                for line in f:
                    if line.strip():
                        created = _parse_created(json.loads(line)["createdAt"])
                        earliest = created if earliest is None else min(earliest, created)
        coverage[code] = _coverage_entry(info, earliest)
    return coverage


def load_articles(root: Path) -> dict[str, dict]:
    """Articles only (see `load_c1`)."""
    return load_c1(root)[0]


def covered(code: str, coverage: dict[str, dict] | None, window_start: dt.datetime) -> bool:
    """Whether the C1 file of `code` holds every article of a window starting at `window_start`."""
    if coverage is None:
        return True
    info = coverage.get(code)
    return info is not None and info["start"] <= window_start


def check_day_coverage(
    universe: list[str],
    coverage: dict[str, dict],
    day: dt.date,
    prev_day: dt.date,
    allow_incomplete: bool = False,
) -> dict:
    """Per-day C1 readiness: missing (no `.done`), truncated, and uncovered universe codes.

    Raises `IncompleteDay` listing the missing codes unless `allow_incomplete`.
    """
    lo = dt.datetime.combine(prev_day, CUTOFF, tzinfo=KST)
    missing = [c for c in universe if c not in coverage]
    report = {
        "universe": len(universe),
        "missing": missing,
        "truncated": [c for c in universe if c in coverage and coverage[c]["status"] in TRUNCATED],
        "uncovered": [c for c in universe if not covered(c, coverage, lo)],
    }
    if missing and not allow_incomplete:
        shown = ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else "")
        raise IncompleteDay(
            f"{day}: {len(missing)} universe code(s) without C1 .done: {shown} "
            "(finish the backfill or pass --allow-incomplete)"
        )
    return report


def bucket_articles(
    articles: dict[str, dict], trading_days: list[dt.date]
) -> dict[dt.date, list[dict]]:
    """Assign each article once to the trading day whose window [prev 15:20, d 15:20) holds it.

    Lets the per-day loop look only at its own articles instead of rescanning
    every article for every day; `day_pairs` still applies the window filter.
    """
    days = sorted(trading_days)
    cutoffs = [dt.datetime.combine(d, CUTOFF, tzinfo=KST) for d in days]
    out: dict[dt.date, list[dict]] = {}
    for a in articles.values():
        i = bisect.bisect_right(cutoffs, a["published_at"])
        if i < len(days):
            out.setdefault(days[i], []).append(a)
    return out


def day_pairs(
    articles: dict[str, dict] | Iterable[dict],
    day: dt.date,
    prev_day: dt.date,
    universe: list[str],
    coverage: dict[str, dict] | None = None,
    stats: dict | None = None,
    stocks: dict | None = None,
) -> list[dict]:
    """(article, code) pairs the sweep would judge on `day`, with the discovery time.

    S12: a code whose C1 coverage starts after the day's window start
    (`covered`) is dropped for that day -- both its pairs and its use as a
    related code for discovery -- and counted in `stats["excluded_pairs"]`.
    `coverage=None` treats every code as covered.

    Pairs need the code in the stock master (`stocks`, like the live
    `tradable_tickers`); such codes are still swept (they cost requests and can
    discover an article first). Dropped pairs: `stats["not_in_master"]`.

    Live-sweep parity (C5, F3): discovery times come from `simulate_sweep` --
    the live runner's request order (code order, paging, one detail call per
    new article, 1 request/s from 08:00, none from 15:20) over each covered
    universe code's article list. An article is judged for all of its pairs
    (related codes ∩ universe) when its detail call completes, so every pair
    shares `discovered_at`. Articles the sweep can't reach before 15:20 are
    dropped.
    """
    lo = dt.datetime.combine(prev_day, CUTOFF, tzinfo=KST)
    hi = dt.datetime.combine(day, CUTOFF, tzinfo=KST)
    open_ = dt.datetime.combine(day, OPEN, tzinfo=KST)
    members = set(universe)
    excluded = 0
    candidates: list[tuple[dict, list[str]]] = []
    arrivals: dict[str, list[tuple[str, dt.datetime]]] = {}
    for a in articles.values() if isinstance(articles, dict) else articles:
        pub = a["published_at"]
        if not lo <= pub < hi:
            continue
        in_universe = [c for c in a["codes"] if c in members]
        codes = [c for c in in_universe if covered(c, coverage, lo)]
        excluded += len(in_universe) - len(codes)
        if not codes:
            continue
        candidates.append((a, codes))
        for code in codes:
            arrivals.setdefault(code, []).append((a["news_id"], pub))
    found = simulate_sweep(universe, day, arrivals, stale_before=lo, start=SWEEP_START, end=CUTOFF)
    out = []
    not_in_master = 0
    for a, codes in candidates:
        discovered = found.get(a["news_id"])
        if discovered is None:
            continue
        aid = live_article_id(a["news_id"])
        for code in codes:
            if stocks is not None and code not in stocks:
                not_in_master += 1
                continue
            out.append(
                {
                    **{
                        k: a[k] for k in ("news_id", "title", "summary", "category", "published_at")
                    },
                    "key": f"{aid}|{code}",
                    "article_id": aid,
                    "code": code,
                    "day": day,
                    "discovered_at": discovered,
                    "session": "pre_market" if a["published_at"] < open_ else "intraday",
                }
            )
    out.sort(key=lambda p: (p["discovered_at"], p["key"]))
    if stats is not None:
        stats["excluded_pairs"] = stats.get("excluded_pairs", 0) + excluded
        stats["not_in_master"] = stats.get("not_in_master", 0) + not_in_master
    return out


def build_day_events(
    pairs: list[dict],
    stocks: dict,
    cache: dict[str, list[dict]],
    judge_lag: dt.timedelta,
    model: str = DEFAULT_MODEL,
    stats: dict | None = None,
) -> tuple[list[ReplayEvent], int]:
    """Events for pairs with a cached judgment of their current input (`cache_row_valid`).

    Pairs without a valid row are excluded (counted); `stats["stale_cache"]`
    counts those that had only rows for a different input.
    """
    events = []
    excluded = stale = 0
    rows = {r["key"]: r for r in pair_rows(pairs, stocks)}
    for p in pairs:
        candidates = cache.get(p["key"], [])
        valid = next((r for r in candidates if cache_row_valid(r, rows[p["key"]], model)), None)
        if valid is None:
            excluded += 1
            stale += bool(candidates)
            continue
        features = valid["v2"]
        s = stocks.get(p["code"])
        decide = p["discovered_at"] + judge_lag
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
    if stats is not None:
        stats["stale_cache"] = stats.get("stale_cache", 0) + stale
    return events, excluded


def pair_rows(pairs: list[dict], stocks: dict) -> list[dict]:
    """C4 input rows (contract fields + month/session/day/published_at for stratifying).

    body = summary, else title: the live mapper judges summary, else
    content[:500], else title, and backfilled rows carry no content (F5).
    `summary` keeps the raw C1 summary (legacy cache check, empty counts).
    """
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
                "body": p["summary"] or p["title"],
                "summary": p["summary"],
                "category": p["category"],
                "day": str(p["day"]),
                "month": str(p["day"])[:7],
                "session": p["session"],
                "published_at": p["published_at"].isoformat(),
            }
        )
    return rows


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


def load_cache(path: Path) -> dict[str, list[dict]]:
    """Pair key -> every cache row for it (re-judged inputs append new rows)."""
    if not path.exists():
        return {}
    out: dict[str, list[dict]] = {}
    for line in path.open():
        if line.strip():
            r = json.loads(line)
            out.setdefault(r["key"], []).append(r)
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


def dashboard_keys(day: dt.date, archive: Path = ARCHIVE) -> set[str] | None:
    """Pair keys the live dashboard daemon judged on `day`, or None if no archive file."""
    path = archive / f"{day}.jsonl"
    if not path.exists():
        return None
    keys = set()
    for line in path.open():
        if line.strip():
            try:
                keys.add(json.loads(line)["key"])
            except (ValueError, KeyError):
                continue
    return keys


def prepare_day(
    articles: Iterable[dict],
    day: dt.date,
    prev_day: dt.date,
    universe: list[str],
    coverage: dict[str, dict],
    allow_incomplete: bool,
    archive: Path = ARCHIVE,
    stocks: dict | None = None,
) -> tuple[list[dict], str]:
    """Coverage-checked pairs for one day plus its summary line (raises `IncompleteDay`)."""
    report = check_day_coverage(universe, coverage, day, prev_day, allow_incomplete)
    stats: dict = {}
    pairs = day_pairs(articles, day, prev_day, universe, coverage, stats, stocks)
    dash = dashboard_keys(day, archive)
    overlap = "n/a" if dash is None else str(sum(p["key"] in dash for p in pairs))
    empty = sum(not p["summary"] for p in pairs)
    line = (
        f"{day}: universe {report['universe']} truncated {len(report['truncated'])} "
        f"missing {len(report['missing'])} uncovered {len(report['uncovered'])} "
        f"pairs {len(pairs)} excluded_pairs {stats.get('excluded_pairs', 0)} "
        f"not_in_master {stats.get('not_in_master', 0)} "
        f"empty_summary {empty} dashboard_overlap {overlap}"
    )
    return pairs, line


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
    ap.add_argument("--model", default=DEFAULT_MODEL, help="judge model (cache input hash)")
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="proceed on days whose universe has codes without a C1 .done (counts are printed)",
    )
    args = ap.parse_args()

    trading, daily_values, stocks = _db_sources()
    days = parse_days(args.days, trading)
    articles, coverage = load_c1(args.c1_dir)
    buckets = bucket_articles(articles, trading)
    print(
        f"{len(articles)} articles from {len(coverage)} finished codes in {args.c1_dir}; "
        f"{len(days)} trading days"
    )

    def plans():
        for day in days:
            i = bisect.bisect_left(trading, day)
            if i == 0:
                print(f"{day}: no previous trading day in bars -- skipped")
                continue
            universe = select_universe(daily_values(trading[i - 1]))
            try:
                pairs, line = prepare_day(
                    buckets.get(day, []),
                    day,
                    trading[i - 1],
                    universe,
                    coverage,
                    args.allow_incomplete,
                    stocks=stocks,
                )
            except IncompleteDay as exc:
                sys.exit(f"REFUSED: {exc}")
            print(line)
            yield day, pairs

    if args.pairs_out:
        seen: set[str] = set()
        args.pairs_out.parent.mkdir(parents=True, exist_ok=True)
        with args.pairs_out.open("w") as f:
            for _day, pairs in plans():
                rows = [r for r in pair_rows(pairs, stocks) if r["key"] not in seen]
                seen.update(r["key"] for r in rows)
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
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
    done_days = []
    for day, pairs in plans():
        stats: dict = {}
        events, excluded = build_day_events(pairs, stocks, cache, lag, args.model, stats)
        dump_events(args.out / f"{day}.jsonl", events)
        done_days.append(day)
        print(
            day,
            "universe events",
            len(events),
            "excluded (no cached judgment of the current input)",
            excluded,
            "of which stale cache rows",
            stats["stale_cache"],
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "meta.json").write_text(
        json.dumps(
            {"judge_lag_s": lag.total_seconds(), "days": [str(d) for d in done_days]}, indent=1
        )
    )


if __name__ == "__main__":
    main()
