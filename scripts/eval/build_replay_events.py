"""Build replay events per trading day (spec 2026-09-15, data option B).

    uv run python scripts/eval/build_replay_events.py --kr-quant-days 2026-09-07,2026-09-08,2026-09-09,2026-09-10,2026-09-11

- Days with a daemon archive (`data/news_judgments/DAY.jsonl`): archive only.
  Poll times come from `data/news_signal_daemon.log` ("Toss news poll complete").
- `--kr-quant-days`: kr_quant `news_articles` pairs, keyed like the daemon
  (uuid5(TOSS_NEWS_NAMESPACE, id)), judged with the live `score_article`
  (cached in `data/eval/cache/replay_v2.jsonl`), timed with the archive's
  median detect delay on the daemon's poll grid. A day that already has an
  archive file is never overwritten by kr_quant-sourced events -- it is
  skipped with a warning (archive is the ground truth for that day).
- `--check-days`: judge kr_quant pairs for these days into the cache ONLY
  (no event files written) -- used for the reproducibility check against
  archive judgments (see task-5-brief.md Step 3).
Run on trader, then copy `data/eval/replay_events/` to simnode.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import statistics
from dataclasses import asdict
from pathlib import Path
from uuid import uuid5

import psycopg2

from daytrade_it.application.replay.events import (
    ReplayEvent,
    dump_events,
    events_from_archive,
    median_detect_delay,
    schedule_detected,
)
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.brokers.base import KST
from daytrade_it.infrastructure.external.anthropic_client import AnthropicClient
from daytrade_it.infrastructure.external.quant_airflow_db_client import QuantAirflowDBClient
from daytrade_it.infrastructure.external.toss_news_client import TOSS_NEWS_NAMESPACE
from daytrade_it.shared.config import get_settings
from daytrade_it.shared.errors.exceptions import ExternalAPIError

ARCHIVE = Path("data/news_judgments")
DAEMON_LOG = Path("data/news_signal_daemon.log")
OUT = Path("data/eval/replay_events")
CACHE = Path("data/eval/cache/replay_v2.jsonl")


def parse_poll_line(line: str) -> dt.datetime | None:
    """Parse one daemon-log line into a poll timestamp (KST), or None.

    Only "Toss news poll complete" lines count as polls; other JSON lines and
    non-JSON lines (both occur in the log) are skipped.
    """
    if "Toss news poll complete" not in line:
        return None
    try:
        ts = dt.datetime.fromisoformat(json.loads(line)["timestamp"].replace("Z", "+00:00"))
    except (ValueError, KeyError, json.JSONDecodeError):
        return None
    return ts.astimezone(KST)


def poll_times(day: dt.date, log_path: Path = DAEMON_LOG) -> list[dt.datetime]:
    out = []
    for line in log_path.open():
        ts = parse_poll_line(line)
        if ts is not None and ts.date() == day:
            out.append(ts)
    return out


def judge_gap(rows: list[dict]) -> dt.timedelta:
    ts = sorted(dt.datetime.fromisoformat(r["judged_at"]) for r in rows)
    gaps = [
        (b - a).total_seconds()
        for a, b in zip(ts, ts[1:], strict=False)
        if 0 < (b - a).total_seconds() < 60
    ]
    return dt.timedelta(seconds=statistics.median(gaps))


def kr_quant_pairs(day: dt.date, prev_close_day: dt.date) -> list[dict]:
    """Pairs the daemon's filters would keep: published after the prior close, before 15:20."""
    with psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5) as c, c.cursor() as cur:
        cur.execute(
            """SELECT a.id, t.ticker, a.title, a.summary, a.content, a.category, a.published_at
               FROM news_articles a JOIN news_article_tickers t ON t.article_id = a.id
               WHERE a.source = 'toss' AND a.published_at >= %s AND a.published_at < %s""",
            (
                dt.datetime.combine(prev_close_day, dt.time(15, 20), tzinfo=KST),
                dt.datetime.combine(day, dt.time(15, 20), tzinfo=KST),
            ),
        )
        cols = ["id", "ticker", "title", "summary", "content", "category", "published_at"]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def prior_close_day(db: QuantAirflowDBClient, day: dt.date) -> dt.date:
    bars = db.get_adjusted_daily_bars(
        Ticker(value="005930"),
        start=day - dt.timedelta(days=14),
        end=day - dt.timedelta(days=1),
    )
    return bars[-1].date


async def judge(pairs: list[dict], stocks: dict) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if CACHE.exists():
        cache = {json.loads(line)["key"]: json.loads(line)["v2"] for line in CACHE.open()}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    async with AnthropicClient() as claude:
        for p in pairs:
            key = f"{p['live_id']}|{p['ticker']}"
            if key in cache:
                continue
            s = stocks[p["ticker"]]
            try:
                f = await claude.score_article(
                    code=p["ticker"],
                    name=s.name,
                    market=s.market,
                    sector=s.sector,
                    title=p["title"],
                    body=p["summary"] or p["content"],
                    category=p["category"],
                )
            except ExternalAPIError as exc:
                # A single pair's model output can fail the fixed schema
                # (e.g. a persistence value outside the allowed set) even
                # at temperature 0 and after retries -- skip that pair
                # rather than losing every already-judged pair in the run.
                print(f"WARNING: score_article failed for {key}: {exc} -- skipping this pair")
                continue
            cache[key] = {**asdict(f), "score": f.score()}
            with CACHE.open("a") as out:
                out.write(json.dumps({"key": key, "v2": cache[key]}, ensure_ascii=False) + "\n")
    return cache


def load_kr_quant_pairs(db: QuantAirflowDBClient, day: dt.date, stocks: dict) -> list[dict]:
    prior = prior_close_day(db, day)
    pairs = [p for p in kr_quant_pairs(day, prior) if p["ticker"] in stocks]
    for p in pairs:
        p["live_id"] = str(uuid5(TOSS_NEWS_NAMESPACE, p["id"]))
    return pairs


def build_kr_quant_events(
    pairs: list[dict],
    judged: dict[str, dict],
    stocks: dict,
    day: dt.date,
    delay: dt.timedelta,
    poll_every: dt.timedelta,
    gap: dt.timedelta,
) -> list[ReplayEvent]:
    sched = schedule_detected(
        [(p["live_id"], p["ticker"], p["published_at"].astimezone(KST)) for p in pairs],
        day,
        delay,
        poll_every,
        gap,
    )
    events = []
    seen: set[str] = set()
    for p in sorted(pairs, key=lambda x: x["published_at"]):
        key = f"{p['live_id']}|{p['ticker']}"
        if key in seen:
            continue
        seen.add(key)
        if key not in judged:
            # score_article failed schema validation for this pair (see judge()) --
            # no features to build an event from.
            continue
        decide, batch = sched[(p["live_id"], p["ticker"])]
        s = stocks[p["ticker"]]
        events.append(
            ReplayEvent(
                key=key,
                article_id=p["live_id"],
                ticker=p["ticker"],
                name=s.name,
                market=s.market,
                sector=s.sector,
                title=p["title"],
                category=p["category"],
                published_at=p["published_at"].astimezone(KST),
                decide_at=decide,
                batch_at=batch,
                features=judged[key],
                source="kr_quant",
            )
        )
    return events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kr-quant-days", default="")
    ap.add_argument(
        "--check-days",
        default="",
        help="Judge kr_quant pairs for these days into the cache only -- no event files written "
        "(reproducibility check against archive judgments; R4).",
    )
    args = ap.parse_args()
    stocks = QuantAirflowDBClient().get_stock_master()
    archive_days = sorted(dt.date.fromisoformat(p.stem) for p in ARCHIVE.glob("*.jsonl"))
    all_rows: list[dict] = []
    for day in archive_days:
        rows = [json.loads(line) for line in (ARCHIVE / f"{day}.jsonl").open()]
        all_rows += rows
        events = events_from_archive(rows, poll_times(day), stocks)
        dump_events(OUT / f"{day}.jsonl", events)
        print(day, "archive events", len(events))
    delay, gap = median_detect_delay(all_rows), judge_gap(all_rows)
    poll_every = dt.timedelta(seconds=get_settings().krx_news.toss_poll_interval_seconds)
    db = QuantAirflowDBClient()

    check_days = [dt.date.fromisoformat(d) for d in args.check_days.split(",") if d]
    for day in check_days:
        pairs = load_kr_quant_pairs(db, day, stocks)
        judged = asyncio.run(judge(pairs, stocks))
        print(day, "check-days judged", len(pairs), "pairs into cache (no event file written)")
        del judged  # cache already persisted to disk by judge()

    kq_days = [dt.date.fromisoformat(d) for d in args.kr_quant_days.split(",") if d]
    for day in kq_days:
        if day in archive_days:
            print(
                f"WARNING: {day} has an archive file -- skipping kr_quant events "
                "(archive days use the archive only, per data option B)"
            )
            continue
        pairs = load_kr_quant_pairs(db, day, stocks)
        judged = asyncio.run(judge(pairs, stocks))
        events = build_kr_quant_events(pairs, judged, stocks, day, delay, poll_every, gap)
        dump_events(OUT / f"{day}.jsonl", events)
        print(day, "kr_quant events", len(events))

    # kr_quant_days for meta.json is derived from what's actually on disk, not just this
    # invocation's --kr-quant-days -- so a --check-days-only run (no --kr-quant-days) never
    # clobbers the kr_quant_days recorded by an earlier run.
    existing_kq_days = sorted(
        d for p in OUT.glob("*.jsonl") if (d := dt.date.fromisoformat(p.stem)) not in archive_days
    )
    (OUT / "meta.json").write_text(
        json.dumps(
            {
                "median_delay_s": delay.total_seconds(),
                "judge_gap_s": gap.total_seconds(),
                "poll_every_s": poll_every.total_seconds(),
                "archive_days": [str(d) for d in archive_days],
                "kr_quant_days": [str(d) for d in existing_kq_days],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
