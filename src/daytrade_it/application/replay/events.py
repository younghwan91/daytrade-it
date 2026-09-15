"""Replay input: one judged (article, ticker) pair with the time the daemon acted on it.

Live days come from the daemon's judgment archive (`data/news_judgments`):
`decide_at` is the real `judged_at`, and a poll's signals are entered right
after its last judgment (`batch_at`). Days before the daemon ran come from
kr_quant articles, detected at publish + the archive's median delay on the
daemon's poll grid. Features are the stored v2 judgment -- replay never
re-asks the model for a pair the daemon already judged.
"""

from __future__ import annotations

import bisect
import datetime as dt
import json
import statistics
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from uuid import UUID

from daytrade_it.domain.entities.news import NewsArticle
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.external.anthropic_client import ArticleFeatures
from daytrade_it.infrastructure.external.quant_airflow_db_client import StockInfo

FIRST_POLL = dt.time(9, 0, 30)
_FEATURE_FIELDS = {f.name for f in fields(ArticleFeatures)}


@dataclass(frozen=True)
class ReplayEvent:
    key: str
    article_id: str
    ticker: str
    name: str
    market: str | None
    sector: str | None
    title: str
    category: str | None
    published_at: dt.datetime
    decide_at: dt.datetime
    batch_at: dt.datetime
    features: dict
    source: str


def to_article(e: ReplayEvent) -> NewsArticle:
    return NewsArticle(
        id=UUID(e.article_id),
        title=e.title,
        content=e.title,
        source="toss",
        url="https://tossinvest.com/news",
        published_at=e.published_at,
        tickers=[Ticker(value=e.ticker)],
        category=e.category,
    )


def features_of(e: ReplayEvent) -> ArticleFeatures:
    return ArticleFeatures(**{k: v for k, v in e.features.items() if k in _FEATURE_FIELDS})


def dump_events(path: Path, events: list[ReplayEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in events:
            row = asdict(e)
            for k in ("published_at", "decide_at", "batch_at"):
                row[k] = row[k].isoformat()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_events(path: Path) -> list[ReplayEvent]:
    out = []
    for line in path.open():
        row = json.loads(line)
        for k in ("published_at", "decide_at", "batch_at"):
            row[k] = dt.datetime.fromisoformat(row[k])
        out.append(ReplayEvent(**row))
    return out


def events_from_archive(
    rows: list[dict], poll_times: list[dt.datetime], stocks: dict[str, StockInfo]
) -> list[ReplayEvent]:
    first: dict[str, dict] = {}
    for r in rows:
        if r.get("v2") and r["key"] not in first:
            first[r["key"]] = r
    kept = sorted(first.values(), key=lambda r: r["judged_at"])
    polls = sorted(poll_times)

    def poll_of(ts: dt.datetime) -> int:
        return bisect.bisect_right(polls, ts) - 1

    last_in_poll: dict[int, dt.datetime] = {}
    for r in kept:
        ts = dt.datetime.fromisoformat(r["judged_at"])
        last_in_poll[poll_of(ts)] = ts
    events = []
    for r in kept:
        judged = dt.datetime.fromisoformat(r["judged_at"])
        stock = stocks.get(r["ticker"])
        events.append(
            ReplayEvent(
                key=r["key"],
                article_id=r["article_id"],
                ticker=r["ticker"],
                name=r["name"],
                market=stock.market if stock else None,
                sector=stock.sector if stock else None,
                title=r["title"],
                category=r.get("category"),
                published_at=dt.datetime.fromisoformat(r["published_at"]),
                decide_at=judged,
                batch_at=last_in_poll[poll_of(judged)] + dt.timedelta(seconds=1),
                features=r["v2"],
                source="archive",
            )
        )
    return events


def median_detect_delay(rows: list[dict]) -> dt.timedelta:
    delays = []
    for r in rows:
        pub = dt.datetime.fromisoformat(r["published_at"])
        if pub.timetz().replace(tzinfo=None) >= FIRST_POLL:
            delays.append((dt.datetime.fromisoformat(r["judged_at"]) - pub).total_seconds())
    return dt.timedelta(seconds=statistics.median(delays))


def schedule_detected(
    items: list[tuple[str, str, dt.datetime]],
    day: dt.date,
    delay: dt.timedelta,
    poll_every: dt.timedelta,
    judge_gap: dt.timedelta,
) -> dict[tuple[str, str], tuple[dt.datetime, dt.datetime]]:
    tz = items[0][2].tzinfo if items else None
    start = dt.datetime.combine(day, FIRST_POLL, tzinfo=tz)
    slots: dict[dt.datetime, list[tuple[str, str, dt.datetime]]] = {}
    for aid, ticker, pub in sorted(items, key=lambda x: x[2]):
        seen = max(pub + delay, start)
        k = -(-(seen - start) // poll_every)  # ceil
        slots.setdefault(start + k * poll_every, []).append((aid, ticker, pub))
    out = {}
    for slot, group in slots.items():
        batch = slot + (len(group) - 1) * judge_gap + dt.timedelta(seconds=1)
        for i, (aid, ticker, _) in enumerate(group):
            out[(aid, ticker)] = (slot + i * judge_gap, batch)
    return out
