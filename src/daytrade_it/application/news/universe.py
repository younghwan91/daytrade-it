"""Pure library for the news universe pool: same pool for live and backtest.

`docs/superpowers/plans/2026-09-16-news-universe-a-plan.md` (C2). Selects the
sweep universe (previous day's traded value, top N), computes the
deterministic time a code is next visited by the fixed-order/fixed-interval
sweep (so a historical article's operational discovery time is computable),
reproduces the live daemon's article key (`TossNewsClient` /
`toss_news_client.TOSS_NEWS_NAMESPACE`) so the dashboard feed and the
sweep pool never double-judge the same article, and groups sweep rows into
per-article related-code sets.

No network, no DB, no I/O -- pure functions only (Global Constraints in the
plan doc). Everything here is deterministic so it can be replayed exactly.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from uuid import uuid5

from krx_news_client.scrapers.toss import build_article_url
from krx_quant_core.market.session import KST

from daytrade_it.infrastructure.external.toss_news_client import TOSS_NEWS_NAMESPACE

#: Default sweep size: previous day's top-N traded-value codes (spec design
#: doc, "대상 종목: 전일 거래대금 상위 N(기본 600, 장 전 고정)").
UNIVERSE_SIZE = 600

#: Default per-code sweep cadence (spec design doc: "1초에 1회씩").
SWEEP_INTERVAL = dt.timedelta(seconds=1)

#: Sweep start-of-day (spec design doc / plan C2: "start" = 08:00 KST).
_DEFAULT_START = dt.time(8, 0)

#: The sweep stops making new discoveries at 15:20 KST (plan C2): a computed
#: visit at or after this time is not one that operationally would have
#: happened, so `next_visit` returns None instead.
_CUTOFF = dt.time(15, 20)


def select_universe(daily_values: dict[str, float], size: int = UNIVERSE_SIZE) -> list[str]:
    """The sweep universe for a day, given the previous day's traded value per code.

    Sorted by value descending, ties broken by code ascending (deterministic).
    Codes that aren't a plain 6-digit string, or whose value isn't positive,
    are ignored -- matches the plan's "전일 거래대금 내림차순, 동률은 코드
    오름차순" plus the implicit "실제 거래된 종목만" filter.
    """
    candidates = [
        code
        for code, value in daily_values.items()
        if value > 0 and len(code) == 6 and code.isdigit()
    ]
    candidates.sort(key=lambda code: (-daily_values[code], code))
    return candidates[:size]


def next_visit(
    code: str,
    universe: list[str],
    day: dt.date,
    after: dt.datetime,
    start: dt.time = _DEFAULT_START,
    interval: dt.timedelta = SWEEP_INTERVAL,
) -> dt.datetime | None:
    """The next time the sweep visits `code`, at or after `after`, on `day`.

    The sweep visits `universe[i]` at `start + (k * len(universe) + i) *
    interval` for k = 0, 1, 2, ... (fixed order, fixed cadence, repeating).
    `after` may be naive (assumed KST) or aware (converted to KST). Returns
    None if `code` isn't in `universe`, or if that next visit would fall at or
    after 15:20 KST on `day` (the sweep makes no more discoveries past that
    point, per the plan's Global Constraints on backtest = live-achievable).

    O(1) arithmetic (plus the one `universe.index` lookup) -- never loops
    over candidate visit timestamps.
    """
    try:
        i = universe.index(code)
    except ValueError:
        return None
    n = len(universe)

    after = after.replace(tzinfo=KST) if after.tzinfo is None else after.astimezone(KST)

    day_start = dt.datetime.combine(day, start, tzinfo=KST)
    cutoff = dt.datetime.combine(day, _CUTOFF, tzinfo=KST)

    diff = after - day_start
    # p = ceil(diff / interval): the smallest slot index (unconstrained to
    # this code's residue class) whose time is >= after. Exact integer
    # arithmetic via timedelta divmod -- no float division.
    q, r = divmod(diff, interval)
    p = q if r == dt.timedelta(0) else q + 1

    if p <= i:
        m = i
    else:
        k = -(-(p - i) // n)  # ceil division, all ints
        m = i + k * n

    visit_time = day_start + m * interval
    if visit_time >= cutoff:
        return None
    return visit_time


def live_article_id(news_id: str) -> str:
    """The live daemon's dedup key for a Toss article, given its Toss news id.

    Matches `TossNewsClient`'s key exactly (`uuid5(TOSS_NEWS_NAMESPACE,
    "toss:" + md5(build_article_url(news_id))[:12])`) so a sweep-discovered
    article and a dashboard-discovered article that are the same Toss article
    resolve to the same id and are never double-judged.
    """
    url = build_article_url(news_id)
    digest = hashlib.md5(url.encode()).hexdigest()[:12]  # noqa: S324 -- id derivation, not security
    return str(uuid5(TOSS_NEWS_NAMESPACE, f"toss:{digest}"))


def related_codes(rows_by_id: dict[str, set[str]]) -> dict[str, tuple[str, ...]]:
    """Article id -> sorted tuple of codes it appeared under across the sweep.

    `rows_by_id` is the raw grouping (article id -> set of codes whose
    per-code sweep page returned that article); this just makes the mapping
    deterministic (sorted, immutable) for downstream use.
    """
    return {article_id: tuple(sorted(codes)) for article_id, codes in rows_by_id.items()}
