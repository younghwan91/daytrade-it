"""Pure library for the news universe pool: same pool for live and backtest.

`docs/superpowers/plans/2026-09-16-news-universe-a-plan.md` (C2). Selects the
sweep universe (previous day's traded value, top N), defines the sweep's
request order once (`SweepCursor`: fixed code order, paging, one detail call
per new article, one interval per request) and replays it deterministically
(`simulate_sweep`) so a historical article's operational discovery time is
the one the live sweep would have had,
reproduces the live daemon's article key (`TossNewsClient` /
`toss_news_client.TOSS_NEWS_NAMESPACE`) so the dashboard feed and the
sweep pool never double-judge the same article, and groups sweep rows into
per-article related-code sets.

No network, no DB, no I/O -- pure functions only (Global Constraints in the
plan doc). Everything here is deterministic so it can be replayed exactly.
"""

from __future__ import annotations

import bisect
import datetime as dt
import hashlib
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
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

#: The sweep makes no request at or after 15:20 KST (plan C2).
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


#: Items per company news page -- the live sweep's `size` parameter (C5).
PAGE_SIZE = 20


@dataclass(frozen=True)
class SweepRequest:
    """One Toss request of the sweep: a company news page, or an article detail."""

    kind: str  # "page" | "detail"
    code: str
    page: int = 1
    news_id: str | None = None


class SweepCursor:
    """The sweep's request order -- shared by the live runner (C5) and `simulate_sweep` (C3).

    Rules (each request costs one interval, enforced by the caller's clock):
      * codes are visited in universe order, repeating;
      * after a page, every item not seen earlier today (via any code), not
        stale (published before `stale_before`, or undated) gets one detail
        request, in page order, before anything else;
      * if the page was full (`page_size` items) and every item on it was new,
        the next page of the same code follows those details;
      * otherwise the next code's page 1.
    `page_result` must be called for each page request before `next_request`.
    """

    def __init__(
        self, universe: list[str], stale_before: dt.datetime, page_size: int = PAGE_SIZE
    ) -> None:
        if not universe:
            raise ValueError("empty universe")
        self.universe = list(universe)
        self.stale_before = stale_before
        self.page_size = page_size
        self.seen: set[str] = set()
        self._pending: deque[SweepRequest] = deque()
        self._index = 0
        self._awaiting: SweepRequest | None = None

    def next_request(self) -> SweepRequest:
        if self._awaiting is not None:
            raise RuntimeError(f"page_result not recorded for {self._awaiting}")
        if not self._pending:
            code = self.universe[self._index % len(self.universe)]
            self._index += 1
            self._pending.append(SweepRequest("page", code, 1))
        request = self._pending.popleft()
        if request.kind == "page":
            self._awaiting = request
        return request

    def page_result(
        self, request: SweepRequest, items: Sequence[tuple[str | None, dt.datetime | None]]
    ) -> list[str]:
        """Record a page's (news id, createdAt) items (empty on an HTTP error); new ids."""
        if request != self._awaiting:
            raise RuntimeError(f"unexpected page result for {request}")
        self._awaiting = None
        new: list[str] = []
        for news_id, created in items:
            if not news_id or news_id in self.seen:
                continue
            if created is None or created < self.stale_before:
                continue
            self.seen.add(news_id)
            new.append(news_id)
        self._pending.extend(SweepRequest("detail", request.code, request.page, n) for n in new)
        if len(items) >= self.page_size and len(new) == len(items):
            self._pending.append(SweepRequest("page", request.code, request.page + 1))
        return new


def page_items(
    listing: Sequence[tuple[dt.datetime, str]],
    at: dt.datetime,
    page: int,
    page_size: int = PAGE_SIZE,
) -> list[tuple[str, dt.datetime]]:
    """Page `page` (1-based) of a code's newest-first list as of `at`.

    `listing` is the code's (createdAt, news id) sorted ascending; an article is
    listed once createdAt <= `at`.
    """
    visible = bisect.bisect_right(listing, (at, "\U0010ffff"))
    hi = visible - (page - 1) * page_size
    if hi <= 0:
        return []
    lo = max(0, hi - page_size)
    return [(news_id, created) for created, news_id in reversed(listing[lo:hi])]


def simulate_sweep(
    universe: list[str],
    day: dt.date,
    arrivals: dict[str, list[tuple[str, dt.datetime]]],
    stale_before: dt.datetime,
    start: dt.time = _DEFAULT_START,
    end: dt.time = _CUTOFF,
    interval: dt.timedelta = SWEEP_INTERVAL,
    page_size: int = PAGE_SIZE,
    trace: list[SweepRequest] | None = None,
) -> dict[str, dt.datetime]:
    """News id -> when the live sweep would have discovered it on `day`.

    Deterministic replay of `SweepCursor` on a synthetic clock: request k starts
    at `start + k * interval` (only if that is before `end`) and completes one
    interval later; a page lists the code's `arrivals` published by the
    request's start (`page_items`); an article's discovery time is when its
    detail request completes (every pair of the article shares it). Articles
    never reached before `end` are absent. `trace`, if given, receives the
    request sequence (the live runner's order must equal it).
    """
    found: dict[str, dt.datetime] = {}
    if not universe:
        return found
    listings = {code: sorted((c, n) for n, c in rows) for code, rows in arrivals.items()}
    cursor = SweepCursor(universe, stale_before, page_size)
    now = dt.datetime.combine(day, start, tzinfo=KST)
    stop = dt.datetime.combine(day, end, tzinfo=KST)
    while now < stop:
        request = cursor.next_request()
        if trace is not None:
            trace.append(request)
        if request.kind == "page":
            items = page_items(listings.get(request.code, ()), now, request.page, page_size)
            cursor.page_result(request, items)
        else:
            assert request.news_id is not None
            found[request.news_id] = now + interval
        now += interval
    return found


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


def judgment_input_hash(model: str, system: str, user: str) -> str:
    """sha256 over the judged input (model, system prompt, user prompt), length-framed.

    Judgment caches key rows by (pair key, this hash) so a pair whose prompt
    inputs changed (e.g. the body fallback) is re-judged instead of reusing a
    judgment of different text.
    """
    h = hashlib.sha256()
    for part in (model, system, user):
        data = part.encode()
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


def judge_queue(
    items: list[tuple[str, dt.datetime, int]], per_pair: dt.timedelta
) -> dict[str, list[dt.datetime]]:
    """Finish time of every (article, code) pair, replaying the live sweep's one judge worker.

    `items` is (news_id, discovered_at, n_pairs) in discovery order (ties
    broken by input order -- this never re-sorts). The live runner judges
    pairs one at a time with a single worker in discovery order (F3/F4); an
    article's pairs are all enqueued together at its `discovered_at` (the
    detail call that found it also resolved its related codes) and judged in
    that order, each pair costing `per_pair`. So pair i (0-based) of an
    article finishes at `start + (i + 1) * per_pair`, where
    `start = max(discovered_at, previous article's last finish)` -- a backlog
    (the worker still busy when a later article arrives) carries forward
    instead of resetting; once the worker catches up (`discovered_at` after
    the previous finish), `start` resets to `discovered_at`.
    """
    finishes: dict[str, list[dt.datetime]] = {}
    prev_finish: dt.datetime | None = None
    for news_id, discovered_at, n_pairs in items:
        start = discovered_at if prev_finish is None else max(discovered_at, prev_finish)
        times = [start + (i + 1) * per_pair for i in range(n_pairs)]
        finishes[news_id] = times
        if times:
            prev_finish = times[-1]
    return finishes


def related_codes(rows_by_id: dict[str, set[str]]) -> dict[str, tuple[str, ...]]:
    """Article id -> sorted tuple of codes it appeared under across the sweep.

    `rows_by_id` is the raw grouping (article id -> set of codes whose
    per-code sweep page returned that article); this just makes the mapping
    deterministic (sorted, immutable) for downstream use.
    """
    return {article_id: tuple(sorted(codes)) for article_id, codes in rows_by_id.items()}
