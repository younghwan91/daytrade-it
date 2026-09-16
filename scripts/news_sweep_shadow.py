"""Universe sweep shadow runner -- judges per-company Toss news like the live daemon, NO orders.

Plan C5 (`docs/superpowers/plans/2026-09-16-news-universe-a-plan.md`). Sweeps
the day's universe (previous trading day's top 600 by traded value, fixed at
start; `application/news/universe.py`) one request per second in fixed order
from 08:00 to 15:20 KST, fetching company news pages
(`/api/v2/news/companies/{code}?number=N&size=20`). For every news id not seen
earlier in the sweep, one detail call (same 1 req/s budget) gives the
article's `companyCodes`; a page full of new items is followed by the next page.
Articles published before the previous session close are skipped (the live
stale rule). The request order is `application/news/universe.SweepCursor` --
the same object `simulate_sweep` replays for the backtest (C3), so detection
times match (F3/F4).

Articles the dashboard daemon already judged stay in the pool (backtest pool =
sweep pool): they still get the detail call, pairs the dashboard archive has
are copied with `"pool": "dashboard"` (not re-judged), and missing related
pairs are judged with `"pool": "universe"`. Universe pairs are judged with
category "" (per-company pages carry no dashboard category; A3 parity).

Judging reuses the live path (`application/news/judging.judge_pair`: same
prompt, same v2 score, same archive row); v4 shadow is off (spec). Rows go to
`data/news_judgments_universe/{day}.jsonl` in the live archive format plus
`"pool": "universe"` and `"sweep_seen_at"` (when the article's detail call
completed = its discovery time). Every request is logged to
`data/news_sweep/{day}.jsonl` (code, page, ts, status, new_ids) for
measuring detection delay. Only with `--with-broker` (uses the LIVE Kiwoom
token: KIWOOM_APP_KEY/SECRET, quotes and minute bars only), the live entry filter
(`build_entry_candidate`, broker quotes/minute bars only) is evaluated from
09:00:30 and written to `data/news_sweep_candidates/{day}.jsonl` -- nothing
is persisted to gptquant and no trader is ever constructed.

Logging: judging goes through `application/news/judging.py`, so its structlog
events keep the live strings ("news_signal_poller: judged", "... scoring
failed") under that module's logger name; this script's own events are
"news_sweep_shadow: ...".

SAFETY -- read before running:
  * Running this during market hours from the trader's IP competes with the
    live daemon's Toss access (same unofficial API; ~3600 extra requests/hour).
    Do NOT schedule it (no cron/systemd). Run it only with the user's explicit
    approval. It refuses to start without `--proxy URL` unless
    `--allow-trader-ip` is passed explicitly.
  * Default is judge-only (no broker). `--with-broker` connects with the LIVE
    Kiwoom app key/secret (quotes/minute bars only, never an order) -- that
    token is shared with the live trader, so opt in only deliberately.
  * The sweep stops after 3 consecutive 429/403 responses.
  * `--dry-run-http` prints the request schedule and exits without any HTTP.
  * `--max-requests N` caps total Toss requests; SIGTERM/SIGINT stop cleanly.

Usage:
    uv run python scripts/news_sweep_shadow.py --dry-run-http --universe-file codes.txt
    uv run python scripts/news_sweep_shadow.py --proxy http://host:3128 --max-requests 600
    uv run python scripts/news_sweep_shadow.py --proxy http://host:3128 --with-broker
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import json
import os
import signal
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx
from krx_news_client.scrapers.toss import COMPANY_NEWS_URL, build_article_url
from krx_quant_core.market.session import KST

from daytrade_it.application.news import judging
from daytrade_it.application.news.judging import (
    JudgmentLog,
    judge_pair,
    pair_key,
    session_close_kst,
)
from daytrade_it.application.news.universe import (
    PAGE_SIZE,
    SWEEP_INTERVAL,
    UNIVERSE_SIZE,
    SweepCursor,
    live_article_id,
    select_universe,
)
from daytrade_it.application.signals.news_entry import tradable_tickers
from daytrade_it.domain.entities.news import NewsArticle
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.external.toss_news_client import fetch_article_detail
from daytrade_it.shared.logging import get_logger

logger = get_logger(__name__)

SWEEP_START = dt.time(8, 0)
SWEEP_STOP = dt.time(15, 20)
#: Candidates (entry filter) only from the live daemon's first poll: before
#: the opening auction prints, quotes are pre-open and not what live used.
CANDIDATES_FROM = dt.time(9, 0, 30)
BLOCK_STATUSES = frozenset({403, 429})
BLOCK_LIMIT = 3

DASHBOARD_JUDGMENT_DIR = Path("data/news_judgments")
UNIVERSE_JUDGMENT_DIR = Path("data/news_judgments_universe")
SWEEP_LOG_DIR = Path("data/news_sweep")
CANDIDATE_DIR = Path("data/news_sweep_candidates")
POLICY_PATH = Path("config/news_policy.json")
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
}


# --- schedule ---------------------------------------------------------------


@dataclass(frozen=True)
class SweepSchedule:
    """Sweep window and a page-1-only slot plan (for `--dry-run-http`).

    The real request order also spends slots on detail calls and extra pages
    (`SweepCursor`), so actual visit times drift later than `planned_at`.
    """

    universe: list[str]
    day: dt.date
    start: dt.time = SWEEP_START
    stop: dt.time = SWEEP_STOP
    interval: dt.timedelta = SWEEP_INTERVAL

    @property
    def _t0(self) -> dt.datetime:
        return dt.datetime.combine(self.day, self.start, tzinfo=KST)

    def first_slot(self, now: dt.datetime) -> int:
        """The first slot at or after `now` (0 before the start)."""
        diff = now - self._t0
        if diff <= dt.timedelta(0):
            return 0
        q, r = divmod(diff, self.interval)
        return q if r == dt.timedelta(0) else q + 1

    def end_slot(self) -> int:
        """First slot at or after the stop time (exclusive bound)."""
        return self.first_slot(dt.datetime.combine(self.day, self.stop, tzinfo=KST))

    def code_at(self, slot: int) -> str:
        return self.universe[slot % len(self.universe)]

    def planned_at(self, slot: int) -> dt.datetime:
        return self._t0 + slot * self.interval


def company_page_url(code: str) -> str:
    return f"{COMPANY_NEWS_URL.format(code=code)}?number=1&size={PAGE_SIZE}"


def dry_run_lines(schedule: SweepSchedule, now: dt.datetime, count: int) -> list[str]:
    first = schedule.first_slot(now)
    end = min(schedule.end_slot(), first + count)
    return [
        f"{schedule.planned_at(m).isoformat()} {schedule.code_at(m)} GET {company_page_url(schedule.code_at(m))}"
        for m in range(first, end)
    ]


# --- request budget + block stop ---------------------------------------------


class RateLimiter:
    """At least `interval` seconds between any two Toss requests (page or detail)."""

    def __init__(
        self,
        interval: float = SWEEP_INTERVAL.total_seconds(),
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._interval = interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    async def wait(self) -> None:
        if self._last is not None:
            remaining = self._last + self._interval - self._clock()
            if remaining > 0:
                await self._sleep(remaining)
        self._last = self._clock()


@dataclass
class BlockGuard:
    """Trips after `limit` consecutive 429/403. A success resets; other errors don't count."""

    limit: int = BLOCK_LIMIT
    consecutive: int = 0
    tripped: bool = False

    def record(self, status: int | None) -> bool:
        if status in BLOCK_STATUSES:
            self.consecutive += 1
            if self.consecutive >= self.limit:
                self.tripped = True
        elif status is not None and 200 <= status < 300:
            self.consecutive = 0
        return self.tripped


# --- dedupe -------------------------------------------------------------------


class DashboardKeys:
    """Rows the live dashboard daemon already judged today (re-read when the file grows)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._size = -1
        self._ids: set[str] = set()
        self._rows: dict[str, dict[str, Any]] = {}

    def _refresh(self) -> None:
        try:
            size = self._path.stat().st_size
        except FileNotFoundError:
            return
        if size == self._size:
            return
        rows: dict[str, dict[str, Any]] = {}
        with self._path.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                    rows.setdefault(row["key"], row)  # first judgment of a pair wins
                except (ValueError, KeyError, TypeError):
                    continue  # a half-written last line
        self._rows, self._size = rows, size
        self._ids = {key.split("|", 1)[0] for key in rows}

    def row(self, key: str) -> dict[str, Any] | None:
        """The dashboard archive row for a pair key, if the dashboard judged that pair."""
        self._refresh()
        return self._rows.get(key)

    def __contains__(self, article_id: str) -> bool:
        self._refresh()
        return article_id in self._ids


def _created_at(item: dict[str, Any]) -> dt.datetime | None:
    try:
        parsed = dt.datetime.fromisoformat(item["createdAt"])
    except (KeyError, TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=KST) if parsed.tzinfo is None else parsed


def is_stale(item: dict, stale_before: dt.datetime) -> bool:
    """Live stale rule: published before the previous session close (or no usable time)."""
    created = _created_at(item)
    return created is None or created < stale_before


# --- article + archive ------------------------------------------------------------


def build_article(item: dict, detail: dict[str, Any] | None) -> NewsArticle:
    """Domain article under the live key; tickers from detail `companyCodes` (page codes if none)."""
    news_id = item["id"]
    codes = (detail or {}).get("companyCodes") or item.get("stockCodes") or []
    tickers: list[Ticker] = []
    for code in codes:
        with contextlib.suppress(ValueError):
            tickers.append(Ticker(value=str(code)))
    # Sorted ascending by code (not API order): C3's `related_codes` sorts an
    # article's codes too (`tuple(sorted(codes))`), so `judge_queue`'s model
    # of "an article's pairs judged one at a time, in order" refers to the
    # same order on both sides (A-plan fix follow-up).
    tickers.sort(key=lambda t: t.value)
    # Mirrors `toss_news_client.map_toss_article_to_domain`, so the judged body
    # (`summary or content`) is summary, else content[:500], else title.
    text = item.get("contentText") or ""
    summary = item.get("summary") or (text[:500] if text else None)
    return NewsArticle(
        id=UUID(live_article_id(news_id)),
        title=item["title"],
        content=text or summary or item["title"],
        summary=summary,
        source="toss",
        url=build_article_url(news_id),
        published_at=_created_at(item),
        tickers=tickers,
        # Per-company pages have no dashboard category; A3 judges history with "" too.
        category="",
    )


class UniverseJudgmentLog(JudgmentLog):
    """Live `JudgmentLog` rows + `"pool": "universe"` and when the sweep first saw the article."""

    def __init__(self, directory: Path, day: dt.date) -> None:
        super().__init__(directory, day)
        self._seen_at: dict[str, dt.datetime] = {}

    def set_context(self, article_id: UUID | str, sweep_seen_at: dt.datetime) -> None:
        self._seen_at[str(article_id)] = sweep_seen_at

    def append(self, key: str, record: dict[str, Any], pool: str = "universe") -> None:
        seen_at = self._seen_at.get(key.split("|", 1)[0])
        super().append(
            key,
            {
                **record,
                "pool": pool,
                "sweep_seen_at": seen_at.isoformat() if seen_at else None,
            },
        )

    def append_dashboard_copy(self, dashboard_row: dict[str, Any]) -> None:
        """The dashboard's row for a pair, unchanged, tagged `pool: "dashboard"` (no re-judging)."""
        record = {k: v for k, v in dashboard_row.items() if k != "key"}
        self.append(dashboard_row["key"], record, pool="dashboard")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


# --- HTTP -----------------------------------------------------------------------------


async def fetch_company_page(
    http: httpx.AsyncClient, code: str, page: int = 1
) -> tuple[int | None, list[dict[str, Any]]]:
    """(HTTP status or None on a network error, the page's items)."""
    try:
        resp = await http.get(
            COMPANY_NEWS_URL.format(code=code), params={"number": page, "size": PAGE_SIZE}
        )
    except httpx.HTTPError as exc:
        logger.warning("news_sweep_shadow: page request failed", code=code, error=str(exc))
        return None, []
    if not resp.is_success:
        return resp.status_code, []
    try:
        body = ((resp.json() or {}).get("result") or {}).get("body")
    except ValueError:
        body = None
    return resp.status_code, body if isinstance(body, list) else []


# --- runner -------------------------------------------------------------------------


@dataclass
class Queued:
    item: dict[str, Any]
    detail: dict[str, Any] | None
    seen_at: dt.datetime


def prior_day_trade_values(db_client: Any, day: dt.date) -> tuple[dt.date, dict[str, float]]:
    """(previous trading day, code -> traded value) from kr_quant daily bars (read-only)."""
    rows = db_client._query(
        "SELECT code, date, trade_value FROM daily_bars_adjusted "
        "WHERE date = (SELECT max(date) FROM daily_bars_adjusted WHERE date < %s AND date >= %s)",
        (day, day - dt.timedelta(days=14)),
    )
    if not rows:
        raise RuntimeError(f"no daily bars in the 14 days before {day}")
    return rows[0][1], {r[0]: float(r[2] or 0) for r in rows}


def _features_from_row(row: dict[str, Any]) -> Any:
    """v2 features back from an archive row (for the shadow entry filter), or None."""
    from daytrade_it.infrastructure.external.anthropic_client import ArticleFeatures

    v2 = dict(row.get("v2") or {})
    v2.pop("score", None)
    try:
        return ArticleFeatures(**v2)
    except TypeError:
        return None


class ShadowRunner:
    def __init__(
        self,
        *,
        day: dt.date,
        schedule: SweepSchedule,
        stocks: dict[str, Any],
        stale_before: dt.datetime,
        http: httpx.AsyncClient,
        claude: Any,
        candidate_ctx: dict[str, Any] | None,
        max_requests: int | None,
    ) -> None:
        self.day = day
        self.schedule = schedule
        self.stocks = stocks
        self.stale_before = stale_before
        self.expires_at = session_close_kst(day)
        self.http = http
        self.claude = claude
        self.candidate_ctx = candidate_ctx
        self.max_requests = max_requests
        self.requests = 0
        self.limiter = RateLimiter()
        self.guard = BlockGuard()
        self.dashboard = DashboardKeys(DASHBOARD_JUDGMENT_DIR / f"{day}.jsonl")
        self.cursor = SweepCursor(schedule.universe, stale_before, PAGE_SIZE)
        self.judgments = UniverseJudgmentLog(UNIVERSE_JUDGMENT_DIR, day)
        self.sweep_log = SWEEP_LOG_DIR / f"{day}.jsonl"
        self.candidate_log = CANDIDATE_DIR / f"{day}.jsonl"
        self.queue: asyncio.Queue[Queued] = asyncio.Queue()
        self.stop = asyncio.Event()
        self.stop_reason = ""
        self.now: Callable[[], dt.datetime] = lambda: dt.datetime.now(KST)

    def _halt(self, reason: str) -> None:
        if not self.stop.is_set():
            self.stop_reason = reason
            logger.warning("news_sweep_shadow: stopping sweep", reason=reason)
            self.stop.set()

    async def _request(self) -> bool:
        """Wait for the 1 req/s budget; False if the sweep must stop instead."""
        if self.stop.is_set():
            return False
        if self.now().time() >= self.schedule.stop:
            self._halt("15:20 KST")
            return False
        if self.max_requests is not None and self.requests >= self.max_requests:
            self._halt(f"max_requests {self.max_requests} reached")
            return False
        await self.limiter.wait()
        if self.stop.is_set():
            return False
        self.requests += 1
        return True

    def _record_status(self, status: int | None) -> None:
        if self.guard.record(status):
            self._halt(f"{BLOCK_LIMIT} consecutive 429/403 from Toss")

    async def sweep(self) -> None:
        now = self.now()
        wait = (self.schedule.planned_at(0) - now).total_seconds()
        if wait > 0:
            logger.info("news_sweep_shadow: waiting for sweep start", seconds=int(wait))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.stop.wait(), wait)
        pending_items: dict[str, dict[str, Any]] = {}
        while not self.stop.is_set():
            if self.now().time() >= self.schedule.stop:
                self._halt("15:20 KST")
                break
            request = self.cursor.next_request()
            if not await self._request():
                break
            ts = self.now()
            if request.kind == "page":
                status, items = await fetch_company_page(self.http, request.code, request.page)
                self._record_status(status)
                new_ids = self.cursor.page_result(
                    request, [(i.get("id"), _created_at(i)) for i in items]
                )
                fresh = set(new_ids)
                for item in items:
                    if item.get("id") in fresh:
                        pending_items.setdefault(item["id"], item)
                append_jsonl(
                    self.sweep_log,
                    {
                        "code": request.code,
                        "page": request.page,
                        "ts": ts.isoformat(),
                        "status": status,
                        "items": len(items),
                        "new_ids": new_ids,
                        "dashboard_ids": [
                            n for n in new_ids if live_article_id(n) in self.dashboard
                        ],
                    },
                )
                continue
            assert request.news_id is not None
            item = pending_items.pop(request.news_id)
            detail, detail_status = await self._fetch_detail(request.news_id)
            self._record_status(detail_status)
            seen_at = self.now()  # detail completed = discovery (simulate_sweep)
            append_jsonl(
                self.sweep_log,
                {
                    "code": request.code,
                    "ts": ts.isoformat(),
                    "detail_id": request.news_id,
                    "status": detail_status,
                },
            )
            await self.queue.put(Queued(item, detail, seen_at))

    async def _fetch_detail(self, news_id: str) -> tuple[dict[str, Any] | None, int | None]:
        """Live `fetch_article_detail`, with the HTTP status captured for the block guard."""
        statuses: list[int] = []

        async def hook(response: httpx.Response) -> None:
            statuses.append(response.status_code)

        self.http.event_hooks["response"].append(hook)
        try:
            detail = await fetch_article_detail(self.http, build_article_url(news_id))
        except ValueError:  # malformed JSON on a 2xx
            detail = None
        finally:
            self.http.event_hooks["response"].remove(hook)
        return detail, (statuses[-1] if statuses else None)

    async def judge_worker(self) -> None:
        while True:
            queued = await self.queue.get()
            try:
                await self._judge(queued)
            except Exception:
                logger.exception("news_sweep_shadow: judging failed", news_id=queued.item.get("id"))
            finally:
                self.queue.task_done()

    async def _judge(self, queued: Queued) -> None:
        try:
            article = build_article(queued.item, queued.detail)
        except (KeyError, ValueError):
            logger.warning("news_sweep_shadow: unusable item", news_id=queued.item.get("id"))
            return
        now = self.now()
        members = set(self.schedule.universe)
        # S13: only related codes that are in today's universe (the backtest pool, C3).
        known = [
            (ticker, stock)
            for ticker, stock in tradable_tickers(
                article, self.stocks, self.stale_before, self.expires_at, now
            )
            if ticker.value in members
        ]
        self.judgments.set_context(article.id, queued.seen_at)
        for ticker, stock in known:
            key = pair_key(article, ticker)
            if self.judgments.seen(key):
                continue
            dashboard_row = self.dashboard.row(key)
            features: Any
            if dashboard_row is not None:
                self.judgments.append_dashboard_copy(dashboard_row)
                features = _features_from_row(dashboard_row)
            else:
                features = await judge_pair(
                    article, ticker, stock, self.claude, self.judgments, None
                )
            if features is None or self.candidate_ctx is None:
                continue
            if self.now().time() < CANDIDATES_FROM:
                continue
            self._candidate(article, ticker, features)

    def _candidate(self, article: NewsArticle, ticker: Ticker, features: Any) -> None:
        from daytrade_it.application.signals.news_entry import build_entry_candidate

        ctx = self.candidate_ctx
        assert ctx is not None
        try:
            candidate = build_entry_candidate(
                article,
                ticker,
                features,
                ctx["broker"],
                ctx["db_client"],
                ctx["gen"],
                self.expires_at,
                selection=ctx["selection"],
            )
        except Exception:
            logger.exception("news_sweep_shadow: entry filter failed", ticker=ticker.value)
            return
        if candidate is None:
            return
        append_jsonl(
            self.candidate_log,
            {
                "key": pair_key(article, ticker),
                "pool": "universe",
                "at": self.now().isoformat(),
                "ticker": ticker.value,
                "surprise": candidate.surprise,
                "score": candidate.score,
                "metadata": candidate.signal.metadata,
            },
        )
        logger.info("news_sweep_shadow: shadow candidate (no order)", ticker=ticker.value)

    async def run(self) -> None:
        worker = asyncio.create_task(self.judge_worker())
        try:
            await self.sweep()
            # Clean stop: finish the article being judged, drop the rest.
            dropped = 0
            while not self.queue.empty() and self.stop_reason.startswith("signal"):
                self.queue.get_nowait()
                self.queue.task_done()
                dropped += 1
            if dropped:
                logger.warning("news_sweep_shadow: dropped unjudged articles", count=dropped)
            await self.queue.join()
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        logger.info(
            "news_sweep_shadow: done",
            reason=self.stop_reason,
            requests=self.requests,
            seen=len(self.cursor.seen),
            # Judging runs through the live `judge_pair`, so the sweep inherits
            # the same pre-LLM filter the daemon uses (off by default).
            prefilter_skipped=judging.prefilter_skips,
        )


def load_universe_file(path: Path) -> list[str]:
    return [c.strip() for c in path.read_text().split() if c.strip()]


def check_network_args(args: argparse.Namespace) -> None:
    """Refuse a real sweep from the trader's own IP unless explicitly allowed (F6)."""
    if args.dry_run_http or args.proxy or args.allow_trader_ip:
        return
    raise SystemExit(
        "refusing to sweep Toss from this host's IP (competes with the live daemon): "
        "pass --proxy URL, or --allow-trader-ip to override deliberately"
    )


async def main_async(args: argparse.Namespace) -> None:
    check_network_args(args)
    day = args.day or dt.datetime.now(KST).date()

    db_client = None
    if args.universe_file:
        universe = load_universe_file(args.universe_file)[: args.size]
        prev_day = None
    else:
        from daytrade_it.infrastructure.external.quant_airflow_db_client import (
            QuantAirflowDBClient,
        )

        db_client = QuantAirflowDBClient()
        prev_day, values = prior_day_trade_values(db_client, day)
        universe = select_universe(values, args.size)
    if not universe:
        raise SystemExit("empty universe")
    schedule = SweepSchedule(universe, day)

    if args.dry_run_http:
        count = args.max_requests or 20
        print(f"# universe={len(universe)} day={day} prev_day={prev_day} (no HTTP calls)")
        for line in dry_run_lines(schedule, dt.datetime.now(KST), count):
            print(line)
        return

    if db_client is None:
        from daytrade_it.infrastructure.external.quant_airflow_db_client import (
            QuantAirflowDBClient,
        )

        db_client = QuantAirflowDBClient()
        prev_day, _ = prior_day_trade_values(db_client, day)
    stocks = db_client.get_stock_master()
    assert prev_day is not None
    stale_before = session_close_kst(prev_day)

    candidate_ctx: dict[str, Any] | None = None
    broker = None
    if args.with_broker:
        from kiwoom_client import KiwoomAPI

        from daytrade_it.application.trading.policy import load_policy
        from daytrade_it.domain.services.signal_generator import SignalConfig, SignalGenerator
        from daytrade_it.infrastructure.brokers.kiwoom_broker import KiwoomBroker

        # Quotes/minute bars only (no-chase filter); no order method is ever called.
        broker = KiwoomBroker(
            api=KiwoomAPI(
                app_key=os.environ["KIWOOM_APP_KEY"],
                app_secret=os.environ["KIWOOM_APP_SECRET"],
                is_mock=os.environ.get("GPTQ_KIWOOM_IS_MOCK", "true").lower() != "false",
            )
        )
        if not broker.connect():
            raise SystemExit("broker connect failed (omit --with-broker to judge only)")
        selection = load_policy(POLICY_PATH).selection
        candidate_ctx = {
            "broker": broker,
            "db_client": db_client,
            "gen": SignalGenerator(SignalConfig(min_confidence=0.0)),
            "selection": selection,
        }

    from daytrade_it.infrastructure.external.anthropic_client import AnthropicClient

    logger.warning(
        "news_sweep_shadow: starting -- competes with the live daemon's Toss access "
        "unless run from a separate IP/proxy",
        universe=len(universe),
        prev_day=str(prev_day),
        proxy=bool(args.proxy),
        broker=args.with_broker,
        max_requests=args.max_requests,
    )
    try:
        async with (
            AnthropicClient() as claude,
            httpx.AsyncClient(headers=HTTP_HEADERS, timeout=15, proxy=args.proxy) as http,
        ):
            runner = ShadowRunner(
                day=day,
                schedule=schedule,
                stocks=stocks,
                stale_before=stale_before,
                http=http,
                claude=claude,
                candidate_ctx=candidate_ctx,
                max_requests=args.max_requests,
            )
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, runner._halt, f"signal {sig.name}")
            await runner.run()
    finally:
        if broker is not None:
            broker.disconnect()
        db_client.close()


def _date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run-http", action="store_true", help="Print schedule, no HTTP")
    parser.add_argument("--max-requests", type=int, default=None, help="Cap on Toss requests")
    parser.add_argument(
        "--with-broker",
        action="store_true",
        help="Also run the shadow entry filter; uses the live Kiwoom token "
        "(quotes/minute bars only, no orders). Default: judge + archive only",
    )
    parser.add_argument("--proxy", default=None, help="httpx proxy URL for Toss requests")
    parser.add_argument(
        "--allow-trader-ip",
        action="store_true",
        help="Allow sweeping without --proxy (competes with the live daemon's Toss access)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help=f"Universe size (default: DAYTRADE_NEWS_UNIVERSE_SIZE, else {UNIVERSE_SIZE})",
    )
    parser.add_argument(
        "--universe-file",
        type=Path,
        default=None,
        help="Codes (whitespace-separated) instead of DB",
    )
    parser.add_argument("--day", type=_date, default=None, help="Session date (default today KST)")
    return parser


def main() -> None:
    asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    main()
