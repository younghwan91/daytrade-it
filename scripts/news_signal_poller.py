"""Live news→signal daemon.

Polls Toss's live news dashboard and, for every (article, tagged company)
pair, scores it with Claude (`infrastructure/external/anthropic_client.py`,
prompt v2). A bullish main/partial-subject pair whose stock is not already
up > 2% vs. the prior close (`application/signals/entry_filter.py`) becomes
a COMPOSITE BUY signal in `gptquant.trading_signals`, expiring at that
session's close, for `execute_trade(signal_id=...)` to pick up.

Every rule here was measured against real same-session outcomes before
being adopted -- see docs/research/2026-09-13-news-prompt-eval.md.
Articles published before the previous session's close are skipped (the
eval only labeled first-session reactions), and nothing is polled before
09:00:30 KST (the eval entered pre-market news at the opening price).

With `--execute`, persisted signals go to `AutoTrader` (1 share, flat by
the close); orders are real only with `--confirm-real` as well.

Every judged pair (not only signals) is appended to
`data/news_judgments/YYYY-MM-DD.jsonl`, together with a shadow judgment
from prompt v4 (full article body, fetched from Toss's detail endpoint and
archived under `data/toss_detail/`). v4 never affects signals; it is there
for a forward comparison with v2.

Usage:
    uv run python scripts/news_signal_poller.py                          # signals only
    uv run python scripts/news_signal_poller.py --execute                # + dry-run trading
    uv run python scripts/news_signal_poller.py --execute --confirm-real # + REAL orders
    uv run python scripts/news_signal_poller.py --once                   # single poll
    uv run python scripts/news_signal_poller.py --execute --paper        # + PAPER orders

Paper mode (`--execute --paper`, never with `--confirm-real`): quotes and
minute bars still come from the real Kiwoom API (read-only), but orders go
to `CorePaperBroker` -- krx-quant-core's `PaperBroker`, filled against those
real quotes -- and AutoTrader writes its ledger to `data/auto_trader_paper/`,
never `data/auto_trader/`. Paper cash starts at 10,000,000 KRW. Pairs are
judged exactly as in live mode (same prefilter, same Claude calls), but
nothing is shared with live dedup state: judgments go to
`data/news_judgments_paper/` and signals are NOT persisted to the gptquant DB
(dedup by (article, ticker) is in memory, plus the paper judgment log across
restarts). Not in cron.

Single instance per mode: `run()` holds `<repo>/data/auto_trader/poller.lock`
(live/dry-run) or `<repo>/data/auto_trader_paper/poller.lock` (`--paper`)
(krx-quant-core `InstanceLock`, anchored to the repo root, not the cwd) for its
whole life; a second poller of the same mode logs an ERROR with the holder pid
and returns (exit 0) before building any client. Paper and live never block
each other. `news_signal_daemon.sh` keeps its own flock too.

⚠️ 장중 trader 에서 live poller(`--paper` 없이)를 손으로 띄우지 말 것 — 실데몬이
하루 종일 쉰다. (A hand-started non-paper poller holds the live lock: the cron
daemon's restarts all exit on it and live trading, exits included, idles for the
rest of the day.) `--paper` uses its own lock and is independent of the live
daemon, but it still calls Claude for every judged pair -- it spends the same
Anthropic credits the live daemon needs.

Kill switch `data/live_STOP`: stops new signals and entries; exits keep
being managed until the close.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx
from kiwoom_client import KiwoomAPI
from krx_quant_core.execution.oms import AlreadyRunning, InstanceLock
from sqlalchemy import select

from daytrade_it.application.news.mention_prefilter import skip_reason
from daytrade_it.application.signals.entry_filter import generate_no_chase_confirmation
from daytrade_it.application.trading.auto_trader import AutoTrader, rank_batch
from daytrade_it.domain.entities.news import NewsArticle
from daytrade_it.domain.services.signal_generator import SignalConfig, SignalGenerator
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.brokers.base import KST, BrokerInterface
from daytrade_it.infrastructure.brokers.kiwoom_broker import KiwoomBroker
from daytrade_it.infrastructure.external.anthropic_client import (
    PROMPT_VERSION,
    AnthropicClient,
    ArticleFeatures,
)
from daytrade_it.infrastructure.external.quant_airflow_db_client import (
    QuantAirflowDBClient,
    StockInfo,
)
from daytrade_it.infrastructure.external.toss_news_client import (
    TossNewsClient,
    extract_full_body,
    fetch_article_detail,
)
from daytrade_it.infrastructure.persistence.database import DatabaseManager
from daytrade_it.infrastructure.persistence.models import TradingSignalModel
from daytrade_it.infrastructure.persistence.unit_of_work import UnitOfWork
from daytrade_it.shared.logging import get_logger

logger = get_logger(__name__)

KILL_SWITCH_PATH = Path("data/live_STOP")
#: Where `--paper` writes the consecutive-scoring-failure stop instead -- paper
#: mode must never write the live kill file (it would block tomorrow's entries).
PAPER_KILL_SWITCH_PATH = Path("data/paper_STOP")
DEFAULT_STOP_TIME = dt.time(15, 20)  # KST -- news polling stops here
#: With --execute, the process keeps managing exits until the closing auction
#: ends (KRX continuous trading ends 15:20; the auction runs to 15:30).
EXITS_UNTIL = dt.time(15, 30)
SESSION_CLOSE = dt.time(15, 20)  # signals expire here; day trades are flat by the close
FIRST_POLL = dt.time(9, 0, 30)
MANAGE_TICK_SECONDS = 30
LEDGER_DIR = Path("data/auto_trader")
PAPER_LEDGER_DIR = Path("data/auto_trader_paper")
JUDGMENT_DIR_PAPER = Path("data/news_judgments_paper")
#: Anchored to the repo root (not the cwd) so a poller started by hand from
#: another directory still collides with the cron daemon's lock.
POLLER_LOCK_PATH = Path(__file__).resolve().parents[1] / "data" / "auto_trader" / "poller.lock"
#: `--paper` shares no state with live (own ledger/judgments, no DB writes,
#: paper_STOP), so it takes its own lock -- a paper run never idles the live daemon.
PAPER_POLLER_LOCK_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "auto_trader_paper" / "poller.lock"
)
JUDGMENT_DIR = Path("data/news_judgments")
DETAIL_ARCHIVE_DIR = Path("data/toss_detail")
DETAIL_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
}


def kill_switch_active() -> bool:
    return KILL_SWITCH_PATH.exists()


#: scalp-it 수집기가 60 초마다 다시 읽는 핀 목록. 저장 전용이라 그쪽 매매 판단이나
#: 주문 화이트리스트에는 닿지 않고(그쪽 테스트로 고정), 파일이 없거나 깨져도
#: 그쪽은 경고만 찍고 지나간다. 우리 쪽 실패도 매매를 막아서는 안 되므로 모든
#: 예외를 삼킨다 -- 체결 데이터 수집은 있으면 좋은 것이지 매매의 전제가 아니다.
TICK_PIN_PATH = Path.home() / "git" / "scalp-it" / "data" / "collect_pins_extra.txt"

#: 한 번에 핀할 수 있는 종목 수. 키움 동시 구독 상한이 100 이고 그 자리를
#: scalp-it 실매매와 나눠 쓴다. 우리 매수 판정은 하루 70 건 전후라 전부 핀하면
#: 상한을 혼자 다 먹는다 -- 실제로 09-17 에는 10 시부터 커버리지가 13% 로
#: 무너졌다. 최신 30 개면 그날 후반 뉴스가 들어갈 자리가 남는다.
TICK_PIN_MAX = 30

#: 삽입 순서를 유지해야 오래된 것부터 버릴 수 있어 set 이 아니라 list 다.
_pinned_codes: list[str] = []
_pin_day: dt.date | None = None


def _pin_for_tick_collection(code: str) -> None:
    """매수 판정이 난 종목을 수집기 핀 목록에 넣는다. 하루 단위로 새로 쓴다.

    **덧붙이기만 하면 스스로를 막는다.** 키움 실시간 구독은 동시 100 종목이
    상한이고 그 자리를 scalp-it 실매매와 나눠 쓴다. 09-17 에 70 개를 핀했더니
    09-18 에는 그 70 개가 자리를 잡고 있어 새 뉴스 종목이 못 들어갔다 --
    판정 이후 호가 커버리지가 31% -> 23% 로 떨어졌다. 어제 관심 종목이 오늘
    관심 종목을 막는 구조였다.

    그래서 날짜가 바뀌면 목록을 비우고, 한 번에 `TICK_PIN_MAX` 개까지만 둔다.
    상한을 넘으면 **오래된 것부터** 버린다 -- 뉴스는 최신일수록 값이 있고,
    자리가 없으면 새 종목이 아예 못 들어오기 때문이다.
    """
    global _pin_day
    today = dt.datetime.now(KST).date()
    if _pin_day != today:
        _pin_day = today
        _pinned_codes.clear()
    if code in _pinned_codes:
        return
    _pinned_codes.append(code)
    del _pinned_codes[:-TICK_PIN_MAX]  # 오래된 것부터 버린다
    try:
        if TICK_PIN_PATH.parent.is_dir():
            TICK_PIN_PATH.write_text(
                f"# daytrade-it {today.isoformat()} 매수 판정 종목 (매일 새로 씀)\n"
                + "".join(f"{c}\n" for c in _pinned_codes)
            )
    except Exception:  # noqa: BLE001 -- 수집 실패가 매매를 멈춰서는 안 된다
        logger.warning("news_signal_poller: tick pin write failed", code=code)


#: 판정 연속 실패가 이만큼 쌓이면 신규 진입을 자동으로 멈춘다. 한 번의 일시적
#: 오류로 하루를 날리지 않을 만큼 크고, 크레딧 소진·키 만료처럼 계속되는 고장을
#: 장 초반에 잡을 만큼 작은 값.
SCORING_FAILURE_LIMIT = 10

_scoring_failures = 0


class JudgmentLog:
    """Every (article, ticker) judgment, one JSONL per day.

    Kept for all pairs -- not just the ones that became signals -- so a
    later forward evaluation of v2 vs shadow v4 isn't biased toward what the
    live rule selected. Also dedups re-judging after a restart.
    """

    def __init__(self, directory: Path, day: dt.date) -> None:
        self._path = directory / f"{day.isoformat()}.jsonl"
        self._keys: set[str] = set()
        if self._path.exists():
            for line in self._path.open():
                self._keys.add(json.loads(line)["key"])

    def seen(self, key: str) -> bool:
        return key in self._keys

    def append(self, key: str, record: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as f:
            f.write(json.dumps({"key": key, **record}, ensure_ascii=False, default=str) + "\n")
        self._keys.add(key)


def session_close_kst(day: dt.date) -> dt.datetime:
    return dt.datetime.combine(day, SESSION_CLOSE, tzinfo=KST)


async def process_pair(
    article: NewsArticle,
    ticker: Ticker,
    stock: StockInfo,
    claude: AnthropicClient,
    broker: BrokerInterface,
    db_client: QuantAirflowDBClient,
    gen: SignalGenerator,
    db: DatabaseManager,
    expires_at: dt.datetime,
    trader: AutoTrader | None,
    judgments: JudgmentLog,
    full_body_text: str | None,
    pending: list | None = None,
    paper_seen: set[str] | None = None,
) -> None:
    """Judge one (article, company) pair; persist a BUY signal if one emerges.

    Every failure is caught and logged, never raised -- one bad pair must
    not stop the loop or the other pairs in the batch.

    `paper_seen` (paper mode only): dedup in this set instead of the signal DB,
    and never persist -- live's dedup state must not see paper signals.
    """
    pair_key = f"{article.id}|{ticker.value}"
    if judgments.seen(pair_key):
        return

    # 판정 전에 버릴 쌍은 버린다 -- 시황 묶음·표·지수 해설은 97,721 건에서 매수
    # 후보가 거의 나오지 않았다. 제목과 관련 종목 수만 보므로 백테스트와 같은
    # 판단을 한다(본문은 두 세계가 서로 다르다).
    reason = skip_reason(article.title, len(article.tickers))
    if reason is not None:
        logger.info(
            "news_signal_poller: prefilter skipped",
            ticker=ticker.value,
            name=stock.name,
            reason=reason,
            related_n=len(article.tickers),
            title=(article.title or "")[:80],
        )
        # 판정한 쌍과 같은 형식으로 남긴다 -- 이 아카이브를 재생하는 백테가
        # 운영이 실제로 판정한 풀을 그대로 보게.
        judgments.append(
            pair_key,
            {
                "article_id": str(article.id),
                "ticker": ticker.value,
                "name": stock.name,
                "title": article.title,
                "category": article.category,
                "published_at": article.published_at.isoformat(),
                "judged_at": dt.datetime.now(KST).isoformat(),
                "full_body_chars": 0,
                "prefilter_skipped": True,
                "prefilter_reason": reason,
                "prefilter_tag": (
                    reason.split(":", 1)[1] if reason.startswith("roundup:") else None
                ),
                "v2": None,
                "v4_shadow": None,
            },
        )
        return

    # Dedup across restarts (TossNewsClient's seen-id cache is in-memory).
    if paper_seen is not None:
        if pair_key in paper_seen:
            return
    else:
        try:
            async with UnitOfWork(db) as uow:
                existing = await uow.session.execute(
                    select(TradingSignalModel.id).where(
                        TradingSignalModel.news_article_id == article.id,
                        TradingSignalModel.ticker == ticker.value,
                    )
                )
                if existing.first() is not None:
                    return
        except Exception:
            logger.exception("news_signal_poller: dedup lookup failed", article_id=str(article.id))
            return

    try:
        features = await claude.score_article(
            code=ticker.value,
            name=stock.name,
            market=stock.market,
            sector=stock.sector,
            title=article.title,
            body=article.summary or article.content,
            category=article.category,
        )
    except Exception as exc:
        global _scoring_failures
        _scoring_failures += 1
        logger.exception(
            "news_signal_poller: scoring failed",
            article_id=str(article.id),
            ticker=ticker.value,
            consecutive=_scoring_failures,
        )
        if _scoring_failures == SCORING_FAILURE_LIMIT:
            # 판정을 못 하면 뉴스 매매의 근거가 없다. 09-16 에는 크레딧이 떨어져
            # 09:07~14:58 동안 86 건이 실패했는데도 데몬이 조용히 계속 돌았고,
            # 아무도 장이 끝날 때까지 몰랐다. 신규 진입만 막고 청산은 계속 돈다.
            stop_path = KILL_SWITCH_PATH if paper_seen is None else PAPER_KILL_SWITCH_PATH
            stop_path.parent.mkdir(parents=True, exist_ok=True)
            stop_path.write_text(
                f"{dt.datetime.now(KST).isoformat()} 판정 연속 실패 "
                f"{SCORING_FAILURE_LIMIT} 회로 신규 진입을 자동 중단한다.\n"
                f"마지막 오류: {type(exc).__name__}: {exc}\n"
                "원인을 고친 뒤 이 파일을 지우면 재개된다.\n"
            )
            logger.critical(
                "news_signal_poller: 판정 연속 실패 -- 신규 진입 중단",
                consecutive=_scoring_failures,
                error=f"{type(exc).__name__}: {exc}",
            )
        return

    globals()["_scoring_failures"] = 0  # 연속 실패만 센다 -- 한 번 성공하면 초기화.

    sentiment = features.to_sentiment()
    logger.info(
        "news_signal_poller: judged",
        ticker=ticker.value,
        name=stock.name,
        subject=features.subject,
        score=round(sentiment.score, 3),
        label=sentiment.label.value,
        title=article.title[:80],
    )

    # Shadow v4 (full body): recorded for a forward comparison, never used
    # for signals. See anthropic_client._USER_TEMPLATE_FULL_BODY.
    shadow: ArticleFeatures | None = None
    if full_body_text:
        try:
            shadow = await claude.score_article(
                code=ticker.value,
                name=stock.name,
                market=stock.market,
                sector=stock.sector,
                title=article.title,
                body=full_body_text,
                category=article.category,
                full_body=True,
            )
        except Exception:
            logger.exception("news_signal_poller: shadow v4 scoring failed", ticker=ticker.value)

    judgments.append(
        pair_key,
        {
            "article_id": str(article.id),
            "ticker": ticker.value,
            "name": stock.name,
            "title": article.title,
            "category": article.category,
            "published_at": article.published_at.isoformat(),
            "judged_at": dt.datetime.now(KST).isoformat(),
            "full_body_chars": len(full_body_text or ""),
            "v2": {**asdict(features), "score": features.score()},
            "v4_shadow": {**asdict(shadow), "score": shadow.score()} if shadow else None,
        },
    )
    # Entries are BUY-only: KRX retail can't short, and no SELL-side rule
    # was evaluated (dev had n=6 bearish main-subject pairs).
    if not sentiment.is_bullish:
        return

    news_signal = gen.generate_from_news(
        article.model_copy(update={"tickers": [ticker]}), sentiment
    )
    if news_signal is None:
        return

    # 매수 판정이 난 종목을 scalp-it 수집기에 넘겨 체결·호가를 쌓게 한다.
    # 지금 `ticks`/`quotes` 는 우리가 실제로 산 종목의 21.4% 만 덮고 있어,
    # 체결강도·호가 잔량이 이 프로젝트에서 유일하게 검증하지 못한 가설로
    # 남아 있다. 하루 약 20 종목이고 수집기는 60 초마다 이 파일을 다시 읽는다.
    _pin_for_tick_collection(ticker.value)

    try:
        entry_signal = generate_no_chase_confirmation(ticker, broker, db_client, expires_at)
    except Exception:
        logger.exception("news_signal_poller: entry filter failed", ticker=ticker.value)
        return
    if entry_signal is None:
        return

    composite = gen.combine_signals([news_signal, entry_signal])
    if composite is None:
        return
    composite = composite.model_copy(
        update={
            "news_article_id": article.id,
            "expires_at": expires_at,
            "metadata": {
                **composite.metadata,
                "prompt_version": PROMPT_VERSION,
                "article_title": article.title[:200],
                "subject": features.subject,
                "material_type": features.material_type,
                "surprise": features.surprise,
                "news_score": sentiment.score,
                "pre_entry_move": entry_signal.metadata["pre_entry_move"],
            },
        }
    )

    if paper_seen is not None:
        paper_seen.add(pair_key)
        saved = composite  # in memory only; AutoTrader looks it up by its id
    else:
        try:
            async with UnitOfWork(db) as uow:
                saved = await uow.signals.add(composite)
                await uow.commit()
        except Exception:
            logger.exception("news_signal_poller: failed to persist signal", ticker=ticker.value)
            return

    logger.info(
        "news_signal_poller: ACTIONABLE SIGNAL PERSISTED",
        signal_id=str(saved.id),
        ticker=ticker.value,
        name=stock.name,
        direction=composite.direction.value,
        news_score=round(sentiment.score, 3),
        pre_entry_move=entry_signal.metadata["pre_entry_move"],
        article_title=article.title,
    )

    if trader is not None:
        if pending is not None:
            # Entered after the whole poll is judged, in rank_batch order.
            pending.append((saved, int(features.surprise), float(sentiment.score)))
            return
        try:
            await trader.enter(saved)
        except Exception:
            logger.exception("news_signal_poller: trader.enter failed", ticker=ticker.value)


async def enter_batch(trader: AutoTrader, pending: list) -> None:
    """Enter one poll's signals in `rank_batch` order; one failure never stops the rest."""
    ranked = rank_batch(pending)
    logger.info(
        "news_signal_poller: entering batch by surprise", order=[s.ticker.value for s in ranked]
    )
    for signal in ranked:
        try:
            await trader.enter(signal)
        except Exception:
            logger.exception("news_signal_poller: trader.enter failed", ticker=signal.ticker.value)


async def process_article(
    article: NewsArticle,
    stocks: dict[str, StockInfo],
    stale_before: dt.datetime,
    expires_at: dt.datetime,
    **deps: object,
) -> None:
    # Published before the previous session closed: that session already
    # had the chance to trade it, and the eval never measured such articles
    # (every labeled pair traded on its first session after publication).
    if article.published_at < stale_before:
        return
    if dt.datetime.now(KST) >= expires_at:
        return
    known = [(t, stocks[t.value]) for t in article.tickers if t.value in stocks]
    if not known:
        return

    full_body_text: str | None = None
    http = deps.pop("http")
    detail = await fetch_article_detail(http, str(article.url))  # type: ignore[arg-type]
    if detail is not None:
        day_dir = DETAIL_ARCHIVE_DIR / dt.datetime.now(KST).date().isoformat()
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / f"{article.id}.json").write_text(json.dumps(detail, ensure_ascii=False))
        full_body_text = extract_full_body(detail) or None

    for ticker, stock in known:
        await process_pair(
            article,
            ticker,
            stock,
            expires_at=expires_at,
            full_body_text=full_body_text,
            **deps,  # type: ignore[arg-type]
        )


async def run(
    *,
    once: bool,
    stop_at: dt.time,
    execute: bool,
    confirm_real: bool,
    exits_until: dt.time = EXITS_UNTIL,
    paper: bool = False,
) -> None:
    """Hold this mode's single-instance lock, then poll/trade until the stop time."""
    lock_path = PAPER_POLLER_LOCK_PATH if paper else POLLER_LOCK_PATH
    lock = InstanceLock(lock_path)
    try:
        lock.acquire()
    except AlreadyRunning:
        try:
            holder_pid = lock_path.read_text().strip() or "unknown"
        except OSError:
            holder_pid = "unknown"
        # ERROR, not WARNING: if the holder is a hand-started poller, the cron
        # daemon's restart loop keeps landing here and live trading idles all day.
        logger.error(
            "news_signal_poller: another poller holds the instance lock -- exiting "
            "(if that is a hand-started poller during market hours, the live daemon is idle)",
            lock=str(lock_path),
            holder_pid=holder_pid,
        )
        return
    try:
        await _run_locked(
            once=once,
            stop_at=stop_at,
            execute=execute,
            confirm_real=confirm_real,
            exits_until=exits_until,
            paper=paper,
        )
    finally:
        lock.release()


async def _run_locked(
    *,
    once: bool,
    stop_at: dt.time,
    execute: bool,
    confirm_real: bool,
    exits_until: dt.time,
    paper: bool,
) -> None:
    db = DatabaseManager(database_url=os.environ["GPTQ_DATABASE_URL"])
    api = KiwoomAPI(
        app_key=os.environ["KIWOOM_APP_KEY"],
        app_secret=os.environ["KIWOOM_APP_SECRET"],
        is_mock=os.environ.get("GPTQ_KIWOOM_IS_MOCK", "true").lower() != "false",
    )
    broker = KiwoomBroker(api=api)
    if not broker.connect():
        logger.error("news_signal_poller: broker connect failed, exiting")
        return

    db_client = QuantAirflowDBClient()
    stocks = db_client.get_stock_master()
    # min_confidence=0: the BUY set must be exactly score > SIGNAL_THRESHOLD,
    # the rule that was evaluated (see anthropic_client.ArticleFeatures).
    gen = SignalGenerator(SignalConfig(min_confidence=0.0))

    today = dt.date.today()
    prior_bars = db_client.get_adjusted_daily_bars(
        Ticker(value="005930"),
        start=today - dt.timedelta(days=14),
        end=today - dt.timedelta(days=1),
    )
    if not prior_bars:
        logger.error("news_signal_poller: cannot determine previous trading day, exiting")
        broker.disconnect()
        return
    stale_before = session_close_kst(prior_bars[-1].date)
    expires_at = session_close_kst(today)

    # Articles are marked seen when polled. Polling before the open would
    # burn them on a pre-open quote (the eval entered pre-market news at the
    # opening price), so wait for the opening auction to print.
    if not once:
        first_poll_at = dt.datetime.combine(today, FIRST_POLL, tzinfo=KST)
        wait = (first_poll_at - dt.datetime.now(KST)).total_seconds()
        if wait > 0:
            logger.info("news_signal_poller: waiting for market open", seconds=int(wait))
            await asyncio.sleep(wait)

    trader: AutoTrader | None = None
    if execute and paper:
        # Imported here so the live poller never loads the paper engine.
        from daytrade_it.infrastructure.brokers.core_paper_broker import (  # noqa: PLC0415
            CorePaperBroker,
        )

        # 종이매매는 `data/paper_STOP` 만 따른다. 기본값은 실매매 스위치
        # `data/live_STOP` 이라, 실매매를 멈춰 두면 종이 진입까지 전부 거부됐다
        # -- 09-17·09-18 이틀 동안 실제로 74 건이 그렇게 거부돼, 판정 기록은
        # 쌓였는데 "무엇을 샀을 것인가"는 한 건도 안 남았다. 폴러 루프는 이미
        # 모드별로 나뉘어 있었고(위 `stop_now`) 주문 경로만 안 따르고 있었다.
        # 프로세스당 한 모드이고 잠금이 두 폴러를 막으므로 전역 치환이 안전하다.
        from daytrade_it.interfaces.mcp import tools as _mcp_tools  # noqa: PLC0415

        _mcp_tools._KILL_SWITCH_PATH = PAPER_KILL_SWITCH_PATH

        # live=True: AutoTrader places nothing when live=False. Orders go to
        # the in-process paper broker; quotes stay real (read-only).
        trader = AutoTrader(CorePaperBroker(broker), PAPER_LEDGER_DIR, live=True)
        logger.warning(
            "news_signal_poller: auto-trading enabled",
            live=False,
            mode="PAPER (krx-quant-core PaperBroker on real quotes)",
            ledger=str(PAPER_LEDGER_DIR),
        )
    elif execute:
        trader = AutoTrader(broker, LEDGER_DIR, live=confirm_real)
        logger.warning(
            "news_signal_poller: auto-trading enabled",
            live=confirm_real,
            mode="LIVE ORDERS" if confirm_real else "dry run (no --confirm-real)",
        )

    try:
        judgments = JudgmentLog(JUDGMENT_DIR_PAPER if paper else JUDGMENT_DIR, today)
        paper_seen: set[str] | None = set() if paper else None
        async with (
            AnthropicClient() as claude,
            TossNewsClient() as news_client,
            httpx.AsyncClient(headers=DETAIL_HTTP_HEADERS, timeout=15) as http,
        ):
            next_news_poll = dt.datetime.now(KST)
            kill_logged = False
            while True:
                now = dt.datetime.now(KST)
                # No trader: nothing to manage after news stops. With a
                # trader, keep managing exits through the closing auction --
                # stopping at 15:20 left an unfilled last sell held overnight.
                exit_at = stop_at if trader is None else max(stop_at, exits_until)
                if not once and now.time() >= exit_at:
                    logger.info("news_signal_poller: exiting", at=str(exit_at))
                    break
                news_open = once or now.time() < stop_at

                # Exits keep running even with the kill switch on: stopping
                # here would leave the day's positions held overnight.
                if trader is not None:
                    try:
                        trader.manage()
                    except Exception:
                        logger.exception("news_signal_poller: trader.manage failed")

                # 모드별 스위치를 본다. 종이매매는 `data/paper_STOP` 만 따르므로,
                # 실매매가 `data/live_STOP` 으로 멈춰 있는 동안에도 판정·기록은
                # 계속 쌓인다. 그 기록이 지금 이 프로젝트의 병목이다 -- 신뢰할 수
                # 있는 아카이브 날짜가 09-14, 09-15 이틀뿐이라 모든 판단이 거기서
                # 막힌다. 멈춘 날을 데이터 없는 날로 버리지 않기 위한 구분이다.
                stop_now = (
                    kill_switch_active() if paper_seen is None else PAPER_KILL_SWITCH_PATH.exists()
                )
                if stop_now:
                    if not kill_logged:
                        logger.warning(
                            "news_signal_poller: kill switch active -- no new signals or "
                            "entries; exits still managed",
                            switch=str(
                                KILL_SWITCH_PATH if paper_seen is None else PAPER_KILL_SWITCH_PATH
                            ),
                        )
                        kill_logged = True
                elif news_open and now >= next_news_poll:
                    next_news_poll = now + dt.timedelta(seconds=news_client.poll_interval)
                    try:
                        new_articles = await news_client.poll_once()
                    except Exception:
                        logger.exception(
                            "news_signal_poller: poll_once failed, will retry next interval"
                        )
                        new_articles = []

                    pending: list = []
                    for article in new_articles:
                        await process_article(
                            article,
                            stocks,
                            stale_before,
                            expires_at,
                            claude=claude,
                            broker=broker,
                            db_client=db_client,
                            gen=gen,
                            db=db,
                            trader=trader,
                            judgments=judgments,
                            http=http,
                            pending=pending,
                            paper_seen=paper_seen,
                        )
                    if trader is not None and pending:
                        await enter_batch(trader, pending)

                if once:
                    break
                await asyncio.sleep(MANAGE_TICK_SECONDS)
    finally:
        if trader is not None and not once and dt.datetime.now(KST).time() >= exits_until:
            try:
                trader.report_leftovers()
            except Exception:
                logger.exception("news_signal_poller: leftover position report failed")
        broker.disconnect()
        await db.engine.dispose()


def _parse_time(value: str) -> dt.time:
    hour, minute = value.split(":")
    return dt.time(int(hour), int(minute))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Poll once and exit (for testing/cron)")
    parser.add_argument(
        "--stop-at",
        type=_parse_time,
        default=DEFAULT_STOP_TIME,
        help="Stop news polling at this local time, HH:MM (default 15:20)",
    )
    parser.add_argument(
        "--exits-until",
        type=_parse_time,
        default=EXITS_UNTIL,
        help="With --execute, keep managing exits until HH:MM (default 15:30)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Hand persisted BUY signals to AutoTrader (dry run unless --confirm-real)",
    )
    parser.add_argument(
        "--confirm-real",
        action="store_true",
        help="With --execute: send REAL orders to the live account",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="With --execute: paper orders on real quotes (ledger data/auto_trader_paper)",
    )
    return parser


def parse_args(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> argparse.Namespace:
    args = parser.parse_args(argv)
    if args.confirm_real and not args.execute:
        parser.error("--confirm-real requires --execute")
    if args.paper and args.confirm_real:
        parser.error("--paper and --confirm-real are mutually exclusive")
    if args.paper and not args.execute:
        parser.error("--paper requires --execute")
    return args


def main() -> None:
    args = parse_args(build_parser())
    asyncio.run(
        run(
            once=args.once,
            stop_at=args.stop_at,
            execute=args.execute,
            confirm_real=args.confirm_real,
            exits_until=args.exits_until,
            paper=args.paper,
        )
    )


if __name__ == "__main__":
    main()
