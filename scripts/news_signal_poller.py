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
DEFAULT_STOP_TIME = dt.time(15, 20)  # KST -- news polling stops here
#: With --execute, the process keeps managing exits until the closing auction
#: ends (KRX continuous trading ends 15:20; the auction runs to 15:30).
EXITS_UNTIL = dt.time(15, 30)
SESSION_CLOSE = dt.time(15, 20)  # signals expire here; day trades are flat by the close
FIRST_POLL = dt.time(9, 0, 30)
MANAGE_TICK_SECONDS = 30
LEDGER_DIR = Path("data/auto_trader")
JUDGMENT_DIR = Path("data/news_judgments")
DETAIL_ARCHIVE_DIR = Path("data/toss_detail")
DETAIL_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
}


def kill_switch_active() -> bool:
    return KILL_SWITCH_PATH.exists()


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
) -> None:
    """Judge one (article, company) pair; persist a BUY signal if one emerges.

    Every failure is caught and logged, never raised -- one bad pair must
    not stop the loop or the other pairs in the batch.
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
    except Exception:
        logger.exception(
            "news_signal_poller: scoring failed", article_id=str(article.id), ticker=ticker.value
        )
        return

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
    if execute:
        trader = AutoTrader(broker, LEDGER_DIR, live=confirm_real)
        logger.warning(
            "news_signal_poller: auto-trading enabled",
            live=confirm_real,
            mode="LIVE ORDERS" if confirm_real else "dry run (no --confirm-real)",
        )

    try:
        judgments = JudgmentLog(JUDGMENT_DIR, today)
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

                if kill_switch_active():
                    if not kill_logged:
                        logger.warning(
                            "news_signal_poller: kill switch active -- no new signals or "
                            "entries; exits still managed"
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


def main() -> None:
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
    args = parser.parse_args()
    if args.confirm_real and not args.execute:
        parser.error("--confirm-real requires --execute")
    asyncio.run(
        run(
            once=args.once,
            stop_at=args.stop_at,
            execute=args.execute,
            confirm_real=args.confirm_real,
            exits_until=args.exits_until,
        )
    )


if __name__ == "__main__":
    main()
