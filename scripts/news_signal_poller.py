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

This process places NO orders itself -- it only writes signal rows.
Executing a signal is a separate, explicit `execute_trade` call (through
the MCP server, by whoever/whatever is driving it), which carries its own
hard safety rails (1-share cap, LIMIT-only, price band, kill switch,
duplicate-entry guard).

Usage:
    uv run python scripts/news_signal_poller.py                # run until 15:20 KST
    uv run python scripts/news_signal_poller.py --once          # single poll, then exit
    uv run python scripts/news_signal_poller.py --stop-at 23:59 # override stop time (testing)

Honors the same kill switch `execute_trade` does (`data/live_STOP`): if
present at startup or between polls, the poller logs and exits rather than
generating further signals -- a human stopped live trading, this process
should not keep feeding it new signals either.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kiwoom_client import KiwoomAPI
from sqlalchemy import select

from daytrade_it.application.signals.entry_filter import generate_no_chase_confirmation
from daytrade_it.domain.entities.news import NewsArticle
from daytrade_it.domain.services.signal_generator import SignalConfig, SignalGenerator
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.brokers.base import KST, BrokerInterface
from daytrade_it.infrastructure.brokers.kiwoom_broker import KiwoomBroker
from daytrade_it.infrastructure.external.anthropic_client import PROMPT_VERSION, AnthropicClient
from daytrade_it.infrastructure.external.quant_airflow_db_client import (
    QuantAirflowDBClient,
    StockInfo,
)
from daytrade_it.infrastructure.external.toss_news_client import TossNewsClient
from daytrade_it.infrastructure.persistence.database import DatabaseManager
from daytrade_it.infrastructure.persistence.models import TradingSignalModel
from daytrade_it.infrastructure.persistence.unit_of_work import UnitOfWork
from daytrade_it.shared.logging import get_logger

logger = get_logger(__name__)

KILL_SWITCH_PATH = Path("data/live_STOP")
DEFAULT_STOP_TIME = dt.time(15, 20)  # KST -- mirrors scalp-it's daemon stop time
SESSION_CLOSE = dt.time(15, 20)  # signals expire here; day trades are flat by the close
FIRST_POLL = dt.time(9, 0, 30)


def kill_switch_active() -> bool:
    return KILL_SWITCH_PATH.exists()


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
) -> None:
    """Judge one (article, company) pair; persist a BUY signal if one emerges.

    Every failure is caught and logged, never raised -- one bad pair must
    not stop the loop or the other pairs in the batch.
    """
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
    for ticker in article.tickers:
        stock = stocks.get(ticker.value)
        if stock is None:
            continue
        await process_pair(article, ticker, stock, expires_at=expires_at, **deps)  # type: ignore[arg-type]


async def run(*, once: bool, stop_at: dt.time) -> None:
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

    try:
        async with AnthropicClient() as claude, TossNewsClient() as news_client:
            while True:
                if kill_switch_active():
                    logger.warning("news_signal_poller: kill switch active, stopping")
                    break
                if not once and dt.datetime.now(KST).time() >= stop_at:
                    logger.info(
                        "news_signal_poller: stop time reached, exiting", stop_at=str(stop_at)
                    )
                    break

                try:
                    new_articles = await news_client.poll_once()
                except Exception:
                    logger.exception(
                        "news_signal_poller: poll_once failed, will retry next interval"
                    )
                    new_articles = []

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
                    )

                if once:
                    break
                await asyncio.sleep(news_client.poll_interval)
    finally:
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
        help="Stop polling at this local time, HH:MM (default 15:20)",
    )
    args = parser.parse_args()
    asyncio.run(run(once=args.once, stop_at=args.stop_at))


if __name__ == "__main__":
    main()
