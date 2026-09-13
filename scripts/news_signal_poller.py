"""Live news→signal daemon.

Polls Toss's live news dashboard, scores each new article with Claude
(`infrastructure/external/anthropic_client.py`), and -- when a technical
confirmation agrees with the news direction (`application/signals/
technical_confirmation.py`) -- persists an actionable COMPOSITE signal to
`gptquant.trading_signals` via `SignalRepository`, so `execute_trade(
signal_id=...)` can pick it up later. See 2026-09-13's session notes
(memory: daytrade-it-news-signal-pipeline) for why this shape: news alone
is deliberately not actionable (`SignalGenerator.combine_signals()`), and
a price-only ML classifier was tried and rejected (no edge) before this
simple rule-based technical check.

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

from daytrade_it.application.signals.technical_confirmation import generate_technical_confirmation
from daytrade_it.domain.entities.news import NewsArticle
from daytrade_it.domain.services.signal_generator import SignalGenerator
from daytrade_it.domain.value_objects import Sentiment
from daytrade_it.infrastructure.brokers.base import BrokerInterface
from daytrade_it.infrastructure.brokers.kiwoom_broker import KiwoomBroker
from daytrade_it.infrastructure.external.anthropic_client import AnthropicClient
from daytrade_it.infrastructure.external.quant_airflow_db_client import QuantAirflowDBClient
from daytrade_it.infrastructure.external.toss_news_client import TossNewsClient
from daytrade_it.infrastructure.persistence.database import DatabaseManager
from daytrade_it.infrastructure.persistence.models import TradingSignalModel
from daytrade_it.infrastructure.persistence.unit_of_work import UnitOfWork
from daytrade_it.shared.logging import get_logger

logger = get_logger(__name__)

KILL_SWITCH_PATH = Path("data/live_STOP")
DEFAULT_STOP_TIME = dt.time(15, 20)  # KST -- mirrors scalp-it's daemon stop time


def kill_switch_active() -> bool:
    return KILL_SWITCH_PATH.exists()


async def process_article(
    article: NewsArticle,
    claude: AnthropicClient,
    broker: BrokerInterface,
    db_client: QuantAirflowDBClient,
    gen: SignalGenerator,
    db: DatabaseManager,
) -> None:
    """Score one article and persist an actionable signal if one emerges.

    Every failure mode here is caught and logged, never raised -- one bad
    article (a Claude schema violation, a broker hiccup, a missing price
    history) must not take down the whole poll loop or block the other
    articles in the same batch.
    """
    if not article.tickers:
        return
    ticker = article.tickers[0]

    # Dedup across restarts: TossNewsClient's seen-id cache is in-memory
    # only, so a crash-restart would otherwise re-score every article on
    # the dashboard (Claude cost) and re-persist the same signals.
    try:
        async with UnitOfWork(db) as uow:
            existing = await uow.session.execute(
                select(TradingSignalModel.id).where(
                    TradingSignalModel.news_article_id == article.id
                )
            )
            if existing.first() is not None:
                return
    except Exception:
        logger.exception("news_signal_poller: dedup lookup failed", article_id=str(article.id))
        return

    try:
        sentiment: Sentiment = await claude.analyze_sentiment(article)
    except Exception:
        logger.exception("news_signal_poller: sentiment scoring failed", article_id=str(article.id))
        return

    news_signal = gen.generate_from_news(article, sentiment)
    if news_signal is None:
        return

    try:
        tech_signal = generate_technical_confirmation(ticker, broker, db_client)
    except Exception:
        logger.exception("news_signal_poller: technical confirmation failed", ticker=ticker.value)
        return
    if tech_signal is None:
        return

    # combine_signals() resolves a direction conflict by confidence and
    # drops the losing side -- so news BUY + technical SELL would yield a
    # technical-only SELL "composite" with no news behind it (seen live
    # 2026-09-13). This daemon's thesis is news-driven: the technical check
    # may only confirm the news direction, never override it.
    if tech_signal.direction != news_signal.direction:
        return

    composite = gen.combine_signals([news_signal, tech_signal])
    if composite is None:
        return
    composite = composite.model_copy(update={"news_article_id": article.id})

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
        direction=composite.direction.value,
        confidence=composite.confidence.value,
        article_title=article.title,
    )


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
    gen = SignalGenerator()

    try:
        async with AnthropicClient() as claude, TossNewsClient() as news_client:
            while True:
                if kill_switch_active():
                    logger.warning("news_signal_poller: kill switch active, stopping")
                    break
                if not once and dt.datetime.now().time() >= stop_at:
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
                    await process_article(article, claude, broker, db_client, gen, db)

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
