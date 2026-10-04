"""News pair → entry candidate: the live poller's rules, shared with replay.

`scripts/news_signal_poller.py` and `application/replay/day_replay.py` both
call these, so a backtest runs exactly the filter and signal build the live
daemon runs. Nothing here is a new rule: it is the poller's post-judgment
path (BUY-only, no-chase filter, composite signal) moved to one place.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from daytrade_it.application.signals.entry_filter import generate_no_chase_confirmation
from daytrade_it.domain.entities.news import NewsArticle
from daytrade_it.domain.entities.signal import TradingSignal
from daytrade_it.domain.services.signal_generator import SignalGenerator
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.brokers.base import BrokerInterface
from daytrade_it.infrastructure.external.anthropic_client import PROMPT_VERSION, ArticleFeatures
from daytrade_it.infrastructure.external.quant_airflow_db_client import (
    QuantAirflowDBClient,
    StockInfo,
)


@dataclass(frozen=True)
class EntryCandidate:
    """A persisted-signal-to-be with the two numbers `rank_batch` orders by."""

    signal: TradingSignal
    surprise: int
    score: float


def tradable_tickers(
    article: NewsArticle,
    stocks: dict[str, StockInfo],
    stale_before: dt.datetime,
    expires_at: dt.datetime,
    now: dt.datetime,
) -> list[tuple[Ticker, StockInfo]]:
    """Tagged tickers the daemon would judge: known stock, fresh article, session open.

    Mirrors `news_signal_poller.process_article`: published before the previous
    session closed means that session already had the chance to trade it.
    """
    if article.published_at < stale_before or now >= expires_at:
        return []
    return [(t, stocks[t.value]) for t in article.tickers if t.value in stocks]


def build_entry_candidate(
    article: NewsArticle,
    ticker: Ticker,
    features: ArticleFeatures,
    broker: BrokerInterface,
    db_client: QuantAirflowDBClient,
    gen: SignalGenerator,
    expires_at: dt.datetime,
) -> EntryCandidate | None:
    """Composite BUY for a bullish, not-extended pair, else None.

    Broker/DB errors from the no-chase filter propagate: the caller logs them
    (`news_signal_poller: entry filter failed` live, `replay: entry filter
    failed` in replay) and treats the pair as no entry.
    """
    sentiment = features.to_sentiment()
    # Entries are BUY-only: KRX retail can't short, and no SELL-side rule
    # was evaluated (dev had n=6 bearish main-subject pairs).
    if not sentiment.is_bullish:
        return None
    news_signal = gen.generate_from_news(
        article.model_copy(update={"tickers": [ticker]}), sentiment
    )
    if news_signal is None:
        return None
    entry_signal = generate_no_chase_confirmation(ticker, broker, db_client, expires_at)
    if entry_signal is None:
        return None
    composite = gen.combine_signals([news_signal, entry_signal])
    if composite is None:
        return None
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
    return EntryCandidate(composite, int(features.surprise), float(sentiment.score))
