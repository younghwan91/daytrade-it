"""News pair → entry candidate: the live poller's rules, shared with replay.

`scripts/news_signal_poller.py` and `application/replay/day_replay.py` both
call these, so a backtest runs exactly the filter and signal build the live
daemon runs (spec 2026-09-15-bayes-pipeline-optimization-design.md).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal

from daytrade_it.application.signals.entry_filter import generate_no_chase_confirmation
from daytrade_it.domain.entities.news import NewsArticle
from daytrade_it.domain.entities.signal import TradingSignal
from daytrade_it.domain.services.signal_generator import SignalGenerator
from daytrade_it.domain.value_objects import Sentiment, SentimentLabel, Ticker
from daytrade_it.infrastructure.brokers.base import KST, BrokerInterface
from daytrade_it.infrastructure.external.anthropic_client import (
    PROMPT_VERSION,
    SIGNAL_THRESHOLD,
    ArticleFeatures,
)
from daytrade_it.infrastructure.external.quant_airflow_db_client import (
    QuantAirflowDBClient,
    StockInfo,
)
from daytrade_it.shared.logging import get_logger

logger = get_logger(__name__)

SESSIONS: tuple[str, ...] = ("premarket", "intraday")
_REGULAR_OPEN = dt.time(9, 0)


@dataclass(frozen=True)
class SelectionConfig:
    """Which judged pairs become entry candidates. Defaults = the live rules.

    - `min_score`: BUY needs `score() > min_score` (strict, as the evaluated rule).
    - `subjects` / `exclude_persistence` / `exclude_material`: v2 feature gates.
    - `min_pre_move` / `max_pre_move`: bounds on the move vs. the prior close;
      `max_pre_move=None` uses `entry_filter.MAX_PRE_ENTRY_MOVE` at call time.
    - `sessions`: `premarket` if published before 09:00 KST, else `intraday`.
    - `rank_key`: `auto_trader.rank_batch` key (surprise | persistence | score).
    """

    min_score: float = SIGNAL_THRESHOLD
    subjects: tuple[str, ...] = ("main", "partial")
    exclude_persistence: tuple[str, ...] = ()
    exclude_material: tuple[str, ...] = ()
    min_pre_move: Decimal | None = None
    max_pre_move: Decimal | None = None
    sessions: tuple[str, ...] = SESSIONS
    rank_key: str = "surprise"


def article_session(published_at: dt.datetime) -> str:
    return "premarket" if published_at.astimezone(KST).time() < _REGULAR_OPEN else "intraday"


def _selection_reject(
    article: NewsArticle, features: ArticleFeatures, score: float, selection: SelectionConfig
) -> str | None:
    if score <= selection.min_score:
        return "score"
    if features.subject not in selection.subjects:
        return "subject"
    if features.persistence in selection.exclude_persistence:
        return "persistence"
    if features.material_type in selection.exclude_material:
        return "material_type"
    if article_session(article.published_at) not in selection.sessions:
        return "session"
    return None


@dataclass(frozen=True)
class EntryCandidate:
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
    """Tagged tickers the daemon would judge: known stock, fresh article, session open."""
    # Published before the previous session closed: that session already
    # had the chance to trade it, and the eval never measured such articles.
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
    selection: SelectionConfig = SelectionConfig(),
) -> EntryCandidate | None:
    """Composite BUY for a selected, not-extended pair, else None."""
    sentiment = features.to_sentiment()
    # Entries are BUY-only: KRX retail can't short.
    if sentiment.score <= 0:
        return None
    reason = _selection_reject(article, features, sentiment.score, selection)
    if reason is not None:
        # Below the live threshold was always a silent skip; keep it silent.
        if sentiment.is_bullish:
            logger.info("news_entry: not selected", ticker=ticker.value, reason=reason)
        return None
    if not sentiment.is_bullish:
        # min_score below SIGNAL_THRESHOLD admits it: relabel so it's a BUY.
        sentiment = Sentiment(
            score=sentiment.score, label=SentimentLabel.BULLISH, confidence=sentiment.confidence
        )
    news_signal = gen.generate_from_news(
        article.model_copy(update={"tickers": [ticker]}), sentiment
    )
    if news_signal is None:
        return None
    entry_signal = generate_no_chase_confirmation(
        ticker, broker, db_client, expires_at, max_move=selection.max_pre_move
    )
    if entry_signal is None:
        return None
    if selection.min_pre_move is not None and (
        Decimal(entry_signal.metadata["pre_entry_move"]) < selection.min_pre_move
    ):
        logger.info("news_entry: not selected", ticker=ticker.value, reason="min_pre_move")
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
                "persistence": features.persistence,
                "surprise": features.surprise,
                "news_score": sentiment.score,
                "pre_entry_move": entry_signal.metadata["pre_entry_move"],
            },
        }
    )
    return EntryCandidate(composite, int(features.surprise), float(sentiment.score))
