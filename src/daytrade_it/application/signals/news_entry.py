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
_MARKET_TICKER = "069500"  # KODEX 200
_MARKET_LOOKBACK_DAYS = 10


@dataclass(frozen=True)
class SelectionConfig:
    """Which judged pairs become entry candidates. Defaults = the live rules.

    - `min_score`: BUY needs `score() > min_score` (strict, as the evaluated rule).
    - `subjects` / `exclude_persistence` / `exclude_material`: v2 feature gates.
    - `min_pre_move` / `max_pre_move`: bounds on the move vs. the prior close;
      `max_pre_move=None` uses `entry_filter.MAX_PRE_ENTRY_MOVE` at call time.
    - `sessions`: `premarket` if published before 09:00 KST of the trading day
      (including the previous evening), else `intraday`.
    - `rank_key`: `auto_trader.rank_batch` key (surprise | persistence | score | bandit).
    - `bandit_model`: path to a `trading.bandit` model file; None (default) =
      no bandit, no extra broker calls. The policy loader resolves a relative
      path against the policy file's directory.
    """

    min_score: float = SIGNAL_THRESHOLD
    subjects: tuple[str, ...] = ("main", "partial")
    exclude_persistence: tuple[str, ...] = ()
    exclude_material: tuple[str, ...] = ()
    min_pre_move: Decimal | None = None
    max_pre_move: Decimal | None = None
    sessions: tuple[str, ...] = SESSIONS
    rank_key: str = "surprise"
    bandit_model: str | None = None


def article_session(published_at: dt.datetime, expires_at: dt.datetime) -> str:
    """`premarket` iff published before 09:00 KST on `expires_at`'s trading day.

    The previous evening's articles (after the prior close) count as premarket.
    A naive `published_at` is taken to be KST.
    """
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=KST)
    open_at = dt.datetime.combine(expires_at.astimezone(KST).date(), _REGULAR_OPEN, tzinfo=KST)
    return "premarket" if published_at < open_at else "intraday"


def _selection_reject(
    article: NewsArticle,
    features: ArticleFeatures,
    score: float,
    selection: SelectionConfig,
    expires_at: dt.datetime,
) -> str | None:
    if score <= selection.min_score:
        return "score"
    if features.subject not in selection.subjects:
        return "subject"
    if features.persistence in selection.exclude_persistence:
        return "persistence"
    if features.material_type in selection.exclude_material:
        return "material_type"
    if article_session(article.published_at, expires_at) not in selection.sessions:
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
    reason = _selection_reject(article, features, sentiment.score, selection, expires_at)
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
    bandit_meta: dict[str, float] = {}
    if selection.bandit_model is not None:
        verdict = _bandit_verdict(
            article,
            ticker,
            features,
            broker,
            db_client,
            expires_at,
            selection.bandit_model,
            float(Decimal(entry_signal.metadata["pre_entry_move"])),
        )
        if verdict is None:
            return None
        bandit_meta = verdict
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
                **bandit_meta,
            },
        }
    )
    return EntryCandidate(composite, int(features.surprise), float(sentiment.score))


#: minute (naive KST) → (069500 price, 069500 prior close); one entry, one fetch per poll minute.
_MARKET_CACHE: dict[dt.datetime, tuple[Decimal | None, Decimal | None]] = {}


def _market_snapshot(
    broker: BrokerInterface, db_client: QuantAirflowDBClient, now: dt.datetime
) -> tuple[Decimal | None, Decimal | None]:
    """(069500 price, prior close), fetched once per minute and shared by a batch.

    A failed or empty fetch is cached as None for that minute too, so a broker
    outage costs one call per poll, not one per candidate (Kiwoom token is shared).
    """
    minute = now.astimezone(KST).replace(tzinfo=None, second=0, microsecond=0)
    cached = _MARKET_CACHE.get(minute)
    if cached is not None:
        return cached
    price: Decimal | None = None
    prev_close: Decimal | None = None
    try:
        quote = broker.get_quote(_MARKET_TICKER)
        last = quote.last_price or quote.mid_price
        price = last if last and last > 0 else None
        today = now.astimezone(KST).date()
        daily = db_client.get_adjusted_daily_bars(
            Ticker(value=_MARKET_TICKER),
            start=today - dt.timedelta(days=_MARKET_LOOKBACK_DAYS),
            end=today - dt.timedelta(days=1),
        )
        if daily and daily[-1].close:
            prev_close = Decimal(str(daily[-1].close))
    except Exception as exc:
        logger.warning("news_entry: market snapshot failed", error=f"{type(exc).__name__}: {exc}")
    _MARKET_CACHE.clear()
    _MARKET_CACHE[minute] = (price, prev_close)
    return price, prev_close


def _bandit_verdict(
    article: NewsArticle,
    ticker: Ticker,
    features: ArticleFeatures,
    broker: BrokerInterface,
    db_client: QuantAirflowDBClient,
    expires_at: dt.datetime,
    model_path: str,
    pre_entry_move: float,
) -> dict[str, float] | None:
    """{"bandit_m", "bandit_p"} if the bandit enters, else None.

    Fail-closed: a model-load or bar-fetch error, a missing 069500 price or
    prior close, or fewer than 5 completed bars skips the candidate (logged).
    """
    from daytrade_it.application.trading import bandit

    def skip(reason: str, **extra: object) -> None:
        logger.info("news_entry: bandit skip", ticker=ticker.value, reason=reason, **extra)

    try:
        model = bandit.load_model(model_path)
        decided_at = dt.datetime.now(KST)
        bars = broker.get_minute_bars(ticker.value)
    except Exception as exc:
        skip("error", error=f"{type(exc).__name__}: {exc}")
        return None
    mkt_price, mkt_prev_close = _market_snapshot(broker, db_client, decided_at)
    if mkt_price is None or mkt_prev_close is None or mkt_prev_close <= 0:
        skip("market data unavailable")
        return None
    if len(bandit.completed_bars(bars, decided_at)) < bandit.RET_BARS:
        skip("insufficient bars")
        return None
    try:
        x = bandit.features_for(
            features,
            pre_entry_move,
            article.published_at,
            expires_at,
            decided_at,
            bars,
            mkt_price,
            mkt_prev_close,
        )
        enter, m, p = bandit.decide(model, x)
    except Exception as exc:
        skip("error", error=f"{type(exc).__name__}: {exc}")
        return None
    if not enter:
        skip("decision", m=m, p=p)
        return None
    return {"bandit_m": m, "bandit_p": p}
