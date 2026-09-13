"""Signal generation domain service for creating trading signals from various sources.

Sentiment-only signal discipline (Task 11 / Phase 4b):
    Deep-research finding #1 (folded into the plan, see
    ``.superpowers/sdd/PLAN/task-11-brief.md``) is explicit: news sentiment
    alone does not reliably beat a benchmark as a standalone trading
    signal -- it only helps when *combined* with technical/ML signals.

    ``generate_from_news`` below therefore only ever produces a *component*
    signal (``source=SignalSource.NEWS``) meant as one input to
    ``combine_signals()``, never a directly actionable recommendation on
    its own. ``combine_signals()`` enforces this structurally: it refuses
    to produce a composite signal (returns ``None``) unless at least one
    non-sentiment source (ML, TECHNICAL, or FUNDAMENTAL) corroborates the
    NEWS-sourced signal(s). A caller cannot get an actionable signal out of
    sentiment alone, even by accident (e.g. by passing a single NEWS signal
    into ``combine_signals``) -- see ``combine_signals``'s docstring.

    The sibling standalone-signal-from-sentiment class,
    ``NewsBasedStrategy.evaluate_signal`` in ``trading_strategy.py``, was
    retired for the same reason (see that module's docstring) -- it was
    unused anywhere outside its own module/tests, and duplicated exactly
    the sentiment-only signal-generation pattern this finding argues
    against.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from daytrade_it.domain.entities import (
    NewsArticle,
    SignalDirection,
    SignalSource,
    TradingSignal,
)
from daytrade_it.domain.value_objects import Confidence, Sentiment, Ticker


@dataclass(frozen=True)
class SignalConfig:
    """Configuration for signal generation.

    Attributes:
        min_confidence: Minimum confidence for signal generation
        signal_validity_hours: Hours until signal expires
        sentiment_weight: Weight for sentiment-based signals
        ml_weight: Weight for ML-based signals
        technical_weight: Weight for technical signals
    """

    min_confidence: float = 0.5
    signal_validity_hours: int = 24
    sentiment_weight: float = 0.4
    ml_weight: float = 0.4
    technical_weight: float = 0.2


class SignalGenerator:
    """Domain service for generating trading signals from various sources.

    Generates signals from news sentiment, ML predictions, and combines
    multiple signals using confidence weighting.
    """

    def __init__(self, config: SignalConfig | None = None) -> None:
        """Initialize signal generator with configuration.

        Args:
            config: Signal generation configuration.
        """
        self.config = config or SignalConfig()

    def generate_from_news(
        self,
        article: NewsArticle,
        sentiment: Sentiment,
    ) -> TradingSignal | None:
        """Generate a NEWS-sourced *component* signal from an article and sentiment.

        NOT independently actionable. This produces one input signal
        (``source=SignalSource.NEWS``) meant to be passed into
        ``combine_signals()`` alongside an ML/technical signal for the same
        ticker -- per deep-research finding #1 (see module docstring),
        sentiment alone does not reliably generate a tradeable signal.
        ``combine_signals()`` will refuse (return ``None``) to combine a
        list made up only of NEWS-sourced signals, so this method's output
        cannot become an actionable ``TradingSignal`` without ML/technical
        corroboration.

        Args:
            article: The news article source.
            sentiment: Analyzed sentiment of the article.

        Returns:
            A NEWS-sourced component TradingSignal if conditions are met,
            None otherwise. Treat the result as advisory input to
            ``combine_signals()``, not as a standalone recommendation.
        """
        # Require minimum confidence
        if sentiment.confidence < self.config.min_confidence:
            return None

        # Neutral sentiment doesn't generate signals
        if sentiment.is_neutral:
            return None

        # Must have tickers to trade
        if not article.has_tickers:
            return None

        # Determine direction from sentiment
        direction = SignalDirection.BUY if sentiment.is_bullish else SignalDirection.SELL

        # Calculate confidence from sentiment
        signal_confidence = self._calculate_signal_confidence(
            base_confidence=sentiment.confidence,
            sentiment_score=abs(sentiment.score),
        )

        if signal_confidence < self.config.min_confidence:
            return None

        # Estimate return and risk based on sentiment magnitude
        sentiment_magnitude = abs(sentiment.score)
        predicted_return = Decimal(str(sentiment_magnitude * 8))  # Up to 8%
        predicted_drawdown = Decimal(str(-sentiment_magnitude * 4))  # Up to 4%

        # Calculate risk/reward
        risk_reward = (
            abs(predicted_return / predicted_drawdown) if predicted_drawdown != 0 else Decimal("0")
        )

        return TradingSignal(
            ticker=article.tickers[0],
            direction=direction,
            confidence=Confidence(value=signal_confidence),
            source=SignalSource.NEWS,
            predicted_return=predicted_return,
            predicted_drawdown=predicted_drawdown,
            risk_reward_ratio=risk_reward,
            expires_at=datetime.now(UTC) + timedelta(hours=self.config.signal_validity_hours),
            news_article_id=article.id,
            metadata={
                "source_type": "news",
                "standalone_actionable": False,
                "sentiment_score": sentiment.score,
                "sentiment_label": sentiment.label.value,
                "article_source": article.source,
                "article_title": article.title[:100],
            },
        )

    def generate_from_ml_prediction(
        self,
        prediction: dict[str, Any],
    ) -> TradingSignal | None:
        """Generate a trading signal from ML model prediction.

        Expected prediction format:
        {
            "ticker": str,
            "direction": "BUY" | "SELL",
            "confidence": float (0-1),
            "predicted_return": float,
            "predicted_drawdown": float,
            "model_name": str (optional),
            "features": dict (optional)
        }

        Args:
            prediction: ML model prediction dictionary.

        Returns:
            TradingSignal if valid, None otherwise.
        """
        # Validate required fields
        required_fields = ["ticker", "direction", "confidence"]
        if not all(field in prediction for field in required_fields):
            return None

        try:
            ticker = Ticker(value=prediction["ticker"])
            confidence = float(prediction["confidence"])
            direction = SignalDirection(prediction["direction"].upper())
        except (ValueError, KeyError):
            return None

        # Check minimum confidence
        if confidence < self.config.min_confidence:
            return None

        # Extract optional predictions
        predicted_return = prediction.get("predicted_return")
        predicted_drawdown = prediction.get("predicted_drawdown")

        # Calculate risk/reward if metrics provided
        risk_reward: Decimal | None = None
        if (
            predicted_return is not None
            and predicted_drawdown is not None
            and predicted_drawdown < 0
        ):
            risk_reward = Decimal(str(abs(predicted_return / predicted_drawdown)))

        return TradingSignal(
            ticker=ticker,
            direction=direction,
            confidence=Confidence(value=confidence),
            source=SignalSource.ML,
            predicted_return=Decimal(str(predicted_return)) if predicted_return else None,
            predicted_drawdown=Decimal(str(predicted_drawdown)) if predicted_drawdown else None,
            risk_reward_ratio=risk_reward,
            expires_at=datetime.now(UTC) + timedelta(hours=self.config.signal_validity_hours),
            metadata={
                "source_type": "ml",
                "model_name": prediction.get("model_name", "unknown"),
                "features": prediction.get("features", {}),
            },
        )

    def combine_signals(
        self,
        signals: list[TradingSignal],
    ) -> TradingSignal | None:
        """Combine multiple signals into a single composite signal.

        Uses confidence-weighted averaging to combine signals for the same ticker.
        All signals must be for the same ticker and direction.

        Sentiment-only discipline (Task 11 / deep-research finding #1): a
        signal set made up *entirely* of ``SignalSource.NEWS`` signals is
        refused (returns ``None``) -- sentiment alone does not reliably
        produce a tradeable signal, only sentiment combined with an
        ML/TECHNICAL/FUNDAMENTAL signal does. This applies even to a
        single-signal list: passing one lone NEWS signal in no longer
        returns it unchanged as if it were actionable.

        Args:
            signals: List of signals to combine.

        Returns:
            Combined TradingSignal if valid, None otherwise.
        """
        if not signals:
            return None

        # Refuse to produce an actionable signal from sentiment alone --
        # require at least one non-NEWS (ML/TECHNICAL/FUNDAMENTAL) source.
        non_news_sources = {SignalSource.ML, SignalSource.TECHNICAL, SignalSource.FUNDAMENTAL}
        if not any(s.source in non_news_sources for s in signals):
            return None

        if len(signals) == 1:
            return signals[0]

        # All signals must be for the same ticker
        tickers = {s.ticker.value for s in signals}
        if len(tickers) > 1:
            return None

        # Determine consensus direction using confidence weighting
        buy_confidence = sum(
            s.confidence.value for s in signals if s.direction == SignalDirection.BUY
        )
        sell_confidence = sum(
            s.confidence.value for s in signals if s.direction == SignalDirection.SELL
        )

        if buy_confidence == sell_confidence == 0:
            return None

        consensus_direction = (
            SignalDirection.BUY if buy_confidence > sell_confidence else SignalDirection.SELL
        )

        # Filter to signals matching consensus direction
        aligned_signals = [s for s in signals if s.direction == consensus_direction]

        if not aligned_signals:
            return None

        # Calculate weighted average confidence
        total_confidence = sum(s.confidence.value for s in aligned_signals)
        avg_confidence = total_confidence / len(aligned_signals)

        # Calculate weighted average returns and drawdowns
        weighted_return = self._weighted_average(
            [
                (s.predicted_return, s.confidence.value)
                for s in aligned_signals
                if s.predicted_return
            ]
        )
        weighted_drawdown = self._weighted_average(
            [
                (s.predicted_drawdown, s.confidence.value)
                for s in aligned_signals
                if s.predicted_drawdown
            ]
        )

        # Calculate combined risk/reward
        risk_reward: Decimal | None = None
        if weighted_return and weighted_drawdown and float(weighted_drawdown) < 0:
            risk_reward = Decimal(str(abs(float(weighted_return) / float(weighted_drawdown))))

        # Build source metadata
        source_breakdown = {}
        for signal in aligned_signals:
            source_name = signal.source.value.lower()
            if source_name not in source_breakdown:
                source_breakdown[source_name] = 0
            source_breakdown[source_name] += 1

        return TradingSignal(
            ticker=signals[0].ticker,
            direction=consensus_direction,
            confidence=Confidence(value=min(avg_confidence, 1.0)),
            source=SignalSource.COMPOSITE,
            predicted_return=weighted_return,
            predicted_drawdown=weighted_drawdown,
            risk_reward_ratio=risk_reward,
            expires_at=datetime.now(UTC) + timedelta(hours=self.config.signal_validity_hours),
            metadata={
                "source_type": "composite",
                "signal_count": len(aligned_signals),
                "source_breakdown": source_breakdown,
                "buy_confidence_sum": buy_confidence,
                "sell_confidence_sum": sell_confidence,
            },
        )

    def _calculate_signal_confidence(
        self,
        base_confidence: float,
        sentiment_score: float,
    ) -> float:
        """Calculate overall signal confidence.

        Combines base confidence with sentiment magnitude for
        a weighted confidence score.

        Args:
            base_confidence: Base confidence from sentiment analysis.
            sentiment_score: Absolute sentiment score (0-1).

        Returns:
            Combined confidence score (0-1).
        """
        # Weight: 70% from analysis confidence, 30% from sentiment magnitude
        return (base_confidence * 0.7) + (sentiment_score * 0.3)

    def _weighted_average(
        self,
        values_weights: list[tuple[Decimal | None, float]],
    ) -> Decimal | None:
        """Calculate confidence-weighted average of values.

        Args:
            values_weights: List of (value, weight) tuples.

        Returns:
            Weighted average as Decimal, or None if no valid values.
        """
        valid_pairs = [(v, w) for v, w in values_weights if v is not None]
        if not valid_pairs:
            return None

        total_weight = sum(w for _, w in valid_pairs)
        if total_weight == 0:
            return None

        weighted_sum = sum(float(v) * w for v, w in valid_pairs)
        return Decimal(str(weighted_sum / total_weight))


__all__ = [
    "SignalConfig",
    "SignalGenerator",
]
