"""
Signal Generation Handler.

Generates trading signals based on news sentiment, price data,
and ML model predictions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from gpt_quant_v2.domain.entities.signal import (
    SignalDirection,
    SignalSource,
    TradingSignal,
)
from gpt_quant_v2.domain.events.trading_events import SignalGeneratedEvent
from gpt_quant_v2.domain.value_objects import Confidence, Ticker

if TYPE_CHECKING:
    from gpt_quant_v2.application.ml.feature_engineer import FeatureEngineer
    from gpt_quant_v2.infrastructure.external.polygon_client import PolygonClient
    from gpt_quant_v2.infrastructure.messaging.redis_pubsub import RedisPubSub
    from gpt_quant_v2.infrastructure.persistence.repositories import (
        NewsRepository,
        SignalRepository,
    )

logger = structlog.get_logger(__name__)


@dataclass
class GenerateSignalCommand:
    """Command to generate trading signal for a ticker."""

    ticker: str
    use_news: bool = True
    use_price: bool = True
    use_ml: bool = True
    min_confidence: float = 0.6
    correlation_id: str | None = None


@dataclass
class SignalGenerationConfig:
    """Configuration for signal generation."""

    # News sentiment thresholds
    bullish_sentiment_threshold: float = 0.3
    bearish_sentiment_threshold: float = -0.3
    min_news_articles: int = 2

    # Price momentum thresholds
    momentum_lookback_minutes: int = 60
    bullish_momentum_threshold: float = 0.005  # 0.5%
    bearish_momentum_threshold: float = -0.005

    # ML model settings
    ml_min_confidence: float = 0.7

    # Signal expiration
    signal_expiry_hours: int = 4


@dataclass
class SignalGenerationResult:
    """Result of signal generation."""

    ticker: str
    signal: TradingSignal | None
    components: dict = field(default_factory=dict)
    reason: str = ""


class SignalGenerationHandler:
    """
    Generates trading signals using multiple data sources.

    Signal generation pipeline:
    1. Fetch recent news sentiment
    2. Fetch price momentum data
    3. Run ML model prediction (if available)
    4. Combine signals with weighted voting
    5. Apply confidence threshold
    6. Store and publish signal
    """

    def __init__(
        self,
        news_repository: NewsRepository,
        signal_repository: SignalRepository,
        polygon_client: PolygonClient,
        feature_engineer: FeatureEngineer,
        pubsub: RedisPubSub,
        config: SignalGenerationConfig | None = None,
    ) -> None:
        """Initialize with dependencies."""
        self._news_repo = news_repository
        self._signal_repo = signal_repository
        self._polygon = polygon_client
        self._feature_eng = feature_engineer
        self._pubsub = pubsub
        self._config = config or SignalGenerationConfig()
        self._ml_model = None  # Loaded lazily

    async def handle(self, command: GenerateSignalCommand) -> SignalGenerationResult:
        """
        Generate a trading signal for the given ticker.

        Args:
            command: Signal generation parameters.

        Returns:
            SignalGenerationResult with signal (if generated) and component details.
        """
        log = logger.bind(
            ticker=command.ticker,
            correlation_id=command.correlation_id,
        )
        log.info("Generating trading signal")

        ticker = Ticker(value=command.ticker.upper())
        components: dict = {}

        # Gather signal components in parallel
        tasks = []

        if command.use_news:
            tasks.append(self._get_news_signal(ticker, log))
        if command.use_price:
            tasks.append(self._get_momentum_signal(ticker, log))
        if command.use_ml:
            tasks.append(self._get_ml_signal(ticker, log))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        idx = 0
        if command.use_news:
            if isinstance(results[idx], dict):
                components["news"] = results[idx]
            else:
                log.warning("News signal failed", error=str(results[idx]))
                components["news"] = {"direction": None, "confidence": 0}
            idx += 1

        if command.use_price:
            if isinstance(results[idx], dict):
                components["momentum"] = results[idx]
            else:
                log.warning("Momentum signal failed", error=str(results[idx]))
                components["momentum"] = {"direction": None, "confidence": 0}
            idx += 1

        if command.use_ml:
            if isinstance(results[idx], dict):
                components["ml"] = results[idx]
            else:
                log.warning("ML signal failed", error=str(results[idx]))
                components["ml"] = {"direction": None, "confidence": 0}
            idx += 1

        # Combine signals
        combined = self._combine_signals(components)
        log.info("Combined signal", **combined)

        # Check confidence threshold
        if combined["confidence"] < command.min_confidence:
            log.info(
                "Signal below confidence threshold",
                confidence=combined["confidence"],
                threshold=command.min_confidence,
            )
            return SignalGenerationResult(
                ticker=ticker.value,
                signal=None,
                components=components,
                reason=f"Confidence {combined['confidence']:.2f} below threshold {command.min_confidence}",
            )

        if combined["direction"] is None:
            return SignalGenerationResult(
                ticker=ticker.value,
                signal=None,
                components=components,
                reason="No clear direction from signals",
            )

        # Create the signal
        signal = TradingSignal(
            ticker=ticker,
            direction=combined["direction"],
            confidence=Confidence(value=combined["confidence"]),
            source=SignalSource.COMPOSITE,
            predicted_return=combined.get("predicted_return"),
            predicted_drawdown=combined.get("predicted_drawdown"),
            risk_reward_ratio=combined.get("risk_reward_ratio"),
            expires_at=datetime.now(UTC)
            + timedelta(hours=self._config.signal_expiry_hours),
            metadata={
                "components": components,
                "generated_at": datetime.now(UTC).isoformat(),
            },
        )

        # Save to repository
        await self._signal_repo.save(signal)

        # Publish event
        event = SignalGeneratedEvent(
            signal_id=signal.id,
            ticker=ticker.value,
            direction=signal.direction.value,
            confidence=signal.confidence.value,
            source=signal.source.value,
        )
        await self._pubsub.publish(f"signals.{ticker.value}", event.model_dump())
        await self._pubsub.publish("signals.all", event.model_dump())

        log.info(
            "Signal generated",
            signal_id=str(signal.id),
            direction=signal.direction.value,
            confidence=signal.confidence.value,
        )

        return SignalGenerationResult(
            ticker=ticker.value,
            signal=signal,
            components=components,
            reason="Signal generated successfully",
        )

    async def _get_news_signal(
        self,
        ticker: Ticker,
        log: structlog.BoundLogger,
    ) -> dict:
        """Get signal component from news sentiment."""
        try:
            # Get recent analyzed news
            articles = await self._news_repo.get_recent(
                ticker=ticker.value,
                hours=24,
                limit=20,
            )

            if len(articles) < self._config.min_news_articles:
                return {"direction": None, "confidence": 0, "reason": "Insufficient news"}

            # Calculate average sentiment
            sentiments = [
                a.sentiment.score
                for a in articles
                if a.sentiment and a.sentiment.score is not None
            ]

            if not sentiments:
                return {"direction": None, "confidence": 0, "reason": "No sentiment data"}

            avg_sentiment = sum(sentiments) / len(sentiments)
            confidence = min(len(sentiments) / 10, 1.0)  # More articles = more confidence

            if avg_sentiment > self._config.bullish_sentiment_threshold:
                direction = SignalDirection.BUY
            elif avg_sentiment < self._config.bearish_sentiment_threshold:
                direction = SignalDirection.SELL
            else:
                direction = None

            return {
                "direction": direction,
                "confidence": confidence * abs(avg_sentiment),
                "avg_sentiment": avg_sentiment,
                "article_count": len(sentiments),
            }

        except Exception as e:
            log.error("News signal error", error=str(e))
            raise

    async def _get_momentum_signal(
        self,
        ticker: Ticker,
        log: structlog.BoundLogger,
    ) -> dict:
        """Get signal component from price momentum."""
        try:
            # Get recent price bars
            bars = await self._polygon.get_bars(
                ticker=ticker.value,
                timespan="minute",
                limit=self._config.momentum_lookback_minutes,
            )

            if len(bars) < 10:
                return {"direction": None, "confidence": 0, "reason": "Insufficient price data"}

            # Calculate returns
            prices = [b["close"] for b in bars]
            returns_1m = (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0
            returns_5m = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0
            returns_10m = (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 else 0

            # Calculate volume ratio
            volumes = [b["volume"] for b in bars]
            recent_vol = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else 0
            avg_vol = sum(volumes) / len(volumes) if volumes else 1
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1

            # Momentum score (weighted average of returns)
            momentum = (returns_1m * 0.5 + returns_5m * 0.3 + returns_10m * 0.2)

            # Higher volume = higher confidence
            confidence = min(vol_ratio, 2.0) / 2.0  # Cap at 100%

            if momentum > self._config.bullish_momentum_threshold:
                direction = SignalDirection.BUY
            elif momentum < self._config.bearish_momentum_threshold:
                direction = SignalDirection.SELL
            else:
                direction = None

            return {
                "direction": direction,
                "confidence": confidence * min(abs(momentum) * 50, 1.0),
                "momentum": momentum,
                "volume_ratio": vol_ratio,
                "returns_1m": returns_1m,
                "returns_5m": returns_5m,
            }

        except Exception as e:
            log.error("Momentum signal error", error=str(e))
            raise

    async def _get_ml_signal(
        self,
        ticker: Ticker,
        log: structlog.BoundLogger,
    ) -> dict:
        """Get signal component from ML model."""
        try:
            # Check if model is loaded
            if self._ml_model is None:
                return {"direction": None, "confidence": 0, "reason": "No model loaded"}

            # Get features
            features = await self._feature_eng.extract_features(ticker.value)

            if features is None or features.empty:
                return {"direction": None, "confidence": 0, "reason": "Feature extraction failed"}

            # Get prediction
            prediction = self._ml_model.predict_proba(features)

            # Assuming 3 classes: SELL (0), HOLD (1), BUY (2)
            buy_prob = prediction[0][2] if len(prediction[0]) > 2 else 0
            sell_prob = prediction[0][0] if len(prediction[0]) > 0 else 0

            if buy_prob > self._config.ml_min_confidence:
                direction = SignalDirection.BUY
                confidence = buy_prob
            elif sell_prob > self._config.ml_min_confidence:
                direction = SignalDirection.SELL
                confidence = sell_prob
            else:
                direction = None
                confidence = max(buy_prob, sell_prob)

            return {
                "direction": direction,
                "confidence": confidence,
                "buy_prob": buy_prob,
                "sell_prob": sell_prob,
                "hold_prob": prediction[0][1] if len(prediction[0]) > 1 else 0,
            }

        except Exception as e:
            log.error("ML signal error", error=str(e))
            raise

    def _combine_signals(self, components: dict) -> dict:
        """
        Combine signal components with weighted voting.

        Weights:
        - News sentiment: 0.4
        - Price momentum: 0.3
        - ML prediction: 0.3
        """
        weights = {"news": 0.4, "momentum": 0.3, "ml": 0.3}

        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0

        for name, weight in weights.items():
            component = components.get(name, {})
            direction = component.get("direction")
            confidence = component.get("confidence", 0)

            if direction == SignalDirection.BUY:
                buy_score += weight * confidence
                total_weight += weight
            elif direction == SignalDirection.SELL:
                sell_score += weight * confidence
                total_weight += weight

        if total_weight == 0:
            return {"direction": None, "confidence": 0}

        # Normalize scores
        buy_score /= total_weight
        sell_score /= total_weight

        # Determine direction
        if buy_score > sell_score and buy_score > 0.3:
            direction = SignalDirection.BUY
            confidence = buy_score
        elif sell_score > buy_score and sell_score > 0.3:
            direction = SignalDirection.SELL
            confidence = sell_score
        else:
            direction = None
            confidence = max(buy_score, sell_score)

        # Estimate return/drawdown based on component analysis
        predicted_return = None
        if direction and "momentum" in components:
            momentum = components["momentum"].get("momentum", 0)
            predicted_return = Decimal(str(round(momentum * 100, 2)))  # As percentage

        return {
            "direction": direction,
            "confidence": round(confidence, 3),
            "buy_score": round(buy_score, 3),
            "sell_score": round(sell_score, 3),
            "predicted_return": predicted_return,
            "predicted_drawdown": None,  # Would need volatility analysis
            "risk_reward_ratio": None,
        }

    def set_model(self, model) -> None:
        """Set the ML model for predictions."""
        self._ml_model = model
