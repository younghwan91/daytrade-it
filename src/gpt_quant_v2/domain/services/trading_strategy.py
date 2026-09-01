"""Trading strategy domain service for evaluating and executing trading signals."""

from dataclasses import dataclass
from decimal import Decimal

from gpt_quant_v2.domain.entities import (
    NewsArticle,
    Position,
    SignalDirection,
    SignalSource,
    TradingSignal,
)
from gpt_quant_v2.domain.value_objects import Confidence, Price, Sentiment


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration parameters for trading strategy.

    Attributes:
        min_confidence: Minimum confidence threshold for signals (0.0 to 1.0)
        min_risk_reward: Minimum risk/reward ratio required
        take_profit_pct: Take profit percentage (e.g., 5.0 for 5%)
        stop_loss_pct: Stop loss percentage (e.g., 2.0 for 2%)
        max_position_pct: Maximum position size as percentage of account value
    """

    min_confidence: float = 0.6
    min_risk_reward: float = 2.0
    take_profit_pct: float = 5.0
    stop_loss_pct: float = 2.0
    max_position_pct: float = 5.0


class NewsBasedStrategy:
    """Trading strategy based on news sentiment analysis.

    This strategy evaluates news articles and their sentiment to generate
    trading signals with appropriate position sizing and exit conditions.
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        """Initialize strategy with configuration.

        Args:
            config: Strategy configuration, uses defaults if not provided.
        """
        self.config = config or StrategyConfig()

    def evaluate_signal(
        self,
        news: NewsArticle,
        sentiment: Sentiment,
    ) -> TradingSignal | None:
        """Evaluate news and sentiment to generate a trading signal.

        Args:
            news: The news article to evaluate.
            sentiment: The sentiment analysis result.

        Returns:
            A TradingSignal if conditions are met, None otherwise.
        """
        # Check minimum confidence threshold
        if sentiment.confidence < self.config.min_confidence:
            return None

        # Neutral sentiment doesn't generate signals
        if sentiment.is_neutral:
            return None

        # News must have associated tickers
        if not news.has_tickers:
            return None

        # Determine signal direction based on sentiment
        direction = SignalDirection.BUY if sentiment.is_bullish else SignalDirection.SELL

        # Calculate predicted return and drawdown based on sentiment strength
        predicted_return = Decimal(str(abs(sentiment.score) * 10))  # Up to 10% return
        predicted_drawdown = Decimal(str(-abs(sentiment.score) * 5))  # Up to 5% drawdown

        # Calculate risk/reward ratio
        risk_reward = self._calculate_risk_reward(
            float(predicted_return),
            float(predicted_drawdown),
        )

        # Check minimum risk/reward requirement
        if risk_reward < self.config.min_risk_reward:
            return None

        # Use first ticker for the signal
        ticker = news.tickers[0]

        return TradingSignal(
            ticker=ticker,
            direction=direction,
            confidence=Confidence(value=sentiment.confidence),
            source=SignalSource.NEWS,
            predicted_return=predicted_return,
            predicted_drawdown=predicted_drawdown,
            risk_reward_ratio=Decimal(str(risk_reward)),
            news_article_id=news.id,
            metadata={
                "sentiment_score": sentiment.score,
                "sentiment_label": sentiment.label.value,
                "news_source": news.source,
                "news_title": news.title[:100],
            },
        )

    def calculate_position_size(
        self,
        signal: TradingSignal,
        account_value: Decimal,
    ) -> int:
        """Calculate the number of shares to trade.

        Uses confidence-weighted position sizing with maximum position limits.

        Args:
            signal: The trading signal.
            account_value: Total account value in dollars.

        Returns:
            Number of shares to trade (always positive).
        """
        if account_value <= 0:
            return 0

        # Base position size as percentage of account
        base_pct = self.config.max_position_pct / 100

        # Weight by confidence (higher confidence = larger position)
        confidence_weight = signal.confidence.value

        # Calculate maximum dollar amount for position
        position_value = account_value * Decimal(str(base_pct * confidence_weight))

        # We need a price to calculate shares - use metadata or return 0
        # In practice, current price would be fetched externally
        if "current_price" in signal.metadata:
            price = Decimal(str(signal.metadata["current_price"]))
            if price > 0:
                shares = int(position_value / price)
                return max(0, shares)

        # Return position value as shares if no price (placeholder)
        return int(position_value / Decimal("100"))  # Assume $100/share default

    def should_exit(
        self,
        position: Position,
        current_price: Price,
    ) -> tuple[bool, str]:
        """Determine if a position should be exited.

        Checks take profit and stop loss conditions.

        Args:
            position: The current position.
            current_price: Current market price.

        Returns:
            Tuple of (should_exit, reason).
        """
        if position.average_price <= 0:
            return False, ""

        # Calculate return percentage
        price_change = current_price.value - position.average_price
        return_pct = (price_change / position.average_price) * Decimal("100")

        # Adjust for position direction
        if position.is_short:
            return_pct = -return_pct

        # Check take profit
        if return_pct >= Decimal(str(self.config.take_profit_pct)):
            return True, f"Take profit triggered at {return_pct:.2f}%"

        # Check stop loss
        if return_pct <= -Decimal(str(self.config.stop_loss_pct)):
            return True, f"Stop loss triggered at {return_pct:.2f}%"

        return False, ""

    def _calculate_risk_reward(
        self,
        predicted_return: float,
        predicted_drawdown: float,
    ) -> float:
        """Calculate risk/reward ratio.

        Args:
            predicted_return: Expected return (positive).
            predicted_drawdown: Expected drawdown (negative).

        Returns:
            Risk/reward ratio (return / abs(drawdown)).
        """
        if predicted_drawdown >= 0:
            return float("inf") if predicted_return > 0 else 0.0
        return abs(predicted_return / predicted_drawdown)


__all__ = [
    "NewsBasedStrategy",
    "StrategyConfig",
]
