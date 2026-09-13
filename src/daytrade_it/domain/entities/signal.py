"""TradingSignal entity representing a trading recommendation."""

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import Field, model_validator

from daytrade_it.domain.entities.base import Entity
from daytrade_it.domain.value_objects import Confidence, Ticker


class SignalDirection(StrEnum):
    """Trading signal direction."""

    BUY = "BUY"
    SELL = "SELL"


class SignalSource(StrEnum):
    """Source of the trading signal."""

    NEWS = "NEWS"
    TECHNICAL = "TECHNICAL"
    ML = "ML"
    FUNDAMENTAL = "FUNDAMENTAL"
    COMPOSITE = "COMPOSITE"


class TradingSignal(Entity):
    """Domain entity representing a trading signal/recommendation.

    Attributes:
        ticker: Stock ticker for the signal
        direction: BUY or SELL recommendation
        confidence: Confidence level of the signal
        source: Origin of the signal (NEWS, TECHNICAL, ML)
        predicted_return: Expected return percentage
        predicted_drawdown: Expected maximum drawdown
        risk_reward_ratio: Risk/reward ratio
        expires_at: Signal expiration timestamp
        metadata: Additional signal metadata
    """

    ticker: Ticker
    direction: SignalDirection
    confidence: Confidence
    source: SignalSource
    predicted_return: Decimal | None = Field(default=None)
    predicted_drawdown: Decimal | None = Field(default=None, le=0)
    risk_reward_ratio: Decimal | None = Field(default=None, gt=0)
    expires_at: datetime | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)
    news_article_id: UUID | None = Field(default=None)

    @model_validator(mode="after")
    def validate_expiration(self) -> "TradingSignal":
        """Ensure expires_at is in the future when set."""
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        return self

    def __str__(self) -> str:
        return f"Signal({self.direction} {self.ticker.value} @ {self.confidence})"

    def __repr__(self) -> str:
        return (
            f"TradingSignal(id={self.id!r}, ticker={self.ticker!r}, "
            f"direction={self.direction!r}, confidence={self.confidence!r}, "
            f"source={self.source!r})"
        )

    @property
    def is_expired(self) -> bool:
        """Check if the signal has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(UTC) > self.expires_at

    @property
    def is_bullish(self) -> bool:
        """Check if this is a bullish (BUY) signal."""
        return self.direction == SignalDirection.BUY

    @property
    def is_bearish(self) -> bool:
        """Check if this is a bearish (SELL) signal."""
        return self.direction == SignalDirection.SELL

    @property
    def is_high_confidence(self) -> bool:
        """Check if signal has high confidence (>= 0.7)."""
        return self.confidence.value >= Decimal("0.7")

    def has_risk_metrics(self) -> bool:
        """Check if signal has risk metrics calculated."""
        return (
            self.predicted_return is not None
            and self.predicted_drawdown is not None
            and self.risk_reward_ratio is not None
        )
