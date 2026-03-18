"""
Trading Domain Events

Events that occur during the trading lifecycle.
"""

from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import Field

from .base import DomainEvent


class NewsAnalyzed(DomainEvent):
    """
    Event emitted when a news article has been analyzed for sentiment.
    
    Attributes:
        article_id: ID of the analyzed article
        ticker: Stock ticker the news relates to
        sentiment_score: Sentiment score (-1.0 to 1.0)
        sentiment_label: Human-readable sentiment label
        confidence: Confidence in the sentiment analysis
        keywords: Key phrases extracted from the article
    """
    
    article_id: UUID
    ticker: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_label: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    confidence: float = Field(ge=0.0, le=1.0)
    keywords: list[str] = Field(default_factory=list)
    analysis_duration_ms: int | None = None


class SignalGenerated(DomainEvent):
    """
    Event emitted when a new trading signal is generated.
    
    Attributes:
        signal_id: Unique identifier for the signal
        ticker: Stock ticker for the signal
        direction: BUY or SELL
        confidence: Signal confidence score
        source: Source of the signal (NEWS, TECHNICAL, ML)
        predicted_return: Expected return percentage
        predicted_drawdown: Expected maximum drawdown
        risk_reward_ratio: Risk/reward ratio
        expires_at: When the signal expires
    """
    
    signal_id: UUID
    ticker: str
    direction: Literal["BUY", "SELL"]
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["NEWS", "TECHNICAL", "ML", "COMBINED"]
    predicted_return: float
    predicted_drawdown: float
    risk_reward_ratio: float
    expires_at: datetime
    trigger_news_id: UUID | None = None


class SignalExpired(DomainEvent):
    """Event emitted when a trading signal expires without being executed."""
    
    signal_id: UUID
    ticker: str
    reason: str = "timeout"


class TradeExecuted(DomainEvent):
    """
    Event emitted when a trade is executed.
    
    Attributes:
        trade_id: Unique identifier for the trade
        signal_id: ID of the signal that triggered this trade
        ticker: Stock ticker
        direction: BUY or SELL
        quantity: Number of shares
        price: Execution price
        total_value: Total trade value
        fees: Trading fees
        broker_order_id: Order ID from the broker
    """
    
    trade_id: UUID
    signal_id: UUID
    ticker: str
    direction: Literal["BUY", "SELL"]
    quantity: int
    price: Decimal
    total_value: Decimal
    fees: Decimal = Decimal("0")
    broker_order_id: str | None = None


class TradeClosed(DomainEvent):
    """
    Event emitted when a trade position is closed.
    
    Attributes:
        trade_id: ID of the original trade
        exit_price: Price at which the position was closed
        pnl: Profit/loss from the trade
        pnl_percent: Percentage return
        hold_duration_seconds: How long the position was held
        exit_reason: Why the position was closed
    """
    
    trade_id: UUID
    ticker: str
    exit_price: Decimal
    pnl: Decimal
    pnl_percent: float
    hold_duration_seconds: int
    exit_reason: Literal["TAKE_PROFIT", "STOP_LOSS", "TIMEOUT", "MANUAL", "SIGNAL"]


class PositionUpdated(DomainEvent):
    """
    Event emitted when a position is updated.
    
    Attributes:
        position_id: ID of the position
        ticker: Stock ticker
        quantity: Updated quantity
        average_price: Updated average price
        current_price: Current market price
        unrealized_pnl: Current unrealized P&L
    """
    
    position_id: UUID
    ticker: str
    quantity: int
    average_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_percent: float


class RiskLimitBreached(DomainEvent):
    """
    Event emitted when a risk limit is breached.
    
    Attributes:
        limit_type: Type of limit breached
        limit_value: The limit that was set
        actual_value: The actual value that breached it
        action_taken: What action was taken in response
    """
    
    limit_type: Literal["DAILY_LOSS", "POSITION_SIZE", "TOTAL_EXPOSURE", "DRAWDOWN"]
    limit_value: Decimal
    actual_value: Decimal
    action_taken: Literal["BLOCKED", "REDUCED", "CLOSED_ALL", "ALERT_ONLY"]
    affected_tickers: list[str] = Field(default_factory=list)


class SystemHealthChanged(DomainEvent):
    """Event emitted when system health status changes."""
    
    component: str
    previous_status: Literal["HEALTHY", "DEGRADED", "UNHEALTHY"]
    current_status: Literal["HEALTHY", "DEGRADED", "UNHEALTHY"]
    details: str | None = None
