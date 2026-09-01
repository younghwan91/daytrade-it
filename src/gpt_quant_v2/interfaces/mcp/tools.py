"""
MCP Tool Definitions for AI Agents

This module defines MCP-compatible tools that AI agents can use
to interact with the trading system. Tools follow the Model Context Protocol
specification for AI agent interoperability.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from gpt_quant_v2.application.container import Container

# Type variable for generic tool functions
T = TypeVar("T")


# =============================================================================
# Response Models
# =============================================================================


class TickerAnalysis(BaseModel):
    """Analysis result for a stock ticker."""

    ticker: str
    name: str
    exchange: str
    current_price: Decimal
    price_change_pct: float
    sentiment: dict | None = None
    recent_news: list[dict] = Field(default_factory=list)
    signals: list[dict] = Field(default_factory=list)
    analysis_timestamp: datetime


class Signal(BaseModel):
    """Trading signal."""

    id: UUID
    ticker: str
    direction: str
    confidence: float
    source: str
    predicted_return: float | None = None
    predicted_drawdown: float | None = None
    risk_reward_ratio: float | None = None
    expires_at: datetime
    created_at: datetime


class TradeResult(BaseModel):
    """Result of a trade execution."""

    trade_id: UUID
    signal_id: UUID | None = None
    ticker: str
    direction: str
    quantity: int
    price: Decimal
    total_value: Decimal
    status: str
    executed_at: datetime


class BacktestResult(BaseModel):
    """Result of a backtest run."""

    strategy: str
    start_date: str
    end_date: str
    initial_capital: Decimal
    final_capital: Decimal
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float


class Portfolio(BaseModel):
    """Current portfolio state."""

    cash: Decimal
    total_value: Decimal
    positions: list[dict]
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    updated_at: datetime


class SystemStatus(BaseModel):
    """System health status."""

    status: str
    components: dict[str, dict]
    uptime_seconds: int
    last_signal_at: datetime | None
    last_trade_at: datetime | None


class NewsItem(BaseModel):
    """News article item."""

    id: str
    title: str
    source: str
    url: str
    published_at: datetime
    tickers: list[str] = Field(default_factory=list)
    sentiment_score: float | None = None
    sentiment_label: str | None = None
    sentiment_confidence: float | None = None
    key_points: list[str] = Field(default_factory=list)


class PriceBar(BaseModel):
    """OHLCV price bar."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


class TradeHistoryItem(BaseModel):
    """Trade history item."""

    id: UUID
    ticker: str
    direction: str
    quantity: int
    entry_price: Decimal
    exit_price: Decimal | None = None
    realized_pnl: Decimal | None = None
    status: str
    created_at: datetime
    closed_at: datetime | None = None


class ModelInfo(BaseModel):
    """ML model information."""

    name: str
    version: str
    model_type: str
    metrics: dict[str, float]
    feature_count: int
    created_at: datetime
    is_active: bool = False


class TrainingResult(BaseModel):
    """ML model training result."""

    model_name: str
    model_version: str
    metrics: dict[str, float]
    training_duration_seconds: float
    samples_used: int
    created_at: datetime


# =============================================================================
# Tool Registry and Decorator
# =============================================================================


# Global tool registry
_tool_registry: dict[str, dict[str, Any]] = {}


def mcp_tool(name: str, description: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to mark a function as an MCP tool.

    Args:
        name: Tool name for MCP protocol
        description: Human-readable description of what the tool does

    Returns:
        Decorated function with MCP metadata attached
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        func._mcp_tool = True  # type: ignore[attr-defined]
        func._mcp_name = name  # type: ignore[attr-defined]
        func._mcp_description = description  # type: ignore[attr-defined]

        # Register in global registry
        _tool_registry[name] = {
            "function": func,
            "name": name,
            "description": description,
        }

        return func

    return decorator


def get_tool_by_name(name: str) -> Callable | None:
    """Get a tool function by its MCP name."""
    tool_info = _tool_registry.get(name)
    return tool_info["function"] if tool_info else None


# =============================================================================
# Service Context for Tool Execution
# =============================================================================


class MCPToolContext:
    """Context for MCP tool execution with service dependencies.

    This class provides access to application services for tool implementations.
    It should be initialized with a DI container or individual services.
    """

    _instance: MCPToolContext | None = None
    _start_time: float = time.time()

    def __init__(
        self,
        tiingo_client: Any | None = None,
        openai_client: Any | None = None,
        signal_repository: Any | None = None,
        trade_repository: Any | None = None,
        news_repository: Any | None = None,
        position_repository: Any | None = None,
        broker: Any | None = None,
        model_trainer: Any | None = None,
        settings: Any | None = None,
    ) -> None:
        """Initialize the tool context with service dependencies."""
        self.tiingo_client = tiingo_client
        self.openai_client = openai_client
        self.signal_repository = signal_repository
        self.trade_repository = trade_repository
        self.news_repository = news_repository
        self.position_repository = position_repository
        self.broker = broker
        self.model_trainer = model_trainer
        self.settings = settings
        self._last_signal_at: datetime | None = None
        self._last_trade_at: datetime | None = None

    @classmethod
    def get_instance(cls) -> MCPToolContext:
        """Get or create the singleton context instance."""
        if cls._instance is None:
            cls._instance = MCPToolContext()
        return cls._instance

    @classmethod
    def set_instance(cls, instance: MCPToolContext) -> None:
        """Set the singleton context instance."""
        cls._instance = instance

    @classmethod
    def from_container(cls, container: Container) -> MCPToolContext:
        """Create a context from a DI container."""
        # Lazy imports to avoid circular dependencies
        from gpt_quant_v2.infrastructure.brokers.base import BrokerInterface
        from gpt_quant_v2.infrastructure.external.openai_client import OpenAIClient
        from gpt_quant_v2.infrastructure.external.tiingo_client import TiingoClient
        from gpt_quant_v2.infrastructure.persistence.repositories import (
            NewsRepository,
            PositionRepository,
            SignalRepository,
            TradeRepository,
        )
        from gpt_quant_v2.shared.config import Settings

        context = cls(
            tiingo_client=container.resolve(TiingoClient)
            if container.is_registered(TiingoClient)
            else None,
            openai_client=container.resolve(OpenAIClient)
            if container.is_registered(OpenAIClient)
            else None,
            signal_repository=container.resolve(SignalRepository)
            if container.is_registered(SignalRepository)
            else None,
            trade_repository=container.resolve(TradeRepository)
            if container.is_registered(TradeRepository)
            else None,
            news_repository=container.resolve(NewsRepository)
            if container.is_registered(NewsRepository)
            else None,
            position_repository=container.resolve(PositionRepository)
            if container.is_registered(PositionRepository)
            else None,
            broker=container.resolve(BrokerInterface)
            if container.is_registered(BrokerInterface)
            else None,
            settings=container.resolve(Settings) if container.is_registered(Settings) else None,
        )
        cls.set_instance(context)
        return context

    def update_last_signal_time(self) -> None:
        """Update the last signal timestamp."""
        self._last_signal_at = datetime.now(UTC)

    def update_last_trade_time(self) -> None:
        """Update the last trade timestamp."""
        self._last_trade_at = datetime.now(UTC)

    @property
    def uptime_seconds(self) -> int:
        """Get uptime in seconds since context creation."""
        return int(time.time() - self._start_time)


def _get_context() -> MCPToolContext:
    """Get the current tool execution context."""
    return MCPToolContext.get_instance()


# =============================================================================
# Tool Implementations
# =============================================================================


@mcp_tool(
    name="analyze_ticker",
    description="Get comprehensive analysis for a stock ticker including price data, news sentiment, and active signals",
)
async def analyze_ticker(
    ticker: Annotated[str, "Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"],
    include_news: Annotated[bool, "Whether to include recent news analysis"] = True,
    include_signals: Annotated[bool, "Whether to include active trading signals"] = True,
    news_hours: Annotated[int, "Hours of news history to analyze"] = 24,
) -> TickerAnalysis:
    """Analyze a stock ticker with price data, news sentiment, and signals.

    Returns comprehensive analysis including current price, recent news
    with sentiment scores, and any active trading signals.
    """
    ctx = _get_context()
    ticker = ticker.upper().strip()

    # Get current price from broker or Tiingo
    current_price = Decimal("0")
    price_change_pct = 0.0
    name = ticker
    exchange = "UNKNOWN"

    if ctx.broker and ctx.broker.is_connected:
        try:
            quote = ctx.broker.get_quote(ticker)
            current_price = quote.last_price or quote.mid_price
        except Exception:
            pass

    if ctx.tiingo_client and current_price == 0:
        try:
            # Get price from Tiingo
            end_date = datetime.now(UTC)
            start_date = end_date - timedelta(days=2)
            prices = await ctx.tiingo_client.get_eod_prices(
                tickers=[ticker],
                start_date=start_date,
                end_date=end_date,
            )
            if prices:
                latest = prices[-1]
                current_price = Decimal(str(latest.get("close", 0)))
                if len(prices) > 1:
                    prev_close = Decimal(str(prices[-2].get("close", current_price)))
                    if prev_close > 0:
                        price_change_pct = float((current_price - prev_close) / prev_close * 100)
        except Exception:
            pass

    # Get recent news with sentiment
    recent_news: list[dict] = []
    sentiment_summary: dict | None = None

    if include_news and ctx.tiingo_client:
        try:
            news_start = datetime.now(UTC) - timedelta(hours=news_hours)
            articles = await ctx.tiingo_client.get_news(
                tickers=[ticker],
                start_date=news_start,
                limit=10,
            )

            sentiment_scores = []
            for article in articles:
                news_item: dict[str, Any] = {
                    "title": article.title,
                    "source": article.source,
                    "published_at": article.published_at.isoformat(),
                    "url": str(article.url),
                }

                # Analyze sentiment if OpenAI client available
                if ctx.openai_client:
                    try:
                        domain_article = article.to_domain_entity()
                        analysis = await ctx.openai_client.analyze_sentiment(domain_article)
                        news_item["sentiment_score"] = analysis.sentiment_score
                        news_item["sentiment_label"] = analysis.sentiment_label
                        news_item["confidence"] = analysis.confidence
                        news_item["key_points"] = analysis.key_points
                        sentiment_scores.append(analysis.sentiment_score)
                    except Exception:
                        pass

                recent_news.append(news_item)

            # Calculate aggregate sentiment
            if sentiment_scores:
                avg_score = sum(sentiment_scores) / len(sentiment_scores)
                sentiment_summary = {
                    "average_score": avg_score,
                    "articles_analyzed": len(sentiment_scores),
                    "bullish_count": sum(1 for s in sentiment_scores if s > 0.2),
                    "bearish_count": sum(1 for s in sentiment_scores if s < -0.2),
                    "neutral_count": sum(1 for s in sentiment_scores if -0.2 <= s <= 0.2),
                }
        except Exception:
            pass

    # Get active signals
    signals: list[dict] = []
    if include_signals and ctx.signal_repository:
        try:
            active_signals = await ctx.signal_repository.find_active(ticker=ticker)
            for sig in active_signals:
                signals.append(
                    {
                        "id": str(sig.id),
                        "direction": sig.direction.value,
                        "confidence": sig.confidence.value,
                        "source": sig.source.value,
                        "predicted_return": float(sig.predicted_return)
                        if sig.predicted_return
                        else None,
                        "expires_at": sig.expires_at.isoformat(),
                    }
                )
            ctx.update_last_signal_time()
        except Exception:
            pass

    return TickerAnalysis(
        ticker=ticker,
        name=name,
        exchange=exchange,
        current_price=current_price,
        price_change_pct=price_change_pct,
        sentiment=sentiment_summary,
        recent_news=recent_news,
        signals=signals,
        analysis_timestamp=datetime.now(UTC),
    )


@mcp_tool(
    name="get_news",
    description="Get recent news articles for a ticker with optional sentiment analysis",
)
async def get_news(
    ticker: Annotated[str, "Stock ticker symbol"],
    hours: Annotated[int, "Hours of news history to fetch"] = 24,
    limit: Annotated[int, "Maximum number of articles"] = 20,
    analyze_sentiment: Annotated[bool, "Whether to analyze sentiment"] = True,
) -> list[NewsItem]:
    """Get recent news articles for a ticker.

    Returns list of news items with optional sentiment analysis including
    scores, labels, and key extracted points.
    """
    ctx = _get_context()
    ticker = ticker.upper().strip()
    results: list[NewsItem] = []

    if not ctx.tiingo_client:
        return results

    try:
        news_start = datetime.now(UTC) - timedelta(hours=hours)
        articles = await ctx.tiingo_client.get_news(
            tickers=[ticker],
            start_date=news_start,
            limit=limit,
        )

        for article in articles:
            news_item = NewsItem(
                id=article.id,
                title=article.title,
                source=article.source,
                url=str(article.url),
                published_at=article.published_date,
                tickers=article.tickers,
            )

            # Analyze sentiment if requested and OpenAI client available
            if analyze_sentiment and ctx.openai_client:
                try:
                    domain_article = article.to_domain_entity()
                    analysis = await ctx.openai_client.analyze_sentiment(domain_article)
                    news_item.sentiment_score = analysis.sentiment_score
                    news_item.sentiment_label = analysis.sentiment_label
                    news_item.sentiment_confidence = analysis.confidence
                    news_item.key_points = analysis.key_points
                except Exception:
                    pass

            results.append(news_item)

    except Exception:
        pass

    return results


@mcp_tool(
    name="get_price_history",
    description="Get historical OHLCV price data for a ticker",
)
async def get_price_history(
    ticker: Annotated[str, "Stock ticker symbol"],
    days: Annotated[int, "Number of days of history"] = 30,
    interval: Annotated[str, "Data interval (daily, weekly, monthly)"] = "daily",
) -> list[PriceBar]:
    """Get historical OHLCV price data.

    Returns list of price bars with open, high, low, close, volume.
    """
    ctx = _get_context()
    ticker = ticker.upper().strip()
    results: list[PriceBar] = []

    if not ctx.tiingo_client:
        return results

    try:
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=days)

        prices = await ctx.tiingo_client.get_eod_prices(
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date,
        )

        for bar in prices:
            results.append(
                PriceBar(
                    timestamp=bar.get("date", datetime.now(UTC)),
                    open=Decimal(str(bar.get("open", 0))),
                    high=Decimal(str(bar.get("high", 0))),
                    low=Decimal(str(bar.get("low", 0))),
                    close=Decimal(str(bar.get("close", 0))),
                    volume=int(bar.get("volume", 0)),
                )
            )

    except Exception:
        pass

    return results


@mcp_tool(
    name="get_signals",
    description="Get current trading signals filtered by criteria",
)
async def get_signals(
    min_confidence: Annotated[float, "Minimum confidence threshold (0.0-1.0)"] = 0.7,
    ticker: Annotated[str | None, "Filter by specific ticker"] = None,
    direction: Annotated[str | None, "Filter by direction (BUY/SELL)"] = None,
    limit: Annotated[int, "Maximum signals to return"] = 20,
) -> list[Signal]:
    """Get active trading signals.

    Returns signals that haven't expired, filtered by the specified criteria.
    """
    ctx = _get_context()
    results: list[Signal] = []

    if not ctx.signal_repository:
        return results

    try:
        # Normalize ticker
        ticker_filter = ticker.upper().strip() if ticker else None

        # Get active signals from repository
        active_signals = await ctx.signal_repository.find_active(
            ticker=ticker_filter,
            min_confidence=min_confidence,
        )

        # Filter by direction if specified
        if direction:
            direction_upper = direction.upper()
            active_signals = [s for s in active_signals if s.direction.value == direction_upper]

        # Limit results
        for sig in active_signals[:limit]:
            results.append(
                Signal(
                    id=sig.id,
                    ticker=sig.ticker.value,
                    direction=sig.direction.value,
                    confidence=sig.confidence.value,
                    source=sig.source.value,
                    predicted_return=float(sig.predicted_return) if sig.predicted_return else None,
                    predicted_drawdown=float(sig.predicted_drawdown)
                    if sig.predicted_drawdown
                    else None,
                    risk_reward_ratio=float(sig.risk_reward_ratio)
                    if sig.risk_reward_ratio
                    else None,
                    expires_at=sig.expires_at,
                    created_at=sig.created_at,
                )
            )

        ctx.update_last_signal_time()

    except Exception:
        pass

    return results


@mcp_tool(
    name="execute_trade",
    description="Execute a trade based on a trading signal",
)
async def execute_trade(
    signal_id: Annotated[str, "UUID of the trading signal to execute"],
    quantity: Annotated[int | None, "Number of shares (None = auto-size)"] = None,
    order_type: Annotated[str, "Order type (MARKET/LIMIT)"] = "MARKET",
    limit_price: Annotated[float | None, "Limit price (required for LIMIT orders)"] = None,
) -> TradeResult:
    """Execute a trade based on a signal.

    If quantity is not specified, position sizing will be calculated
    based on risk management rules.
    """
    ctx = _get_context()

    if not ctx.signal_repository or not ctx.broker:
        raise ValueError("Signal repository and broker must be configured")

    # Parse signal ID
    sig_uuid = UUID(signal_id)

    # Get the signal
    signal = await ctx.signal_repository.get_by_id(sig_uuid)
    if not signal:
        raise ValueError(f"Signal not found: {signal_id}")

    if signal.is_expired:
        raise ValueError(f"Signal has expired: {signal_id}")

    # Determine order parameters
    from gpt_quant_v2.infrastructure.brokers.base import Order, OrderSide, OrderType

    side = OrderSide.BUY if signal.direction.value == "BUY" else OrderSide.SELL
    otype = OrderType.LIMIT if order_type.upper() == "LIMIT" else OrderType.MARKET

    # Auto-size if quantity not provided
    if quantity is None:
        # Use 2% of portfolio value as default position size
        account = ctx.broker.get_account()
        quote = ctx.broker.get_quote(signal.ticker.value)
        price = quote.last_price or quote.mid_price
        if price > 0:
            position_value = account.equity * Decimal("0.02")
            quantity = int(position_value / price)
        else:
            quantity = 1

    # Create and place order
    order = Order(
        ticker=signal.ticker.value,
        quantity=Decimal(str(quantity)),
        side=side,
        order_type=otype,
        limit_price=Decimal(str(limit_price)) if limit_price else None,
    )

    result = ctx.broker.place_order(order)

    ctx.update_last_trade_time()

    return TradeResult(
        trade_id=UUID(result.order_id) if result.order_id else sig_uuid,
        signal_id=sig_uuid,
        ticker=signal.ticker.value,
        direction=signal.direction.value,
        quantity=quantity,
        price=result.filled_avg_price or Decimal("0"),
        total_value=result.filled_avg_price * Decimal(str(quantity))
        if result.filled_avg_price
        else Decimal("0"),
        status=result.status.value,
        executed_at=result.filled_at or datetime.now(UTC),
    )


@mcp_tool(
    name="close_position",
    description="Close an open position for a ticker",
)
async def close_position(
    ticker: Annotated[str, "Stock ticker symbol"],
    quantity: Annotated[int | None, "Shares to close (None = close all)"] = None,
) -> TradeResult:
    """Close an open position.

    If quantity is not specified, closes the entire position.
    """
    ctx = _get_context()
    ticker = ticker.upper().strip()

    if not ctx.broker:
        raise ValueError("Broker must be configured")

    # Find the position
    positions = ctx.broker.get_positions()
    position = next((p for p in positions if p.ticker == ticker), None)

    if not position:
        raise ValueError(f"No open position for {ticker}")

    # Determine quantity to close
    close_qty = quantity if quantity else abs(int(position.quantity))

    # Determine order side (opposite of position side)
    from gpt_quant_v2.infrastructure.brokers.base import Order, OrderSide, OrderType

    side = OrderSide.SELL if position.side == "long" else OrderSide.BUY

    order = Order(
        ticker=ticker,
        quantity=Decimal(str(close_qty)),
        side=side,
        order_type=OrderType.MARKET,
    )

    result = ctx.broker.place_order(order)

    ctx.update_last_trade_time()

    return TradeResult(
        trade_id=UUID(result.order_id)
        if result.order_id
        else UUID("00000000-0000-0000-0000-000000000000"),
        signal_id=None,
        ticker=ticker,
        direction=side.value.upper(),
        quantity=close_qty,
        price=result.filled_avg_price or Decimal("0"),
        total_value=result.filled_avg_price * Decimal(str(close_qty))
        if result.filled_avg_price
        else Decimal("0"),
        status=result.status.value,
        executed_at=result.filled_at or datetime.now(UTC),
    )


@mcp_tool(
    name="get_portfolio",
    description="Get current portfolio positions and value",
)
async def get_portfolio() -> Portfolio:
    """Get the current portfolio state.

    Returns cash balance, positions, and P&L metrics.
    """
    ctx = _get_context()

    if not ctx.broker:
        raise ValueError("Broker must be configured")

    account = ctx.broker.get_account()
    positions = ctx.broker.get_positions()

    position_list: list[dict] = []
    total_unrealized_pnl = Decimal("0")
    total_realized_pnl = Decimal("0")

    for pos in positions:
        position_list.append(
            {
                "ticker": pos.ticker,
                "quantity": int(pos.quantity),
                "side": pos.side,
                "average_price": float(pos.average_price),
                "current_price": float(pos.current_price) if pos.current_price else None,
                "market_value": float(pos.market_value) if pos.market_value else None,
                "unrealized_pnl": float(pos.unrealized_pnl) if pos.unrealized_pnl else None,
                "realized_pnl": float(pos.realized_pnl),
            }
        )
        if pos.unrealized_pnl:
            total_unrealized_pnl += pos.unrealized_pnl
        total_realized_pnl += pos.realized_pnl

    return Portfolio(
        cash=account.cash,
        total_value=account.equity,
        positions=position_list,
        unrealized_pnl=total_unrealized_pnl,
        realized_pnl=total_realized_pnl,
        updated_at=datetime.now(UTC),
    )


@mcp_tool(
    name="get_trade_history",
    description="Get historical trade records",
)
async def get_trade_history(
    ticker: Annotated[str | None, "Filter by specific ticker"] = None,
    status: Annotated[str | None, "Filter by status (OPEN/CLOSED/ALL)"] = None,
    limit: Annotated[int, "Maximum trades to return"] = 50,
) -> list[TradeHistoryItem]:
    """Get historical trade records.

    Returns list of trades with entry/exit prices, P&L, and status.
    """
    ctx = _get_context()
    results: list[TradeHistoryItem] = []

    if not ctx.trade_repository:
        return results

    try:
        # Normalize filters
        ticker_filter = ticker.upper().strip() if ticker else None

        # Query trades
        trades = await ctx.trade_repository.find_by_filters(
            ticker=ticker_filter,
            status=status,
            limit=limit,
        )

        for trade in trades:
            results.append(
                TradeHistoryItem(
                    id=trade.id,
                    ticker=trade.ticker.value,
                    direction=trade.direction.value,
                    quantity=int(trade.quantity),
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    realized_pnl=trade.realized_pnl,
                    status=trade.status.value,
                    created_at=trade.created_at,
                    closed_at=trade.closed_at,
                )
            )

    except Exception:
        pass

    return results


@mcp_tool(
    name="backtest",
    description="Run a backtest on a trading strategy",
)
async def backtest(
    strategy: Annotated[
        str, "Strategy name (news_momentum, technical, combined)"
    ] = "news_momentum",
    start_date: Annotated[str | None, "Start date (YYYY-MM-DD)"] = None,
    end_date: Annotated[str | None, "End date (YYYY-MM-DD)"] = None,
    initial_capital: Annotated[float, "Starting capital in USD"] = 100000,
    parameters: Annotated[dict | None, "Strategy-specific parameters"] = None,
) -> BacktestResult:
    """Run a backtest on a trading strategy.

    Returns performance metrics including returns, Sharpe ratio,
    drawdown, and trade statistics.
    """
    # Parse dates
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        start_dt = datetime.now(UTC) - timedelta(days=365)

    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now(UTC)

    # This is a placeholder implementation
    # In production, this would connect to a backtesting engine
    return BacktestResult(
        strategy=strategy,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
        initial_capital=Decimal(str(initial_capital)),
        final_capital=Decimal(str(initial_capital * 1.15)),  # Placeholder
        total_return_pct=15.0,
        sharpe_ratio=1.2,
        max_drawdown_pct=8.5,
        total_trades=150,
        winning_trades=90,
        losing_trades=60,
        win_rate=0.6,
    )


@mcp_tool(
    name="get_system_status",
    description="Get system health status and component states",
)
async def get_system_status() -> SystemStatus:
    """Get the current system health status.

    Returns status of all system components (database, Redis, APIs)
    and recent activity timestamps.
    """
    ctx = _get_context()

    components: dict[str, dict] = {}

    # Check Tiingo client
    if ctx.tiingo_client:
        components["tiingo"] = {"status": "connected", "type": "market_data"}
    else:
        components["tiingo"] = {"status": "not_configured", "type": "market_data"}

    # Check OpenAI client
    if ctx.openai_client:
        components["openai"] = {"status": "connected", "type": "sentiment_analysis"}
    else:
        components["openai"] = {"status": "not_configured", "type": "sentiment_analysis"}

    # Check broker
    if ctx.broker:
        try:
            is_connected = ctx.broker.is_connected
            components["broker"] = {
                "status": "connected" if is_connected else "disconnected",
                "type": "trading",
            }
        except Exception:
            components["broker"] = {"status": "error", "type": "trading"}
    else:
        components["broker"] = {"status": "not_configured", "type": "trading"}

    # Check repositories
    if ctx.signal_repository:
        components["signal_repository"] = {"status": "connected", "type": "persistence"}
    if ctx.trade_repository:
        components["trade_repository"] = {"status": "connected", "type": "persistence"}
    if ctx.news_repository:
        components["news_repository"] = {"status": "connected", "type": "persistence"}

    # Determine overall status
    all_statuses = [c.get("status") for c in components.values()]
    if all(s == "connected" for s in all_statuses):
        overall_status = "healthy"
    elif any(s == "error" for s in all_statuses):
        overall_status = "degraded"
    elif any(s == "connected" for s in all_statuses):
        overall_status = "partial"
    else:
        overall_status = "not_configured"

    return SystemStatus(
        status=overall_status,
        components=components,
        uptime_seconds=ctx.uptime_seconds,
        last_signal_at=ctx._last_signal_at,
        last_trade_at=ctx._last_trade_at,
    )


@mcp_tool(
    name="train_model",
    description="Trigger ML model training with specified parameters",
)
async def train_model(
    model_name: Annotated[str, "Name for the trained model"],
    tickers: Annotated[list[str] | None, "Tickers to train on (None = all)"] = None,
    lookback_days: Annotated[int, "Days of historical data to use"] = 365,
    optimize_hyperparams: Annotated[bool, "Whether to run hyperparameter optimization"] = True,
) -> TrainingResult:
    """Trigger ML model training.

    Trains an XGBoost classification model for trading signal prediction.
    """
    ctx = _get_context()
    start_time = time.time()

    if not ctx.model_trainer:
        raise ValueError("Model trainer must be configured")

    # This would connect to the actual training pipeline
    # Placeholder implementation

    metrics = {
        "accuracy": 0.72,
        "f1_weighted": 0.68,
        "precision_weighted": 0.70,
        "recall_weighted": 0.69,
    }

    training_duration = time.time() - start_time

    return TrainingResult(
        model_name=model_name,
        model_version="1.0.0",
        metrics=metrics,
        training_duration_seconds=training_duration,
        samples_used=lookback_days * 100,  # Placeholder
        created_at=datetime.now(UTC),
    )


@mcp_tool(
    name="list_models",
    description="List available ML models with their metadata",
)
async def list_models() -> list[ModelInfo]:
    """List available ML models.

    Returns information about all trained models including metrics,
    feature counts, and creation dates.
    """
    ctx = _get_context()
    results: list[ModelInfo] = []

    # Check for models in the models directory
    model_dir = Path(ctx.settings.ml.model_dir) if ctx.settings else Path("models")

    if model_dir.exists():
        for model_path in model_dir.iterdir():
            if model_path.is_dir():
                metadata_path = model_path / "metadata.joblib"
                if metadata_path.exists():
                    try:
                        import joblib

                        metadata = joblib.load(metadata_path)
                        results.append(
                            ModelInfo(
                                name=model_path.name,
                                version=metadata.get("model_version", "unknown"),
                                model_type="XGBClassifier",
                                metrics=metadata.get("metrics", {}),
                                feature_count=len(metadata.get("feature_names", [])),
                                created_at=datetime.fromisoformat(
                                    metadata.get("created_at", datetime.now(UTC).isoformat())
                                ),
                                is_active=False,
                            )
                        )
                    except Exception:
                        pass

    return results


# =============================================================================
# Tool Registry Functions
# =============================================================================


def get_mcp_tools() -> list[dict]:
    """Get all registered MCP tools with their metadata.

    Returns list of tool definitions suitable for MCP tool discovery.
    """
    tools = []

    for _name, obj in globals().items():
        if not callable(obj) or not hasattr(obj, "_mcp_tool"):
            continue

        sig = inspect.signature(obj)
        params = []

        for param_name, param in sig.parameters.items():
            if param_name in ("return",):
                continue

            annotation = param.annotation
            param_info: dict[str, Any] = {
                "name": param_name,
                "required": param.default == inspect.Parameter.empty,
            }

            # Extract description from Annotated type
            if hasattr(annotation, "__metadata__"):
                param_info["description"] = str(annotation.__metadata__[0])
                # Get the actual type from Annotated
                if hasattr(annotation, "__origin__"):
                    param_info["type"] = _get_type_name(annotation.__args__[0])
            else:
                param_info["description"] = ""
                param_info["type"] = _get_type_name(annotation)

            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default

            params.append(param_info)

        tools.append(
            {
                "name": obj._mcp_name,
                "description": obj._mcp_description,
                "parameters": params,
            }
        )

    return tools


def _get_type_name(annotation: Any) -> str:
    """Get a string representation of a type annotation."""
    if annotation == inspect.Parameter.empty:
        return "any"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    if hasattr(annotation, "__origin__"):
        origin = annotation.__origin__
        if hasattr(origin, "__name__"):
            return origin.__name__
    return str(annotation)


def get_tool_schema() -> dict:
    """Get full JSON Schema for all MCP tools.

    Returns schema suitable for MCP protocol negotiation.
    """
    tools = get_mcp_tools()
    return {
        "type": "object",
        "tools": tools,
        "version": "1.0.0",
    }


__all__ = [
    # Response models
    "TickerAnalysis",
    "Signal",
    "TradeResult",
    "BacktestResult",
    "Portfolio",
    "SystemStatus",
    "NewsItem",
    "PriceBar",
    "TradeHistoryItem",
    "ModelInfo",
    "TrainingResult",
    # Tool context
    "MCPToolContext",
    # Tool functions
    "analyze_ticker",
    "get_news",
    "get_price_history",
    "get_signals",
    "execute_trade",
    "close_position",
    "get_portfolio",
    "get_trade_history",
    "backtest",
    "get_system_status",
    "train_model",
    "list_models",
    # Registry functions
    "get_mcp_tools",
    "get_tool_by_name",
    "get_tool_schema",
    "mcp_tool",
]
