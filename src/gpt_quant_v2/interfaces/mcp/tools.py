"""
MCP Tool Definitions for AI Agents

This module defines MCP-compatible tools that AI agents can use
to interact with the trading system.
"""

from datetime import datetime
from decimal import Decimal
from typing import Annotated, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# Response Models
class TickerAnalysis(BaseModel):
    """Analysis result for a stock ticker."""
    
    ticker: str
    name: str
    exchange: str
    current_price: Decimal
    price_change_pct: float
    sentiment: Optional[dict] = None
    recent_news: List[dict] = Field(default_factory=list)
    signals: List[dict] = Field(default_factory=list)
    analysis_timestamp: datetime


class Signal(BaseModel):
    """Trading signal."""
    
    id: UUID
    ticker: str
    direction: str
    confidence: float
    source: str
    predicted_return: float
    predicted_drawdown: float
    risk_reward_ratio: float
    expires_at: datetime
    created_at: datetime


class TradeResult(BaseModel):
    """Result of a trade execution."""
    
    trade_id: UUID
    signal_id: UUID
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
    positions: List[dict]
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    updated_at: datetime


class SystemStatus(BaseModel):
    """System health status."""
    
    status: str
    components: dict
    uptime_seconds: int
    last_signal_at: Optional[datetime]
    last_trade_at: Optional[datetime]


# Tool Definitions (decorators would be applied by MCP framework)
def mcp_tool(name: str, description: str):
    """Decorator to mark a function as an MCP tool."""
    def decorator(func):
        func._mcp_tool = True
        func._mcp_name = name
        func._mcp_description = description
        return func
    return decorator


@mcp_tool(
    name="analyze_ticker",
    description="Get comprehensive analysis for a stock ticker including price data, news sentiment, and active signals"
)
async def analyze_ticker(
    ticker: Annotated[str, "Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"],
    include_news: Annotated[bool, "Whether to include recent news analysis"] = True,
    include_signals: Annotated[bool, "Whether to include active trading signals"] = True,
    news_hours: Annotated[int, "Hours of news history to analyze"] = 24,
) -> TickerAnalysis:
    """
    Analyze a stock ticker with price data, news sentiment, and signals.
    
    Returns comprehensive analysis including current price, recent news
    with sentiment scores, and any active trading signals.
    """
    # Placeholder implementation
    raise NotImplementedError("Connect to application layer")


@mcp_tool(
    name="get_news_sentiment",
    description="Get sentiment analysis for recent news about a ticker"
)
async def get_news_sentiment(
    ticker: Annotated[str, "Stock ticker symbol"],
    hours: Annotated[int, "Hours of news to analyze"] = 24,
    limit: Annotated[int, "Maximum number of articles"] = 10,
) -> List[dict]:
    """
    Get recent news articles with sentiment analysis.
    
    Returns list of news items with sentiment scores, confidence,
    and key extracted points.
    """
    raise NotImplementedError("Connect to application layer")


@mcp_tool(
    name="get_signals",
    description="Get current trading signals filtered by criteria"
)
async def get_signals(
    min_confidence: Annotated[float, "Minimum confidence threshold (0.0-1.0)"] = 0.7,
    ticker: Annotated[Optional[str], "Filter by specific ticker"] = None,
    direction: Annotated[Optional[str], "Filter by direction (BUY/SELL)"] = None,
    limit: Annotated[int, "Maximum signals to return"] = 20,
) -> List[Signal]:
    """
    Get active trading signals.
    
    Returns signals that haven't expired, filtered by the specified criteria.
    """
    raise NotImplementedError("Connect to application layer")


@mcp_tool(
    name="execute_trade",
    description="Execute a trade based on a trading signal"
)
async def execute_trade(
    signal_id: Annotated[str, "UUID of the trading signal to execute"],
    quantity: Annotated[Optional[int], "Number of shares (None = auto-size)"] = None,
    order_type: Annotated[str, "Order type (MARKET/LIMIT)"] = "MARKET",
    limit_price: Annotated[Optional[float], "Limit price (required for LIMIT orders)"] = None,
) -> TradeResult:
    """
    Execute a trade based on a signal.
    
    If quantity is not specified, position sizing will be calculated
    based on risk management rules.
    """
    raise NotImplementedError("Connect to application layer")


@mcp_tool(
    name="close_position",
    description="Close an open position for a ticker"
)
async def close_position(
    ticker: Annotated[str, "Stock ticker symbol"],
    quantity: Annotated[Optional[int], "Shares to close (None = close all)"] = None,
) -> TradeResult:
    """
    Close an open position.
    
    If quantity is not specified, closes the entire position.
    """
    raise NotImplementedError("Connect to application layer")


@mcp_tool(
    name="get_portfolio",
    description="Get current portfolio positions and value"
)
async def get_portfolio() -> Portfolio:
    """
    Get the current portfolio state.
    
    Returns cash balance, positions, and P&L metrics.
    """
    raise NotImplementedError("Connect to application layer")


@mcp_tool(
    name="backtest",
    description="Run a backtest on a trading strategy"
)
async def backtest(
    strategy: Annotated[str, "Strategy name (news_momentum, technical, combined)"] = "news_momentum",
    start_date: Annotated[str, "Start date (YYYY-MM-DD)"] = None,
    end_date: Annotated[str, "End date (YYYY-MM-DD)"] = None,
    initial_capital: Annotated[float, "Starting capital in USD"] = 100000,
    parameters: Annotated[Optional[dict], "Strategy-specific parameters"] = None,
) -> BacktestResult:
    """
    Run a backtest on a trading strategy.
    
    Returns performance metrics including returns, Sharpe ratio,
    drawdown, and trade statistics.
    """
    raise NotImplementedError("Connect to application layer")


@mcp_tool(
    name="get_system_status",
    description="Get system health status and component states"
)
async def get_system_status() -> SystemStatus:
    """
    Get the current system health status.
    
    Returns status of all system components (database, Redis, APIs)
    and recent activity timestamps.
    """
    raise NotImplementedError("Connect to application layer")


@mcp_tool(
    name="get_price_data",
    description="Get historical price data for a ticker"
)
async def get_price_data(
    ticker: Annotated[str, "Stock ticker symbol"],
    period: Annotated[str, "Time period (1d, 5d, 1mo, 3mo, 1y)"] = "1mo",
    interval: Annotated[str, "Data interval (1m, 5m, 15m, 1h, 1d)"] = "1d",
) -> List[dict]:
    """
    Get historical OHLCV price data.
    
    Returns list of price bars with open, high, low, close, volume.
    """
    raise NotImplementedError("Connect to application layer")


# Tool registry for MCP framework
def get_mcp_tools() -> List[dict]:
    """Get all registered MCP tools with their metadata."""
    import inspect
    
    tools = []
    for name, obj in globals().items():
        if callable(obj) and hasattr(obj, '_mcp_tool'):
            sig = inspect.signature(obj)
            params = []
            for param_name, param in sig.parameters.items():
                if param_name in ('return',):
                    continue
                annotation = param.annotation
                param_info = {
                    "name": param_name,
                    "description": str(annotation) if annotation != inspect.Parameter.empty else "",
                    "required": param.default == inspect.Parameter.empty,
                }
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = param.default
                params.append(param_info)
            
            tools.append({
                "name": obj._mcp_name,
                "description": obj._mcp_description,
                "parameters": params,
            })
    
    return tools
