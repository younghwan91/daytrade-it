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
from uuid import UUID, uuid5

from pydantic import BaseModel, Field

from daytrade_it.shared.runtime_host import require_backtest_host

if TYPE_CHECKING:
    from daytrade_it.application.container import Container

# Type variable for generic tool functions
T = TypeVar("T")


# =============================================================================
# Response Models
# =============================================================================
#
# Computation-discipline convention (Task 12 / deep-research finding #6):
# LLM/agent callers make errors on multi-step derived risk metrics (Sharpe,
# max drawdown, Sortino, win rate, risk/reward ratio, ...) unless those
# values are verified via actual code execution rather than eyeballed from
# raw trade/price lists. Every numeric risk-metric field on the response
# models below (sharpe_ratio, max_drawdown_pct, win_rate, predicted_return,
# predicted_drawdown, risk_reward_ratio, total_value, unrealized_pnl,
# realized_pnl, etc.) MUST be populated ONLY by copying a value that was
# already computed by domain/application code -- a handler
# (application/handlers/*), a domain service (domain/services/risk_manager.py,
# signal_generator.py), a repository-persisted entity field, or a broker
# response -- never left blank/None for an agent to "fill in" by reasoning
# over raw data returned in the same or an earlier tool call, and never
# computed inline in a tool function via multi-step arithmetic an agent
# could instead be tempted to redo itself. Deterministic single-expression
# arithmetic over already-authoritative inputs (e.g. `price * quantity` from
# a broker fill) is fine; re-deriving a *risk* metric (Sharpe/MDD/win-rate/
# risk-reward) inline is not -- route those through the owning domain
# service/handler instead. When adding a new response model or field here,
# preserve this: the tool function's job is to relay/format an
# already-computed value, not to compute one.


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
    """Trading signal.

    predicted_return/predicted_drawdown/risk_reward_ratio are relayed
    verbatim from the persisted ``TradingSignal`` entity (populated by
    ``domain/services/signal_generator.py`` and
    ``domain/services/risk_manager.py::calculate_risk_reward``) -- see the
    computation-discipline convention above. Never recompute these from
    ``predicted_return``/other raw fields inside a tool function.
    """

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
    """Result of a trade execution.

    price/total_value are relayed from the broker's fill response
    (``BrokerResult.filled_avg_price``); total_value's single-expression
    ``price * quantity`` is deterministic arithmetic over that
    already-authoritative broker value, not a re-derived risk metric -- see
    the computation-discipline convention above.
    """

    trade_id: UUID
    #: Broker's own order number (Kiwoom `ord_no`, e.g. "0012345"). Not a
    #: UUID -- `trade_id` is derived from it deterministically.
    broker_order_id: str | None = None
    signal_id: UUID | None = None
    ticker: str
    direction: str
    quantity: int
    price: Decimal
    total_value: Decimal
    status: str
    executed_at: datetime


class BacktestResult(BaseModel):
    """Result of a backtest run.

    sharpe_ratio/max_drawdown_pct/win_rate/total_return_pct and the trade
    counts MUST come from a real backtest engine's computed results, never
    be estimated by an agent or hand-computed inline in the ``backtest()``
    tool function -- see the computation-discipline convention above.

    As of Task 13 (Phase 5), ``backtest()`` below wires a real engine:
    ``application.handlers.backtest.BacktestHandler`` runs a genuine
    trade simulation against Task 6's ``QuantAirflowDBClient`` (real KRX
    daily bars) and the shared ``domain.services.cost_model.KoreanCostModel``
    (증권거래세 + 위탁수수료 + slippage -- see that module for researched vs.
    placeholder rate sources), and every numeric field below is relayed
    verbatim from that handler's computed ``BacktestMetrics``.
    ``is_placeholder`` is ``False`` for these real runs.

    IMPORTANT (Task 12 fix-review finding, still true): a class docstring is
    NOT visible to an MCP tool caller -- ``mcp_tool()`` only registers
    ``name``, ``description``, and the function, so an LLM/agent sees only
    the tool description and the returned JSON, never this docstring. The
    ``is_placeholder`` field below is the mechanism that actually reaches a
    caller: it must be ``False`` whenever these are real handler-computed
    metrics, and would only be ``True`` again if some future caller path
    reintroduces a stub (which should not happen without also reverting
    this docstring and the tool's description).

    WHOLE-BRANCH REVIEW FINDING (unresolved as of this fix wave): engine
    real-ness (``is_placeholder``) and strategy real-ness are two
    different things. See ``strategy_is_placeholder`` below -- it is
    currently ALWAYS ``True`` because ``BacktestHandler._generate_signal()``
    is a demo volume-momentum stub, not this system's real
    ``SignalGenerator``/``RiskGate``. Do not treat ``is_placeholder=False``
    alone as evidence the news-driven strategy was validated.
    """

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
    is_placeholder: bool = Field(
        default=False,
        description=(
            "True if these metrics are hardcoded stub values, not a real "
            "backtest engine's computed output (see Phase 5). Callers "
            "MUST check this before treating sharpe_ratio/max_drawdown_pct/"
            "win_rate/etc. as real performance data."
        ),
    )
    strategy_is_placeholder: bool = Field(
        default=True,
        description=(
            "ALWAYS True today -- distinct from is_placeholder above, and "
            "just as important. is_placeholder=False means the BACKTEST "
            "ENGINE is real: real KRX historical bars (Task 6), a real "
            "Korean cost model (증권거래세 + 위탁수수료 + slippage, Task 13), "
            "and genuinely simulated fills/P&L. It says nothing about the "
            "STRATEGY being tested. `application/handlers/backtest.py`'s "
            "`_generate_signal()` is still an unchanged demo "
            "volume-momentum stub (buy on volume > 1.5x average, exit on "
            "+5%/-2%) -- it never consults this system's real "
            "`domain/services/signal_generator.py::SignalGenerator`, "
            "`risk_gate.py::RiskGate`, `risk_manager.py::RiskManager`, or "
            "any news/sentiment data. A caller seeing is_placeholder=False "
            "must NOT conclude the news-driven strategy was validated -- "
            "only that a toy strategy was backtested with real data and "
            "real costs. strategy_is_placeholder will become False only "
            "once backtest.py is rewired to route signal generation "
            "through the real SignalGenerator/RiskGate (tracked as "
            "follow-up work, not done as of this fix wave -- see PLAN.md "
            "Phase 5 Results)."
        ),
    )


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
    """ML model training result.

    Same hardcoded-placeholder shape as ``BacktestResult`` (see its
    docstring): as of Task 12, ``train_model()`` below returns a fixed
    ``metrics`` dict (accuracy/f1/precision/recall) rather than real
    training output, because it is not wired up to an actual training run.
    ``is_placeholder`` is the caller-visible signal for this -- a class
    docstring is never seen by an MCP tool caller, only the returned JSON.
    """

    model_name: str
    model_version: str
    metrics: dict[str, float]
    training_duration_seconds: float
    samples_used: int
    created_at: datetime
    is_placeholder: bool = Field(
        default=False,
        description=(
            "True if `metrics` are hardcoded stub values, not a real "
            "training run's computed output. Callers MUST check this "
            "before treating accuracy/f1/precision/recall as real."
        ),
    )


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
        backtest_handler: Any | None = None,
        risk_manager: Any | None = None,
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
        #: application.handlers.backtest.BacktestHandler, lazily created by
        #: the backtest() tool below (Task 13) if not injected here via a
        #: DI container -- see backtest()'s docstring.
        self.backtest_handler = backtest_handler
        #: domain.services.risk_manager.RiskManager (wraps Task 11's
        #: RiskGate). Whole-branch review finding #4: RiskManager/RiskGate
        #: had zero production call sites -- execute_trade() below now
        #: calls ctx.risk_manager.validate_trade(...) before place_order().
        #: Defaults to a plain ``RiskManager()`` (default RiskConfig +
        #: RiskGate) if not injected, same lazy-default pattern as
        #: backtest_handler above.
        self.risk_manager = risk_manager
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
        from daytrade_it.domain.services.risk_manager import RiskManager
        from daytrade_it.infrastructure.brokers.base import BrokerInterface
        from daytrade_it.infrastructure.external.openai_client import OpenAIClient
        from daytrade_it.infrastructure.external.tiingo_client import TiingoClient
        from daytrade_it.infrastructure.persistence.repositories import (
            NewsRepository,
            PositionRepository,
            SignalRepository,
            TradeRepository,
        )
        from daytrade_it.shared.config import Settings

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
            risk_manager=container.resolve(RiskManager)
            if container.is_registered(RiskManager)
            else RiskManager(),
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


_BROKER_ORDER_NAMESPACE = UUID("5b0d2f5e-3c1a-4e8a-9f3e-6a2b7c1d9e40")


def _trade_id_from_order_id(order_id: str) -> UUID:
    """Deterministic UUID for a broker order number.

    `UUID(order_id)` raised for real Kiwoom order numbers ("0012345") --
    *after* the order had already been accepted, so a caller saw a failure
    for a live order and could retry into a duplicate.
    """
    return uuid5(_BROKER_ORDER_NAMESPACE, order_id)


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
    ticker: Annotated[str, "KRX 6-digit stock code (e.g., '005930', '000660')"],
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
    ticker: Annotated[str, "KRX 6-digit stock code (e.g., '005930')"],
    hours: Annotated[int, "Hours of news history to fetch"] = 24,
    limit: Annotated[int, "Maximum number of articles"] = 20,
    analyze_sentiment: Annotated[bool, "Whether to analyze sentiment"] = True,
) -> list[NewsItem]:
    """Get recent news articles for a ticker.

    Returns list of news items with optional sentiment analysis including
    scores, labels, and key extracted points.

    NOT WIRED TO A REAL SOURCE (whole-branch review finding #2): this tool
    is still implemented against ``TiingoClient`` (US market news), which
    is no longer constructed anywhere in this application (see
    ``mcp_server.py::create_server``/``main``). Task 7's
    ``infrastructure/external/toss_news_client.py``/``dart_client.py``/
    ``news_poller.py`` (Toss/DART, the actual news source for this KRX
    system) have zero MCP-layer consumers. Always raises until that wiring
    is done -- see ``PLAN.md``'s Phase 1 Results / Task 7 and Phase 6
    Results for the tracked gap.

    Raises:
        NotImplementedError: Always, until Toss/DART is wired in here.
    """
    raise NotImplementedError(
        "get_news() is not wired to a real news source -- Toss/DART "
        "(infrastructure/external/toss_news_client.py, dart_client.py, "
        "news_poller.py, Task 7) have zero MCP-layer consumers yet. See "
        "PLAN.md Phase 1 Results / Task 7 and Phase 6 Results."
    )


@mcp_tool(
    name="get_price_history",
    description="Get historical OHLCV price data for a ticker",
)
async def get_price_history(
    ticker: Annotated[str, "KRX 6-digit stock code (e.g., '005930')"],
    days: Annotated[int, "Number of days of history"] = 30,
    interval: Annotated[str, "Data interval (daily, weekly, monthly)"] = "daily",
) -> list[PriceBar]:
    """Get historical OHLCV price data.

    Returns list of price bars with open, high, low, close, volume.

    NOT WIRED TO A REAL SOURCE (whole-branch review finding #2): this tool
    is still implemented against ``TiingoClient`` (US market data), which
    is no longer constructed anywhere in this application (see
    ``mcp_server.py::create_server``/``main``). Task 6's
    ``infrastructure/external/quant_airflow_db_client.py`` (the real KRX
    daily-bar source, already wired into ``backtest()`` below) has no
    MCP-layer tool exposing it directly yet. See ``PLAN.md``'s Phase 1
    Results / Task 6 and Phase 6 Results for the tracked gap.

    Raises:
        NotImplementedError: Always, until a real KRX price source is
            wired in here.
    """
    raise NotImplementedError(
        "get_price_history() is not wired to a real KRX price source -- "
        "TiingoClient (US market data) is no longer constructed anywhere "
        "in this application. Task 6's QuantAirflowDBClient "
        "(infrastructure/external/quant_airflow_db_client.py) has no "
        "MCP-layer tool exposing it directly yet. See PLAN.md Phase 1 "
        "Results / Task 6 and Phase 6 Results."
    )


@mcp_tool(
    name="get_signals",
    description="Get current trading signals filtered by criteria",
)
async def get_signals(
    min_confidence: Annotated[float, "Minimum confidence threshold (0.0-1.0)"] = 0.7,
    ticker: Annotated[str | None, "Filter by specific KRX 6-digit stock code"] = None,
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


def _default_risk_manager() -> Any:
    """Build a default ``RiskManager`` (default ``RiskConfig`` + ``RiskGate``).

    Lazily imported/constructed, same pattern as ``_default_backtest_handler``
    below -- importing this module shouldn't require constructing domain
    services eagerly, only calling a tool that needs one without an
    injected ``ctx.risk_manager`` does.
    """
    from daytrade_it.domain.services.risk_manager import RiskManager  # noqa: PLC0415

    return RiskManager()


def _risk_manager_account_from_broker(account: Any) -> Any:
    """Map a broker ``Account`` (``infrastructure/brokers/base.py``) to
    ``RiskManager``'s own ``Account`` dataclass (``domain/services/risk_manager.py``).

    The two are structurally similar but distinct types. ``daily_pnl``
    defaults to 0 -- nothing in this MCP context currently tracks
    intraday realized P&L for the account, so ``RiskManager``'s daily-loss
    check is a no-op via this call site until that's wired up separately;
    every other check (risk gate, exposure, buying power, risk/reward,
    position size) still runs against real broker data.
    """
    from daytrade_it.domain.services.risk_manager import (
        Account as RiskAccount,  # noqa: PLC0415
    )

    return RiskAccount(
        balance=account.cash,
        equity=account.equity,
        buying_power=account.buying_power,
        daily_pnl=Decimal("0"),
    )


def _risk_manager_positions_from_broker(positions: list[Any]) -> list[Any]:
    """Map broker ``Position`` objects to domain ``Position`` entities for
    ``RiskManager.validate_trade``'s ``current_positions`` argument.

    Best-effort: the domain entity requires ``opened_at``, which the
    broker's own ``Position`` doesn't carry, so ``now_kst()`` is used as a
    placeholder -- fine here since ``validate_trade``'s exposure/position
    checks read ``ticker``/``quantity``/``market_value``, never
    ``opened_at``. Positions the domain entity would reject (zero
    quantity, non-positive average_price) are skipped rather than raising,
    since risk validation shouldn't fail *because of* a malformed existing
    position -- it should still evaluate the trade against the positions
    it can understand.
    """
    from daytrade_it.domain.entities import Position as DomainPosition  # noqa: PLC0415
    from daytrade_it.domain.value_objects import Ticker as DomainTicker  # noqa: PLC0415
    from daytrade_it.infrastructure.brokers.base import now_kst  # noqa: PLC0415

    mapped: list[Any] = []
    for p in positions:
        if p.quantity == 0 or p.average_price <= 0:
            continue
        try:
            mapped.append(
                DomainPosition(
                    ticker=DomainTicker(value=p.ticker),
                    quantity=p.quantity,
                    average_price=p.average_price,
                    current_price=p.current_price,
                    opened_at=now_kst(),
                )
            )
        except ValueError:
            continue
    return mapped


#: Hard operational cap for the first live-order rollout (user directive
#: 2026-09-13: "종목마다 1주 매매씩만"). The caller's requested `quantity`
#: is never used to size the order -- mirrors scalp-it's C-3 audit finding
#: (`docs/research/manju/41-order-safety-audit.md`: "수량은 상수 1로 고정,
#: 외부 입력에서 절대 유도하지 말 것"). The old "2% of equity" auto-sizer
#: this replaced could have produced a many-share order on a low-priced
#: ticker with no cap at all. Raise this only via a reviewed code change,
#: never a runtime parameter.
_LIVE_TRADE_QUANTITY = 1

#: Reject a LIMIT order whose price differs from the current quote's
#: reference price by more than this fraction -- scalp-it's C-4 finding
#: (stale/zero/wrong-direction reference price reaching the exchange as a
#: garbage limit price). Kiwoom's own sign convention on quote fields
#: (`+`/`-` = direction vs. previous close, not the price's own sign) was
#: found live-broken in `kiwoom_broker._to_decimal` while wiring this check
#: (2026-09-13, fixed) -- this band would otherwise have compared against
#: a negative reference price.
_LIVE_TRADE_PRICE_BAND_PCT = Decimal("0.03")

#: Read-only kill switch: if this file exists, execute_trade refuses every
#: order regardless of signal/risk-gate result. Mirrors scalp-it's
#: `data/pair_STOP`. Created/removed by a human, never by this code.
_KILL_SWITCH_PATH = Path("data/live_STOP")


def _kill_switch_active() -> bool:
    return _KILL_SWITCH_PATH.exists()


@mcp_tool(
    name="execute_trade",
    description="Execute a trade based on a trading signal",
)
async def execute_trade(
    signal_id: Annotated[str, "UUID of the trading signal to execute"],
    quantity: Annotated[
        int | None,
        "IGNORED as of 2026-09-13 -- every live order is hard-capped to "
        f"{_LIVE_TRADE_QUANTITY} share regardless of this value.",
    ] = None,
    order_type: Annotated[str, "Order type -- LIMIT only (see docstring)"] = "LIMIT",
    limit_price: Annotated[float | None, "Limit price (required -- MARKET is disabled)"] = None,
) -> TradeResult:
    """Execute a trade based on a signal, hard-capped to one share.

    Deliberately NOT "auto-sized": every order this tool places uses
    `_LIVE_TRADE_QUANTITY` (1), never a caller-supplied or account-equity
    derived quantity -- see that constant's docstring.

    MARKET orders are refused. `kiwoom_broker._ORDER_TYPE_MARKET` ("03")
    is explicitly flagged unverified in that module's docstring -- nothing
    has ever exercised it against Kiwoom, live or mock. LIMIT is the only
    order type scalp-it has ever placed for the same reason.

    A LIMIT order is further rejected if `limit_price` is more than
    `_LIVE_TRADE_PRICE_BAND_PCT` away from the ticker's current quote, and
    every order is refused outright if `_KILL_SWITCH_PATH` exists.
    """
    ctx = _get_context()

    if not ctx.signal_repository or not ctx.broker:
        raise ValueError("Signal repository and broker must be configured")

    if _kill_switch_active():
        raise ValueError(f"Kill switch active ({_KILL_SWITCH_PATH}) -- no orders will be placed")

    if order_type.upper() != "LIMIT":
        raise ValueError(
            "MARKET orders are disabled: kiwoom_broker's market order-type code "
            "('03') has never been verified against Kiwoom, live or mock. "
            "Use LIMIT with an explicit limit_price."
        )
    if limit_price is None:
        raise ValueError("limit_price is required (MARKET orders are disabled)")

    # Parse signal ID
    sig_uuid = UUID(signal_id)

    # Get the signal
    signal = await ctx.signal_repository.get_by_id(sig_uuid)
    if not signal:
        raise ValueError(f"Signal not found: {signal_id}")

    if signal.is_expired:
        raise ValueError(f"Signal has expired: {signal_id}")

    # Determine order parameters
    from daytrade_it.infrastructure.brokers.base import Order, OrderSide, OrderType

    side = OrderSide.BUY if signal.direction.value == "BUY" else OrderSide.SELL
    otype = OrderType.LIMIT

    quantity = _LIVE_TRADE_QUANTITY

    # Price-sanity band (scalp-it C-4): compare the requested limit price
    # against the ticker's current quote before it ever reaches the
    # exchange. Fail-closed if no reference price is available at all --
    # a limit order with no sanity check behind it is exactly what this
    # gate exists to prevent, not a case to silently let through.
    quote = ctx.broker.get_quote(signal.ticker.value)
    reference_price = quote.last_price or quote.mid_price
    if not reference_price or reference_price <= 0:
        raise ValueError(
            f"No usable reference price for {signal.ticker.value} "
            f"(quote={quote!r}) -- refusing to place an unchecked limit order"
        )
    price_decimal = Decimal(str(limit_price))
    deviation = abs(price_decimal / reference_price - 1)
    if deviation > _LIVE_TRADE_PRICE_BAND_PCT:
        raise ValueError(
            f"limit_price {price_decimal} deviates {deviation:.1%} from reference "
            f"{reference_price} (band: {_LIVE_TRADE_PRICE_BAND_PCT:.0%})"
        )

    # Duplicate-entry guard (scalp-it C-2/C-3 pattern, applied here since
    # this tool has no separate blocklist layer): refuse to open a second
    # position in a ticker we already hold. Fail-closed -- if the
    # positions lookup itself fails, that's treated the same as "already
    # holding" rather than silently proceeding, since we cannot prove the
    # opposite.
    # SELL gets the mirror-image check: KRX retail accounts cannot short, so
    # a SELL signal on a ticker we don't hold is a no-holdings sell
    # (scalp-it audit C-4 "무보유 매도 금지") -- refuse it here rather than
    # let Kiwoom reject it after the fact.
    try:
        current_positions_raw = ctx.broker.get_positions()
    except Exception as exc:
        raise ValueError(
            f"Could not verify current positions before ordering (fail-closed): {exc}"
        ) from exc
    holding = any(p.ticker == signal.ticker.value and p.quantity > 0 for p in current_positions_raw)
    if side == OrderSide.BUY and holding:
        raise ValueError(
            f"Already holding {signal.ticker.value} -- refusing duplicate entry "
            "(use close_position to exit first)"
        )
    if side == OrderSide.SELL and not holding:
        raise ValueError(
            f"Not holding {signal.ticker.value} -- refusing a no-holdings SELL "
            "(KRX retail accounts cannot short)"
        )

    # Risk-gate check (whole-branch review finding #4): RiskManager/RiskGate
    # (Task 11's flagship deliverable) had zero production call sites --
    # every order placed through this tool now must clear
    # RiskManager.validate_trade() first. sentiment/disclosure_type are
    # passed as None here (best-effort): the MCP context has no wired
    # Toss/DART consumer yet (see finding #2 / PLAN.md Phase 6 Results),
    # so the risk gate runs on account/exposure/position/risk-reward
    # checks only until that's wired up -- it does not silently skip the
    # gate, it evaluates it with the data actually available.
    if ctx.risk_manager is None:
        ctx.risk_manager = _default_risk_manager()

    risk_account = _risk_manager_account_from_broker(ctx.broker.get_account())
    risk_positions = _risk_manager_positions_from_broker(ctx.broker.get_positions())
    validation = ctx.risk_manager.validate_trade(
        signal=signal,
        account=risk_account,
        current_positions=risk_positions,
        sentiment=None,
        disclosure_type=None,
    )
    if not validation.is_approved:
        raise ValueError(f"Trade rejected by risk gate: {validation.reason}")

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
        trade_id=_trade_id_from_order_id(result.order_id) if result.order_id else sig_uuid,
        broker_order_id=result.order_id or None,
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
    ticker: Annotated[str, "KRX 6-digit stock code (e.g., '005930')"],
    limit_price: Annotated[float, "Limit price (required -- MARKET orders are disabled)"],
    quantity: Annotated[int | None, "Shares (주) to close (None = close all)"] = None,
) -> TradeResult:
    """Close (sell) a held long position with a LIMIT order.

    If quantity is not specified, sells the entire holding -- including
    shares the automated trader did not buy. Automated exits use
    `AutoTrader.flatten()`, which sells only its own ledger quantity.

    LIMIT-only for the same reason as `execute_trade`: Kiwoom's MARKET
    order-type code has never been verified.

    Deliberately NOT routed through ``RiskManager.validate_trade()``
    (whole-branch review finding #4, considered and rejected for this
    call site): ``RiskGate`` is designed as an entry-blocker, not a
    liquidator -- see ``domain/services/risk_gate.py``'s module docstring
    ("Generic bearish/bad sentiment -> entry-blocker only. Existing
    positions are left alone."). Closing a position is a risk-reducing
    exit; gating it the same way as a new entry could block a trader from
    reducing risk exactly when they most need to, which would be actively
    harmful. If a future caller needs the hard-severity
    FORCE_EXIT path, that's ``RiskGate.evaluate()``/``RiskManager``'s
    ``check_force_exit``-style logic to poll separately, not a gate on
    this tool's own call to ``place_order()``.
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

    held = int(position.quantity)
    if held <= 0:
        raise ValueError(f"No long position to close for {ticker} (quantity={held})")
    close_qty = min(quantity, held) if quantity else held

    from daytrade_it.infrastructure.brokers.base import Order, OrderSide, OrderType

    # Always SELL: KRX retail positions are long-only. The previous
    # `SELL if position.side == "long" else BUY` sent a BUY for every
    # KiwoomBroker position, since that adapter never sets `side`.
    side = OrderSide.SELL

    order = Order(
        ticker=ticker,
        quantity=Decimal(str(close_qty)),
        side=side,
        order_type=OrderType.LIMIT,
        limit_price=Decimal(str(limit_price)),
    )

    result = ctx.broker.place_order(order)

    ctx.update_last_trade_time()

    return TradeResult(
        trade_id=_trade_id_from_order_id(result.order_id)
        if result.order_id
        else UUID("00000000-0000-0000-0000-000000000000"),
        broker_order_id=result.order_id or None,
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
    ticker: Annotated[str | None, "Filter by specific KRX 6-digit stock code"] = None,
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


def _default_backtest_handler() -> Any:
    """Build a default BacktestHandler (Task 6 DB client + KoreanCostModel).

    Lazily imported/constructed so importing this module never requires a
    reachable quant-airflow TimescaleDB -- only calling ``backtest()``
    without an injected ``ctx.backtest_handler`` does.
    """
    from daytrade_it.application.handlers.backtest import BacktestHandler  # noqa: PLC0415
    from daytrade_it.infrastructure.external.quant_airflow_db_client import (  # noqa: PLC0415
        QuantAirflowDBClient,
    )

    return BacktestHandler(market_data_client=QuantAirflowDBClient())


@mcp_tool(
    name="backtest",
    description=(
        "Run a backtest on a trading strategy against real KRX historical "
        "daily bars (Task 6's quant-airflow TimescaleDB client) with a "
        "real Korean transaction-cost model (증권거래세 + 위탁수수료 + "
        "slippage). Requires at least one KRX 6-digit ticker code. Check "
        "the response's is_placeholder field (should be False) before "
        "treating sharpe_ratio/max_drawdown_pct/win_rate/etc. as real "
        "performance data -- BUT ALSO check strategy_is_placeholder "
        "(currently always True): the backtest ENGINE (data/costs/fills) "
        "is real, while the STRATEGY being simulated is still a demo "
        "volume-momentum stub, not this system's real news-driven "
        "SignalGenerator/RiskGate. Do not present these results as "
        "validating the news-driven strategy."
    ),
)
async def backtest(
    strategy: Annotated[
        str, "Strategy name (news_momentum, technical, combined)"
    ] = "news_momentum",
    tickers: Annotated[
        list[str] | None, "KRX 6-digit ticker codes to backtest, e.g. ['005930']"
    ] = None,
    start_date: Annotated[str | None, "Start date (YYYY-MM-DD)"] = None,
    end_date: Annotated[str | None, "End date (YYYY-MM-DD)"] = None,
    initial_capital: Annotated[float, "Starting capital in KRW"] = 100000,
    parameters: Annotated[dict | None, "Strategy-specific parameters"] = None,
) -> BacktestResult:
    """Run a backtest on a trading strategy.

    Returns performance metrics including returns, Sharpe ratio,
    drawdown, and trade statistics.

    WIRED TO A REAL BACKTEST ENGINE (Task 13, Phase 5):
    ``application.handlers.backtest.BacktestHandler`` runs a genuine
    simulation against real KRX daily bars and a real Korean cost model --
    see ``BacktestResult`` docstring. Returns ``is_placeholder=False``.

    STRATEGY IS STILL A STUB (whole-branch review finding, unresolved as
    of this fix wave): ``BacktestHandler._generate_signal()`` is a demo
    volume-momentum rule, not this system's real news-driven
    SignalGenerator/RiskGate. Returns ``strategy_is_placeholder=True`` --
    see ``BacktestResult.strategy_is_placeholder``'s field description.

    Raises:
        ValueError: If `tickers` is empty/None -- a backtest needs at
            least one real ticker to fetch data for; it is not guessed.
    """
    if not tickers:
        raise ValueError(
            "backtest() requires at least one ticker (KRX 6-digit code, "
            "e.g. '005930') -- tickers are not inferred."
        )

    # Parse dates
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    else:
        start_dt = datetime.now(UTC) - timedelta(days=365)

    end_dt = (
        datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
        if end_date
        else datetime.now(UTC)
    )

    require_backtest_host()

    ctx = _get_context()
    if ctx.backtest_handler is None:
        ctx.backtest_handler = _default_backtest_handler()

    from daytrade_it.application.handlers.backtest import BacktestCommand  # noqa: PLC0415

    command = BacktestCommand(
        tickers=tickers,
        start_date=start_dt,
        end_date=end_dt,
        initial_capital=Decimal(str(initial_capital)),
        strategy=strategy,
        params=parameters or {},
    )
    result = await ctx.backtest_handler.handle(command)
    metrics = result.metrics

    # Every numeric field below is relayed verbatim from
    # BacktestHandler._calculate_metrics()'s output (or a single-expression
    # Decimal->float conversion of it) -- see the computation-discipline
    # convention above. Nothing here is re-derived.
    return BacktestResult(
        strategy=strategy,
        start_date=result.start_date.strftime("%Y-%m-%d"),
        end_date=result.end_date.strftime("%Y-%m-%d"),
        initial_capital=result.initial_capital,
        final_capital=result.final_capital,
        total_return_pct=float(metrics.total_return_pct),
        sharpe_ratio=float(metrics.sharpe_ratio),
        max_drawdown_pct=float(metrics.max_drawdown_pct),
        total_trades=metrics.total_trades,
        winning_trades=metrics.winning_trades,
        losing_trades=metrics.losing_trades,
        win_rate=float(metrics.win_rate),
        is_placeholder=False,
        # See BacktestResult.strategy_is_placeholder's field description:
        # the ENGINE (data/costs/simulation) is real, but backtest.py's
        # _generate_signal() is still a demo volume-momentum stub, not this
        # system's real SignalGenerator/RiskGate. Always True until that's
        # rewired -- do not flip this without also doing that rewiring.
        strategy_is_placeholder=True,
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

    # KRX news/market-data source status (whole-branch review finding #2):
    # this used to report a "tiingo" component as "connected" whenever
    # ctx.tiingo_client was set, implying a live, connected market-data
    # source -- but TiingoClient is US market data/news, not this system's
    # actual KRX market. mcp_server.py's create_server()/main() no longer
    # construct a TiingoClient at all, and Task 7's Toss/DART poller
    # (infrastructure/external/news_poller.py) has zero MCP-layer
    # consumers -- see PLAN.md Phase 1 Results / Task 7 and Phase 6
    # Results. Report this honestly rather than claiming a connected
    # source that isn't the market this system trades.
    components["krx_news_market_data"] = {
        "status": "not_wired",
        "type": "market_data",
        "detail": (
            "Toss/DART poller (infrastructure/external/news_poller.py) "
            "exists but has no MCP-layer consumer; get_news()/"
            "get_price_history() raise NotImplementedError. See PLAN.md "
            "Phase 6 Results."
        ),
    }

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
    description=(
        "Trigger ML model training with specified parameters. WARNING: "
        "not yet wired to the real training pipeline -- this currently "
        "returns HARDCODED PLACEHOLDER metrics, not a real trained model's "
        "output. Always check the response's is_placeholder field before "
        "treating accuracy/f1/precision/recall as real."
    ),
)
async def train_model(
    model_name: Annotated[str, "Name for the trained model"],
    tickers: Annotated[list[str] | None, "KRX 6-digit stock codes to train on (None = all)"] = None,
    lookback_days: Annotated[int, "Days of historical data to use"] = 365,
    optimize_hyperparams: Annotated[bool, "Whether to run hyperparameter optimization"] = True,
) -> TrainingResult:
    """Trigger ML model training.

    Trains an XGBoost classification model for trading signal prediction.

    NOT YET WIRED TO THE REAL TRAINING PIPELINE: returns
    ``is_placeholder=True`` with hardcoded stub metrics until a real
    training run is integrated. See ``TrainingResult`` docstring.
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
        is_placeholder=True,
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
