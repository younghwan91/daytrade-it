"""Regression guard: MCP tools must relay handler-computed risk metrics, never
agent-computed ones (Task 12 / deep-research finding #6).

This test proves `get_signals()`'s `risk_reward_ratio` field is exactly the
value `domain/services/risk_manager.py::RiskManager.calculate_risk_reward`
computed for the same inputs -- i.e. the MCP response model is relaying an
already-computed domain value verbatim, not exposing raw
predicted_return/predicted_drawdown for an LLM/agent caller to "compute
Sharpe/risk-reward itself" from.

Also covers the Task 12 fix-review finding: a class docstring is invisible
to an actual MCP tool caller (`mcp_tool()` only registers name/description/
function), so `is_placeholder` is the only caller-visible field proving
whether numeric metrics are real or fabricated. `TrainingResult` still has
no real training pipeline wired up (Phase 5+), so `train_model()` must
keep returning `is_placeholder=True`. `BacktestResult` is different as of
Task 13: `backtest()` is now wired to a real `BacktestHandler`, so it must
return `is_placeholder=False` and relay the handler's actually-computed
metrics verbatim, never re-derive them inline.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest

from daytrade_it.domain.entities.signal import SignalDirection, SignalSource, TradingSignal
from daytrade_it.domain.services.risk_manager import RiskManager
from daytrade_it.domain.value_objects import Confidence, Ticker
from daytrade_it.infrastructure.brokers.base import Account as BrokerAccount
from daytrade_it.infrastructure.brokers.base import (
    OrderResult,
    OrderStatusEnum,
)
from daytrade_it.interfaces.mcp.tools import (
    MCPToolContext,
    backtest,
    execute_trade,
    get_news,
    get_price_history,
    get_signals,
    train_model,
)


class _FakeSignalRepository:
    """Minimal async repository stub returning a fixed, pre-built signal."""

    def __init__(self, signals: list[TradingSignal]) -> None:
        self._signals = signals

    async def find_active(
        self, ticker: str | None = None, min_confidence: float = 0.0
    ) -> list[TradingSignal]:
        return self._signals


@pytest.mark.asyncio
async def test_get_signals_risk_reward_ratio_matches_handler_computation() -> None:
    """`Signal.risk_reward_ratio` from the MCP tool must equal a direct
    `RiskManager.calculate_risk_reward` call on the same predicted values --
    proving it's relayed from domain computation, not left for the agent
    to derive from `predicted_return`/`predicted_drawdown` itself.
    """
    predicted_return = 6.0
    predicted_drawdown = -3.0

    # This is the SAME computation path a real handler (SignalGenerator)
    # uses to populate TradingSignal.risk_reward_ratio -- calling it
    # directly here gives us the independently-computed expected value.
    risk_manager = RiskManager()
    expected_risk_reward = risk_manager.calculate_risk_reward(
        predicted_return=predicted_return,
        predicted_drawdown=predicted_drawdown,
    )

    signal = TradingSignal(
        ticker=Ticker(value="005930"),
        direction=SignalDirection.BUY,
        confidence=Confidence(value=0.8),
        source=SignalSource.COMPOSITE,
        predicted_return=Decimal(str(predicted_return)),
        predicted_drawdown=Decimal(str(predicted_drawdown)),
        risk_reward_ratio=Decimal(str(expected_risk_reward)),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )

    ctx = MCPToolContext(signal_repository=_FakeSignalRepository([signal]))
    MCPToolContext.set_instance(ctx)
    try:
        results = await get_signals(min_confidence=0.0, ticker="005930")
    finally:
        MCPToolContext.set_instance(None)  # type: ignore[arg-type]

    assert len(results) == 1
    returned = results[0]

    # The tool must relay the same value the handler/domain-service path
    # computed -- not a value re-derived (or invented) inline.
    assert returned.risk_reward_ratio == pytest.approx(expected_risk_reward)
    assert returned.risk_reward_ratio == pytest.approx(float(signal.risk_reward_ratio))
    assert returned.predicted_return == pytest.approx(predicted_return)
    assert returned.predicted_drawdown == pytest.approx(predicted_drawdown)


@pytest.mark.asyncio
async def test_backtest_tool_requires_tickers() -> None:
    """As of Task 13, `backtest()` runs a real engine against real KRX
    data -- it can no longer fabricate a response with no tickers given, so
    it must raise rather than silently return stub numbers.
    """
    with pytest.raises(ValueError, match="requires at least one ticker"):
        await backtest(strategy="composite", tickers=None, initial_capital=100000)


@pytest.mark.asyncio
async def test_backtest_tool_relays_real_handler_metrics_not_placeholder() -> None:
    """As of Task 13 (Phase 5), `backtest()` is wired to a real
    `BacktestHandler` and must relay its computed metrics verbatim, with
    `is_placeholder=False` -- the caller-visible signal (per the Task 12
    fix-review finding: a class docstring is invisible to an MCP caller)
    that these are genuine handler-computed numbers, not fabricated ones.
    A fake handler is injected via `MCPToolContext.backtest_handler` so
    this stays a unit test (no live DB dependency) while still proving the
    tool function relays rather than re-derives the numbers.
    """
    from daytrade_it.application.handlers.backtest import (
        BacktestMetrics,
    )
    from daytrade_it.application.handlers.backtest import (
        BacktestResult as HandlerBacktestResult,
    )

    expected_metrics = BacktestMetrics(
        total_return=Decimal("50000"),
        total_return_pct=Decimal("0.05"),
        annualized_return=Decimal("0.2"),
        sharpe_ratio=Decimal("1.35"),
        sortino_ratio=Decimal("1.7"),
        max_drawdown=Decimal("10000"),
        max_drawdown_pct=Decimal("0.02"),
        win_rate=Decimal("0.6"),
        profit_factor=Decimal("1.8"),
        total_trades=10,
        winning_trades=6,
        losing_trades=4,
        avg_win=Decimal("5000"),
        avg_loss=Decimal("2000"),
        avg_trade_duration_hours=Decimal("48"),
        best_trade=Decimal("9000"),
        worst_trade=Decimal("-3000"),
    )

    class _FakeBacktestHandler:
        async def handle(self, command):  # noqa: ANN001
            return HandlerBacktestResult(
                start_date=command.start_date,
                end_date=command.end_date,
                initial_capital=command.initial_capital,
                final_capital=command.initial_capital + expected_metrics.total_return,
                metrics=expected_metrics,
                trades=[],
                equity_curve=[],
            )

    ctx = MCPToolContext(backtest_handler=_FakeBacktestHandler())
    MCPToolContext.set_instance(ctx)
    try:
        result = await backtest(
            strategy="composite", tickers=["005930"], initial_capital=100000
        )
    finally:
        MCPToolContext.set_instance(None)  # type: ignore[arg-type]

    assert result.is_placeholder is False
    # Whole-branch review finding #3: engine real-ness (is_placeholder)
    # and strategy real-ness are different things -- the strategy being
    # simulated is still a demo stub regardless of engine wiring.
    assert result.strategy_is_placeholder is True
    # Every numeric field must be relayed verbatim from the handler's
    # BacktestMetrics, not re-derived inline in the tool function.
    assert result.sharpe_ratio == float(expected_metrics.sharpe_ratio)
    assert result.max_drawdown_pct == float(expected_metrics.max_drawdown_pct)
    assert result.win_rate == float(expected_metrics.win_rate)
    assert result.total_return_pct == float(expected_metrics.total_return_pct)
    assert result.total_trades == expected_metrics.total_trades
    assert result.winning_trades == expected_metrics.winning_trades
    assert result.losing_trades == expected_metrics.losing_trades


@pytest.mark.asyncio
async def test_train_model_tool_flags_placeholder_metrics_in_response_data() -> None:
    """Same pattern as `backtest()`: `train_model()` is not wired to a real
    training pipeline yet, so its `metrics` dict is a hardcoded stub.
    `is_placeholder=True` must be visible in the returned data.
    """

    class _StubModelTrainer:
        """Any truthy object satisfies `if not ctx.model_trainer` in
        train_model(); the placeholder path doesn't actually call it."""

    ctx = MCPToolContext(model_trainer=_StubModelTrainer())
    MCPToolContext.set_instance(ctx)
    try:
        result = await train_model(model_name="test-model")
    finally:
        MCPToolContext.set_instance(None)  # type: ignore[arg-type]

    assert result.is_placeholder is True


class _FakeQuote:
    def __init__(self, price: Decimal) -> None:
        self.last_price = price
        self.mid_price = price


class _FakeBroker:
    """Minimal broker stub for execute_trade() risk-gate wiring tests
    (whole-branch review finding #4)."""

    def __init__(self, account: BrokerAccount) -> None:
        self._account = account
        self.place_order_calls: list[object] = []

    def get_account(self) -> BrokerAccount:
        return self._account

    def get_quote(self, ticker: str) -> _FakeQuote:  # noqa: ARG002
        return _FakeQuote(Decimal("70000"))

    def get_positions(self) -> list:
        return []

    def place_order(self, order):  # noqa: ANN001
        self.place_order_calls.append(order)
        return OrderResult(
            order_id=str(UUID(int=1)),
            client_order_id=None,
            ticker=order.ticker,
            quantity=order.quantity,
            filled_quantity=order.quantity,
            side=order.side,
            order_type=order.order_type,
            status=OrderStatusEnum.FILLED,
            filled_avg_price=Decimal("70000"),
        )


def _make_signal_repository() -> _FakeSignalRepository:
    signal = TradingSignal(
        ticker=Ticker(value="005930"),
        direction=SignalDirection.BUY,
        confidence=Confidence(value=0.8),
        source=SignalSource.COMPOSITE,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )

    class _ById:
        async def get_by_id(self, _sig_id):  # noqa: ANN001
            return signal

    return _ById()


@pytest.mark.asyncio
async def test_execute_trade_rejects_when_risk_manager_disapproves() -> None:
    """Whole-branch review finding #4: RiskManager/RiskGate had zero
    production call sites. execute_trade() must now call
    RiskManager.validate_trade() before place_order() and refuse the
    trade (never touching the broker) when it's not approved -- here,
    insufficient buying power (a real, deterministic RiskManager check).
    """
    account = BrokerAccount(
        account_id="acct-1",
        cash=Decimal("1000000"),
        buying_power=Decimal("1"),  # far below the 20% min-buying-power check
        equity=Decimal("1000000"),
    )
    broker = _FakeBroker(account=account)

    ctx = MCPToolContext(
        signal_repository=_make_signal_repository(),
        broker=broker,
        risk_manager=RiskManager(),
    )
    MCPToolContext.set_instance(ctx)
    try:
        with pytest.raises(ValueError, match="rejected by risk gate"):
            await execute_trade(signal_id=str(UUID(int=0)), quantity=10)
    finally:
        MCPToolContext.set_instance(None)  # type: ignore[arg-type]

    assert broker.place_order_calls == []  # never reached the broker


@pytest.mark.asyncio
async def test_execute_trade_places_order_when_risk_manager_approves() -> None:
    """Same wiring, opposite outcome: a healthy account clears the risk
    gate and the order is actually placed."""
    account = BrokerAccount(
        account_id="acct-1",
        cash=Decimal("10000000"),
        buying_power=Decimal("10000000"),
        equity=Decimal("10000000"),
    )
    broker = _FakeBroker(account=account)

    ctx = MCPToolContext(
        signal_repository=_make_signal_repository(),
        broker=broker,
        risk_manager=RiskManager(),
    )
    MCPToolContext.set_instance(ctx)
    try:
        result = await execute_trade(signal_id=str(UUID(int=0)), quantity=10)
    finally:
        MCPToolContext.set_instance(None)  # type: ignore[arg-type]

    assert len(broker.place_order_calls) == 1
    assert result.status == OrderStatusEnum.FILLED.value


@pytest.mark.asyncio
async def test_get_news_not_wired_to_real_source() -> None:
    """Whole-branch review finding #2: get_news() used to silently return
    [] (or serve stale Tiingo/US data) when no news source was
    configured. It must now raise, pointing at the real gap, rather than
    look like a working-but-empty result."""
    with pytest.raises(NotImplementedError, match="Toss/DART"):
        await get_news(ticker="005930")


@pytest.mark.asyncio
async def test_get_price_history_not_wired_to_real_source() -> None:
    with pytest.raises(NotImplementedError, match="KRX price source"):
        await get_price_history(ticker="005930")
