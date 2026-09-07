"""Regression guard: MCP tools must relay handler-computed risk metrics, never
agent-computed ones (Task 12 / deep-research finding #6).

This test proves `get_signals()`'s `risk_reward_ratio` field is exactly the
value `domain/services/risk_manager.py::RiskManager.calculate_risk_reward`
computed for the same inputs -- i.e. the MCP response model is relaying an
already-computed domain value verbatim, not exposing raw
predicted_return/predicted_drawdown for an LLM/agent caller to "compute
Sharpe/risk-reward itself" from.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from gpt_quant_v2.domain.entities.signal import SignalDirection, SignalSource, TradingSignal
from gpt_quant_v2.domain.services.risk_manager import RiskManager
from gpt_quant_v2.domain.value_objects import Confidence, Ticker
from gpt_quant_v2.interfaces.mcp.tools import MCPToolContext, get_signals


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
