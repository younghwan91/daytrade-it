"""Unit tests for RiskManager, focused on Task 11's RiskGate integration.

The risk gate check runs first in validate_trade and can force REJECTED
independent of every other check, on hard-severity disclosures or
sufficiently bearish sentiment. check_force_exit is the separate,
disclosure-only path for existing positions.
"""

from datetime import UTC, datetime
from decimal import Decimal

from daytrade_it.domain.entities import Position, SignalDirection, SignalSource, TradingSignal
from daytrade_it.domain.services.risk_gate import GateAction, RiskGate, RiskGateConfig
from daytrade_it.domain.services.risk_manager import Account, RiskManager, ValidationStatus
from daytrade_it.domain.value_objects import Confidence, Sentiment, Ticker

TICKER = Ticker(value="005930")


def _account(**overrides: object) -> Account:
    defaults: dict = {
        "balance": Decimal("100000000"),
        "equity": Decimal("100000000"),
        "buying_power": Decimal("50000000"),
        "daily_pnl": Decimal("0"),
    }
    defaults.update(overrides)
    return Account(**defaults)


def _signal(**overrides: object) -> TradingSignal:
    defaults: dict = {
        "ticker": TICKER,
        "direction": SignalDirection.BUY,
        "confidence": Confidence(value=0.9),
        "source": SignalSource.COMPOSITE,
        "predicted_return": Decimal("5"),
        "predicted_drawdown": Decimal("-2"),
        "risk_reward_ratio": Decimal("2.5"),
    }
    defaults.update(overrides)
    return TradingSignal(**defaults)


def _position(quantity: Decimal = Decimal("10")) -> Position:
    return Position(
        ticker=TICKER,
        quantity=quantity,
        average_price=Decimal("50000"),
        opened_at=datetime.now(UTC),
    )


class TestValidateTradeRiskGateIntegration:
    def test_hard_disclosure_rejects_regardless_of_other_checks(self) -> None:
        manager = RiskManager()

        result = manager.validate_trade(
            signal=_signal(),
            account=_account(),
            disclosure_type="상장폐지결정",
        )

        assert result.status == ValidationStatus.REJECTED
        assert "Risk gate" in result.reason
        assert result.details["gate_action"] == GateAction.REJECT.value

    def test_bad_sentiment_rejects_a_trade_that_would_otherwise_pass(self) -> None:
        manager = RiskManager()
        sentiment = Sentiment.bearish(score=-0.9, confidence=0.9)

        result = manager.validate_trade(
            signal=_signal(),
            account=_account(),
            sentiment=sentiment,
        )

        assert result.status == ValidationStatus.REJECTED
        assert result.details["gate_action"] == GateAction.REJECT.value

    def test_no_sentiment_no_disclosure_gate_passes_through(self) -> None:
        manager = RiskManager()

        result = manager.validate_trade(signal=_signal(), account=_account())

        assert result.status == ValidationStatus.APPROVED

    def test_routine_disclosure_does_not_block(self) -> None:
        manager = RiskManager()

        result = manager.validate_trade(
            signal=_signal(),
            account=_account(),
            disclosure_type="주요사항보고서(유상증자결정)",
        )

        assert result.status == ValidationStatus.APPROVED

    def test_custom_risk_gate_is_used(self) -> None:
        strict_gate = RiskGate(
            RiskGateConfig(bearish_score_threshold=-0.1, bearish_confidence_threshold=0.1)
        )
        manager = RiskManager(risk_gate=strict_gate)
        mild_sentiment = Sentiment.bearish(score=-0.2, confidence=0.5)

        result = manager.validate_trade(
            signal=_signal(),
            account=_account(),
            sentiment=mild_sentiment,
        )

        assert result.status == ValidationStatus.REJECTED

    def test_gate_check_runs_before_daily_loss_limit_check(self) -> None:
        # Even when the account is already over its daily loss limit, the
        # gate's rejection reason should surface (both would reject, but
        # this confirms ordering/precedence, not just "rejected somehow").
        manager = RiskManager()
        breached_account = _account(daily_pnl=Decimal("-999999"))

        result = manager.validate_trade(
            signal=_signal(),
            account=breached_account,
            disclosure_type="관리종목지정",
        )

        assert result.status == ValidationStatus.REJECTED
        assert "Risk gate" in result.reason


class TestCheckForceExit:
    def test_hard_disclosure_forces_exit(self) -> None:
        manager = RiskManager()

        decision = manager.check_force_exit(TICKER, disclosure_type="횡령ㆍ배임혐의발생")

        assert decision.action == GateAction.FORCE_EXIT
        assert decision.forces_exit

    def test_routine_disclosure_does_not_force_exit(self) -> None:
        manager = RiskManager()

        decision = manager.check_force_exit(TICKER, disclosure_type="자기주식취득결정")

        assert decision.action == GateAction.PASS
        assert not decision.forces_exit

    def test_no_disclosure_does_not_force_exit(self) -> None:
        manager = RiskManager()

        decision = manager.check_force_exit(TICKER, disclosure_type=None)

        assert decision.action == GateAction.PASS
