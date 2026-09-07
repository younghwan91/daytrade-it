"""Unit tests for NewsBasedStrategy post-Task-11 retirement of evaluate_signal.

Confirms the sentiment-only standalone signal path
(``NewsBasedStrategy.evaluate_signal``) no longer exists, while the
position-sizing and exit-rule mechanics it still owns keep working.
"""

from datetime import UTC, datetime
from decimal import Decimal

from krx_signal_engine.domain.entities import Position, SignalDirection, SignalSource, TradingSignal
from krx_signal_engine.domain.services.trading_strategy import NewsBasedStrategy, StrategyConfig
from krx_signal_engine.domain.value_objects import Confidence, Price, Ticker

TICKER = Ticker(value="005930")


class TestEvaluateSignalRetired:
    def test_evaluate_signal_no_longer_exists(self) -> None:
        strategy = NewsBasedStrategy()
        assert not hasattr(strategy, "evaluate_signal")

    def test_strategy_config_no_longer_carries_signal_thresholds(self) -> None:
        # min_confidence/min_risk_reward were only ever consumed by the
        # retired evaluate_signal(); they should no longer be part of the
        # config surface.
        config = StrategyConfig()
        assert not hasattr(config, "min_confidence")
        assert not hasattr(config, "min_risk_reward")


class TestCalculatePositionSizeStillWorks:
    def test_position_size_with_current_price_metadata(self) -> None:
        strategy = NewsBasedStrategy(StrategyConfig(max_position_pct=5.0))
        signal = TradingSignal(
            ticker=TICKER,
            direction=SignalDirection.BUY,
            confidence=Confidence(value=0.8),
            source=SignalSource.COMPOSITE,
            metadata={"current_price": 50000},
        )

        shares = strategy.calculate_position_size(signal, account_value=Decimal("10000000"))

        assert shares > 0

    def test_zero_account_value_returns_zero_shares(self) -> None:
        strategy = NewsBasedStrategy()
        signal = TradingSignal(
            ticker=TICKER,
            direction=SignalDirection.BUY,
            confidence=Confidence(value=0.8),
            source=SignalSource.COMPOSITE,
        )

        assert strategy.calculate_position_size(signal, account_value=Decimal("0")) == 0


class TestShouldExitStillWorks:
    def _position(self, quantity: Decimal = Decimal("10")) -> Position:
        return Position(
            ticker=TICKER,
            quantity=quantity,
            average_price=Decimal("50000"),
            opened_at=datetime.now(UTC),
        )

    def test_take_profit_triggers_exit(self) -> None:
        strategy = NewsBasedStrategy(StrategyConfig(take_profit_pct=5.0, stop_loss_pct=2.0))
        position = self._position()

        should_exit, reason = strategy.should_exit(position, Price(value=Decimal("53000")))

        assert should_exit is True
        assert "Take profit" in reason

    def test_stop_loss_triggers_exit(self) -> None:
        strategy = NewsBasedStrategy(StrategyConfig(take_profit_pct=5.0, stop_loss_pct=2.0))
        position = self._position()

        should_exit, reason = strategy.should_exit(position, Price(value=Decimal("48000")))

        assert should_exit is True
        assert "Stop loss" in reason

    def test_within_band_does_not_exit(self) -> None:
        strategy = NewsBasedStrategy(StrategyConfig(take_profit_pct=5.0, stop_loss_pct=2.0))
        position = self._position()

        should_exit, reason = strategy.should_exit(position, Price(value=Decimal("50500")))

        assert should_exit is False
        assert reason == ""
