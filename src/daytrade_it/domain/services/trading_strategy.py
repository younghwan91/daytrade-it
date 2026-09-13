"""Trading strategy domain service for position sizing and exit rules.

Sentiment-only signal path retired (Task 11 / Phase 4b):
    ``NewsBasedStrategy`` previously had an ``evaluate_signal(news,
    sentiment)`` method that produced a full BUY/SELL ``TradingSignal``
    from sentiment alone, with no ML/technical corroboration -- exactly the
    pattern deep-research finding #1 warns against (news sentiment alone
    does not reliably beat a benchmark; it only helps combined with
    technical/ML signals, see ``signal_generator.py``'s module docstring).

    That method has been removed. It was dead code outside its own
    module (grep across ``src/`` and ``docs/`` found no callers anywhere
    in the application/infrastructure layers, and no tests), so removal
    was a clean retirement rather than a refactor of a live call path. The
    equivalent, disciplined replacement is
    ``SignalGenerator.generate_from_news()`` (produces a NEWS-sourced
    *component* signal) combined with an ML/technical signal via
    ``SignalGenerator.combine_signals()``, which now structurally refuses
    to produce an actionable signal from sentiment alone (see that
    method's docstring) -- plus, for anything sentiment/disclosure should
    be allowed to do *on its own* (block entries, force-exit on severe
    disclosures), the new ``domain.services.risk_gate.RiskGate`` service.

    ``NewsBasedStrategy`` remains as a position-sizing and exit-rule
    helper (``calculate_position_size`` / ``should_exit``) -- those are
    generic risk/execution mechanics that operate on an already-produced
    signal or open position, not sentiment-only signal generation, so they
    are out of scope for this retirement.
"""

from dataclasses import dataclass
from decimal import Decimal

from daytrade_it.domain.entities import Position, TradingSignal
from daytrade_it.domain.value_objects import Price


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration parameters for trading strategy.

    Attributes:
        take_profit_pct: Take profit percentage (e.g., 5.0 for 5%)
        stop_loss_pct: Stop loss percentage (e.g., 2.0 for 2%)
        max_position_pct: Maximum position size as percentage of account value
    """

    take_profit_pct: float = 5.0
    stop_loss_pct: float = 2.0
    max_position_pct: float = 5.0


class NewsBasedStrategy:
    """Position sizing and exit-rule helper for news/sentiment-informed trading.

    Historically also generated trading signals directly from sentiment
    (``evaluate_signal``); that method was retired -- see module docstring.
    What remains here is signal-agnostic: position sizing given an already
    -produced signal, and take-profit/stop-loss exit checks given an open
    position.
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        """Initialize strategy with configuration.

        Args:
            config: Strategy configuration, uses defaults if not provided.
        """
        self.config = config or StrategyConfig()

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


__all__ = [
    "NewsBasedStrategy",
    "StrategyConfig",
]
