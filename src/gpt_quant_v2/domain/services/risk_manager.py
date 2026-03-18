"""Risk management domain service for validating trades and managing exposure."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any

from gpt_quant_v2.domain.entities import Position, TradingSignal
from gpt_quant_v2.domain.value_objects import Ticker


class ValidationStatus(StrEnum):
    """Status of trade validation."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


@dataclass(frozen=True)
class ValidationResult:
    """Result of trade validation.

    Attributes:
        status: Validation status (APPROVED, REJECTED, NEEDS_REVIEW)
        reason: Human-readable reason for the decision
        details: Additional details about the validation
    """

    status: ValidationStatus
    reason: str
    details: dict[str, Any] | None = None

    @property
    def is_approved(self) -> bool:
        """Check if validation was approved."""
        return self.status == ValidationStatus.APPROVED

    @property
    def is_rejected(self) -> bool:
        """Check if validation was rejected."""
        return self.status == ValidationStatus.REJECTED


@dataclass(frozen=True)
class Account:
    """Account information for risk validation.

    Attributes:
        balance: Current account balance
        equity: Total equity including positions
        buying_power: Available buying power
        daily_pnl: Profit/loss for the current day
    """

    balance: Decimal
    equity: Decimal
    buying_power: Decimal
    daily_pnl: Decimal = Decimal("0")


@dataclass(frozen=True)
class RiskConfig:
    """Configuration for risk management limits.

    Attributes:
        max_position_value: Maximum value for a single position
        max_position_shares: Maximum shares in a single position
        daily_loss_limit: Maximum daily loss before trading halts
        max_total_exposure_pct: Maximum total exposure as percent of equity
        min_buying_power_pct: Minimum buying power to maintain as percent
    """

    max_position_value: Decimal = Decimal("50000")
    max_position_shares: int = 1000
    daily_loss_limit: Decimal = Decimal("-5000")
    max_total_exposure_pct: float = 80.0
    min_buying_power_pct: float = 20.0


class RiskManager:
    """Domain service for managing trading risk.

    Validates trades against risk limits and monitors exposure.
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        """Initialize risk manager with configuration.

        Args:
            config: Risk configuration, uses defaults if not provided.
        """
        self.config = config or RiskConfig()

    def check_position_limit(
        self,
        ticker: Ticker,
        quantity: int,
        current_position_quantity: int = 0,
    ) -> bool:
        """Check if position size is within limits.

        Args:
            ticker: Stock ticker.
            quantity: Number of shares to add.
            current_position_quantity: Current position size in shares.

        Returns:
            True if within limits, False otherwise.
        """
        total_quantity = abs(current_position_quantity) + abs(quantity)
        return total_quantity <= self.config.max_position_shares

    def check_daily_loss_limit(self, pnl: Decimal) -> bool:
        """Check if daily P&L is within acceptable limits.

        Args:
            pnl: Current daily profit/loss.

        Returns:
            True if within limits (can continue trading), False if limit breached.
        """
        return pnl > self.config.daily_loss_limit

    def check_total_exposure(self, positions: list[Position]) -> bool:
        """Check if total exposure is within limits.

        Args:
            positions: List of current open positions.

        Returns:
            True if within limits, False otherwise.
        """
        total_exposure = sum(
            abs(pos.market_value or Decimal("0")) for pos in positions
        )
        return total_exposure <= self.config.max_position_value * 10  # Allow 10 positions

    def calculate_risk_reward(
        self,
        predicted_return: float,
        predicted_drawdown: float,
    ) -> float:
        """Calculate risk/reward ratio from predictions.

        Args:
            predicted_return: Expected return percentage (positive).
            predicted_drawdown: Expected maximum drawdown percentage (negative or zero).

        Returns:
            Risk/reward ratio. Higher is better.
        """
        if predicted_drawdown >= 0:
            # No predicted downside risk
            return float("inf") if predicted_return > 0 else 0.0

        return abs(predicted_return / predicted_drawdown)

    def validate_trade(
        self,
        signal: TradingSignal,
        account: Account,
        current_positions: list[Position] | None = None,
    ) -> ValidationResult:
        """Validate a trading signal against risk constraints.

        Performs comprehensive validation including:
        - Daily loss limit check
        - Position size limits
        - Total exposure limits
        - Buying power requirements
        - Risk/reward requirements

        Args:
            signal: The trading signal to validate.
            account: Current account information.
            current_positions: List of current open positions.

        Returns:
            ValidationResult with status and reason.
        """
        current_positions = current_positions or []
        details: dict[str, Any] = {}

        # Check daily loss limit
        if not self.check_daily_loss_limit(account.daily_pnl):
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                reason="Daily loss limit exceeded",
                details={"daily_pnl": str(account.daily_pnl), "limit": str(self.config.daily_loss_limit)},
            )

        # Check total exposure
        if not self.check_total_exposure(current_positions):
            total_exposure = sum(
                abs(pos.market_value or Decimal("0")) for pos in current_positions
            )
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                reason="Total exposure limit exceeded",
                details={"total_exposure": str(total_exposure)},
            )

        # Check buying power
        min_required = account.equity * Decimal(str(self.config.min_buying_power_pct / 100))
        if account.buying_power < min_required:
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                reason="Insufficient buying power",
                details={
                    "buying_power": str(account.buying_power),
                    "required": str(min_required),
                },
            )

        # Check risk/reward if metrics are available
        if signal.has_risk_metrics():
            risk_reward = self.calculate_risk_reward(
                float(signal.predicted_return or 0),
                float(signal.predicted_drawdown or 0),
            )
            details["risk_reward"] = risk_reward

            if risk_reward < 1.5:  # Minimum acceptable R/R
                return ValidationResult(
                    status=ValidationStatus.REJECTED,
                    reason="Risk/reward ratio too low",
                    details={"risk_reward": risk_reward, "minimum": 1.5},
                )

        # Check existing position for the ticker
        existing_position = next(
            (p for p in current_positions if p.ticker == signal.ticker),
            None,
        )
        current_qty = int(existing_position.quantity) if existing_position else 0

        # Estimate position size (simplified - assumes $100/share)
        estimated_shares = 100
        if not self.check_position_limit(signal.ticker, estimated_shares, current_qty):
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                reason="Position size limit exceeded",
                details={
                    "current_position": current_qty,
                    "max_shares": self.config.max_position_shares,
                },
            )

        # All checks passed
        return ValidationResult(
            status=ValidationStatus.APPROVED,
            reason="Trade validated successfully",
            details=details,
        )


__all__ = [
    "Account",
    "RiskConfig",
    "RiskManager",
    "ValidationResult",
    "ValidationStatus",
]
