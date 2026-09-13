"""Trade entity representing an executed or pending trade."""

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

from pydantic import Field, model_validator

from daytrade_it.domain.entities.base import Entity
from daytrade_it.domain.entities.signal import SignalDirection
from daytrade_it.domain.value_objects import Ticker


class TradeStatus(StrEnum):
    """Status of a trade."""

    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class Trade(Entity):
    """Domain entity representing a trade execution.

    Attributes:
        signal_id: ID of the signal that generated this trade
        ticker: Stock ticker being traded
        direction: BUY or SELL
        quantity: Number of shares
        entry_price: Price at which position was entered
        exit_price: Price at which position was exited (if closed)
        status: Current trade status
        pnl: Realized profit/loss
        fees: Trading fees and commissions
    """

    signal_id: UUID | None = Field(default=None)
    ticker: Ticker
    direction: SignalDirection
    quantity: Decimal = Field(gt=0)
    entry_price: Decimal | None = Field(default=None, gt=0)
    exit_price: Decimal | None = Field(default=None, gt=0)
    status: TradeStatus = Field(default=TradeStatus.PENDING)
    pnl: Decimal | None = Field(default=None)
    fees: Decimal = Field(default=Decimal("0"), ge=0)
    filled_at: datetime | None = Field(default=None)
    closed_at: datetime | None = Field(default=None)

    @model_validator(mode="after")
    def validate_trade_state(self) -> "Trade":
        """Validate trade state consistency."""
        if self.status == TradeStatus.FILLED and self.entry_price is None:
            raise ValueError("Filled trade must have entry_price")
        if self.exit_price is not None and self.entry_price is None:
            raise ValueError("exit_price requires entry_price")
        if self.pnl is not None and self.exit_price is None:
            raise ValueError("pnl requires exit_price")
        return self

    def __str__(self) -> str:
        return f"Trade({self.direction} {self.quantity} {self.ticker.value} @ {self.status})"

    def __repr__(self) -> str:
        return (
            f"Trade(id={self.id!r}, ticker={self.ticker!r}, "
            f"direction={self.direction!r}, quantity={self.quantity!r}, "
            f"status={self.status!r})"
        )

    @property
    def is_open(self) -> bool:
        """Check if trade is open (filled but not closed)."""
        return self.status == TradeStatus.FILLED and self.exit_price is None

    @property
    def is_closed(self) -> bool:
        """Check if trade has been closed."""
        return self.exit_price is not None and self.pnl is not None

    @property
    def is_pending(self) -> bool:
        """Check if trade is pending execution."""
        return self.status == TradeStatus.PENDING

    @property
    def is_cancelled(self) -> bool:
        """Check if trade was cancelled."""
        return self.status == TradeStatus.CANCELLED

    @property
    def notional_value(self) -> Decimal | None:
        """Calculate notional value of the trade."""
        if self.entry_price is None:
            return None
        return self.quantity * self.entry_price

    @property
    def return_pct(self) -> Decimal | None:
        """Calculate return percentage if trade is closed."""
        if self.pnl is None or self.entry_price is None:
            return None
        notional = self.quantity * self.entry_price
        if notional == 0:
            return Decimal("0")
        return (self.pnl / notional) * 100

    def fill(self, price: Decimal, filled_at: datetime | None = None) -> None:
        """Mark trade as filled at given price."""
        from datetime import UTC

        self.entry_price = price
        self.status = TradeStatus.FILLED
        self.filled_at = filled_at or datetime.now(UTC)
        self.touch()

    def close(
        self, exit_price: Decimal, fees: Decimal = Decimal("0"), closed_at: datetime | None = None
    ) -> None:
        """Close the trade and calculate PnL."""
        from datetime import UTC

        if self.entry_price is None:
            raise ValueError("Cannot close trade without entry_price")

        self.exit_price = exit_price
        self.fees += fees
        self.closed_at = closed_at or datetime.now(UTC)

        # Calculate PnL based on direction
        price_diff = exit_price - self.entry_price
        if self.direction == SignalDirection.SELL:
            price_diff = -price_diff

        self.pnl = (price_diff * self.quantity) - self.fees
        self.bump_version()

    def cancel(self) -> None:
        """Cancel the trade."""
        if self.status != TradeStatus.PENDING:
            raise ValueError("Can only cancel pending trades")
        self.status = TradeStatus.CANCELLED
        self.touch()
