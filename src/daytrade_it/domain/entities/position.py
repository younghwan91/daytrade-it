"""Position entity representing an open trading position."""

from datetime import datetime
from decimal import Decimal

from pydantic import Field, field_validator, model_validator

from daytrade_it.domain.entities.base import Entity
from daytrade_it.domain.value_objects import Ticker


class Position(Entity):
    """Domain entity representing an open position in a security.

    Attributes:
        ticker: Stock ticker for the position
        quantity: Number of shares held (positive=long, negative=short)
        average_price: Average entry price per share
        current_price: Current market price per share
        unrealized_pnl: Unrealized profit/loss
        opened_at: When the position was opened
    """

    ticker: Ticker
    quantity: Decimal
    average_price: Decimal = Field(gt=0)
    current_price: Decimal | None = Field(default=None, gt=0)
    unrealized_pnl: Decimal | None = Field(default=None)
    opened_at: datetime

    @field_validator("quantity")
    @classmethod
    def quantity_must_not_be_zero(cls, v: Decimal) -> Decimal:
        """수량 0 인 포지션은 존재할 수 없다.

        pydantic v2 의 Field 에는 `ne` 가 없다. `Field(ne=0)` 은 조용히 무시돼
        제약이 전혀 걸리지 않았다(경고만 났다). 검증기로 되살린다.
        """
        if v == 0:
            raise ValueError("quantity must not be zero")
        return v

    @model_validator(mode="after")
    def compute_unrealized_pnl(self) -> "Position":
        """Compute unrealized PnL if current price is set and not already computed."""
        if self.current_price is not None and self.unrealized_pnl is None:
            price_diff = self.current_price - self.average_price
            # Use object.__setattr__ to avoid triggering validate_assignment
            object.__setattr__(self, "unrealized_pnl", price_diff * self.quantity)
        return self

    def __str__(self) -> str:
        direction = "LONG" if self.is_long else "SHORT"
        return f"Position({direction} {abs(self.quantity)} {self.ticker.value})"

    def __repr__(self) -> str:
        return (
            f"Position(id={self.id!r}, ticker={self.ticker!r}, "
            f"quantity={self.quantity!r}, average_price={self.average_price!r})"
        )

    @property
    def is_long(self) -> bool:
        """Check if this is a long position."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if this is a short position."""
        return self.quantity < 0

    @property
    def market_value(self) -> Decimal | None:
        """Calculate current market value of position."""
        if self.current_price is None:
            return None
        return abs(self.quantity) * self.current_price

    @property
    def cost_basis(self) -> Decimal:
        """Calculate total cost basis of position."""
        return abs(self.quantity) * self.average_price

    @property
    def return_pct(self) -> Decimal | None:
        """Calculate return percentage."""
        if self.unrealized_pnl is None:
            return None
        cost_basis = self.cost_basis
        if cost_basis == 0:
            return Decimal("0")
        return (self.unrealized_pnl / cost_basis) * 100

    def update_price(self, price: Decimal) -> None:
        """Update current price and recalculate unrealized PnL."""
        self.current_price = price
        price_diff = price - self.average_price
        self.unrealized_pnl = price_diff * self.quantity
        self.touch()

    def add_to_position(self, quantity: Decimal, price: Decimal) -> None:
        """Add shares to the position, updating average price."""
        if quantity == 0:
            raise ValueError("quantity must be non-zero")

        # Check direction consistency
        if (self.is_long and quantity < 0) or (self.is_short and quantity > 0):
            raise ValueError("Use reduce_position to decrease position size")

        # Calculate new average price
        old_cost = self.quantity * self.average_price
        new_cost = quantity * price
        new_quantity = self.quantity + quantity

        self.average_price = (old_cost + new_cost) / new_quantity
        self.quantity = new_quantity

        # Recalculate unrealized PnL
        if self.current_price is not None:
            price_diff = self.current_price - self.average_price
            self.unrealized_pnl = price_diff * self.quantity

        self.bump_version()

    def reduce_position(self, quantity: Decimal) -> Decimal:
        """Reduce position by given quantity, returns realized PnL."""
        if quantity <= 0:
            raise ValueError("quantity must be positive")

        abs_quantity = abs(self.quantity)
        if quantity > abs_quantity:
            raise ValueError("Cannot reduce by more than position size")

        # Calculate realized PnL for the reduced portion
        realized_pnl = Decimal("0")
        if self.current_price is not None:
            price_diff = self.current_price - self.average_price
            realized_pnl = price_diff * quantity if self.is_long else price_diff * (-quantity)

        # Update quantity
        if self.is_long:
            self.quantity -= quantity
        else:
            self.quantity += quantity

        # Recalculate unrealized PnL
        if self.current_price is not None and self.quantity != 0:
            price_diff = self.current_price - self.average_price
            self.unrealized_pnl = price_diff * self.quantity
        else:
            self.unrealized_pnl = Decimal("0")

        self.bump_version()
        return realized_pnl
