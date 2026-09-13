"""Abstract broker interface and data models for trading operations.

This module defines the contract that all broker implementations must follow,
along with the data structures used for orders, positions, accounts, and quotes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

#: KRX's trading timezone and clock, from krx-quant-core (shared with
#: scalp-it). Order/quote/account timestamps below default to KST
#: wall-clock time (rather than naive local time) so they mean the same
#: thing regardless of the host machine's timezone -- see Task 9's
#: currency/timezone sweep (redevelopment plan, Phase 2). Re-exported here
#: because much of this package imports them from this module.
from krx_quant_core.market.session import KST, now_kst

__all__ = [
    "KST",
    "Account",
    "BrokerInterface",
    "Order",
    "OrderResult",
    "OrderSide",
    "OrderStatus",
    "OrderStatusEnum",
    "OrderType",
    "Position",
    "Quote",
    "now_kst",
]


class OrderType(StrEnum):
    """Order type enumeration."""

    MARKET = "market"
    LIMIT = "limit"


class OrderSide(StrEnum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


class OrderStatusEnum(StrEnum):
    """Order status enumeration."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Account:
    """Trading account information.

    Attributes:
        account_id: Unique account identifier.
        cash: Available cash balance.
        buying_power: Total buying power available.
        equity: Total account equity (cash + positions value).
        currency: Account currency code (defaults to "KRW" -- this system
            trades KRX-listed KRW-denominated equities).
        created_at: Account creation timestamp (KST by default).
    """

    account_id: str
    cash: Decimal
    buying_power: Decimal
    equity: Decimal
    currency: str = "KRW"
    created_at: datetime = field(default_factory=now_kst)

    def __post_init__(self) -> None:
        """Convert numeric fields to Decimal if needed."""
        if not isinstance(self.cash, Decimal):
            self.cash = Decimal(str(self.cash))
        if not isinstance(self.buying_power, Decimal):
            self.buying_power = Decimal(str(self.buying_power))
        if not isinstance(self.equity, Decimal):
            self.equity = Decimal(str(self.equity))


@dataclass
class Position:
    """Trading position information.

    Attributes:
        ticker: Stock/asset ticker symbol.
        quantity: Number of shares (positive=long, negative=short).
        average_price: Average entry price per share.
        current_price: Current market price.
        market_value: Current market value of position.
        unrealized_pnl: Unrealized profit/loss.
        realized_pnl: Realized profit/loss.
        side: Position side (long/short).
    """

    ticker: str
    quantity: Decimal
    average_price: Decimal
    current_price: Decimal | None = None
    market_value: Decimal | None = None
    unrealized_pnl: Decimal | None = None
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    side: str = field(default="")

    def __post_init__(self) -> None:
        """Convert numeric fields and compute derived values."""
        if not isinstance(self.quantity, Decimal):
            self.quantity = Decimal(str(self.quantity))
        if not isinstance(self.average_price, Decimal):
            self.average_price = Decimal(str(self.average_price))
        if self.current_price is not None and not isinstance(self.current_price, Decimal):
            self.current_price = Decimal(str(self.current_price))
        if not isinstance(self.realized_pnl, Decimal):
            self.realized_pnl = Decimal(str(self.realized_pnl))

        # Compute derived fields
        if not self.side:
            self.side = "long" if self.quantity > 0 else "short"
        if self.current_price is not None:
            if self.market_value is None:
                self.market_value = abs(self.quantity) * self.current_price
            if self.unrealized_pnl is None:
                self.unrealized_pnl = (self.current_price - self.average_price) * self.quantity


@dataclass
class Order:
    """Order to be placed with a broker.

    Attributes:
        ticker: Stock/asset ticker symbol.
        quantity: Number of shares to trade.
        side: Buy or sell.
        order_type: Market or limit order.
        limit_price: Limit price (required for limit orders).
        client_order_id: Client-assigned order ID.
        time_in_force: Order validity period (day, gtc, ioc, fok).
        lot_size: Minimum tradeable unit (shares). KRX board lot is 1 share
            for regular KOSPI/KOSDAQ/KONEX orders; kept configurable since
            some order types (e.g. odd-lot venues) differ. Purely
            informational here -- not enforced by this dataclass.
        tick_size: Minimum price increment (KRW) for `limit_price`, per
            KRX's price-band 호가가격단위 table (see
            `daytrade_it.domain.services.tick_size.tick_size_for`). Purely
            informational here -- not enforced by this dataclass; callers
            that need KRX-valid limit prices should compute/validate this
            via `tick_size_for`/`round_to_tick` before constructing an
            Order.
    """

    ticker: str
    quantity: Decimal
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None
    client_order_id: str | None = None
    time_in_force: str = "day"
    lot_size: int = 1
    tick_size: Decimal | None = None

    def __post_init__(self) -> None:
        """Validate and convert fields."""
        if not isinstance(self.quantity, Decimal):
            self.quantity = Decimal(str(self.quantity))
        if self.limit_price is not None and not isinstance(self.limit_price, Decimal):
            self.limit_price = Decimal(str(self.limit_price))
        if self.tick_size is not None and not isinstance(self.tick_size, Decimal):
            self.tick_size = Decimal(str(self.tick_size))
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit_price is required for limit orders")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")


@dataclass
class OrderResult:
    """Result of placing an order.

    Attributes:
        order_id: Broker-assigned order ID.
        client_order_id: Client-assigned order ID.
        ticker: Stock/asset ticker symbol.
        quantity: Ordered quantity.
        filled_quantity: Quantity filled so far.
        side: Buy or sell.
        order_type: Market or limit order.
        status: Current order status.
        limit_price: Limit price if applicable.
        filled_avg_price: Average fill price.
        submitted_at: Order submission timestamp.
        filled_at: Order fill timestamp.
        message: Additional status message.
    """

    order_id: str
    client_order_id: str | None
    ticker: str
    quantity: Decimal
    filled_quantity: Decimal
    side: OrderSide
    order_type: OrderType
    status: OrderStatusEnum
    limit_price: Decimal | None = None
    filled_avg_price: Decimal | None = None
    submitted_at: datetime = field(default_factory=now_kst)
    filled_at: datetime | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        """Convert numeric fields to Decimal if needed."""
        if not isinstance(self.quantity, Decimal):
            self.quantity = Decimal(str(self.quantity))
        if not isinstance(self.filled_quantity, Decimal):
            self.filled_quantity = Decimal(str(self.filled_quantity))
        if self.limit_price is not None and not isinstance(self.limit_price, Decimal):
            self.limit_price = Decimal(str(self.limit_price))
        if self.filled_avg_price is not None and not isinstance(self.filled_avg_price, Decimal):
            self.filled_avg_price = Decimal(str(self.filled_avg_price))

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatusEnum.FILLED

    @property
    def is_active(self) -> bool:
        """Check if order is still active (pending or partial)."""
        return self.status in (
            OrderStatusEnum.PENDING,
            OrderStatusEnum.SUBMITTED,
            OrderStatusEnum.PARTIAL,
        )


@dataclass
class OrderStatus:
    """Current status of an order.

    Attributes:
        order_id: Broker-assigned order ID.
        status: Current order status.
        filled_quantity: Quantity filled so far.
        remaining_quantity: Quantity remaining to be filled.
        filled_avg_price: Average fill price.
        updated_at: Last status update timestamp.
    """

    order_id: str
    status: OrderStatusEnum
    filled_quantity: Decimal
    remaining_quantity: Decimal
    filled_avg_price: Decimal | None = None
    updated_at: datetime = field(default_factory=now_kst)

    def __post_init__(self) -> None:
        """Convert numeric fields to Decimal if needed."""
        if not isinstance(self.filled_quantity, Decimal):
            self.filled_quantity = Decimal(str(self.filled_quantity))
        if not isinstance(self.remaining_quantity, Decimal):
            self.remaining_quantity = Decimal(str(self.remaining_quantity))
        if self.filled_avg_price is not None and not isinstance(self.filled_avg_price, Decimal):
            self.filled_avg_price = Decimal(str(self.filled_avg_price))


@dataclass
class Quote:
    """Market quote for a ticker.

    Attributes:
        ticker: Stock/asset ticker symbol.
        bid: Best bid price.
        ask: Best ask price.
        bid_size: Size at best bid.
        ask_size: Size at best ask.
        last_price: Last traded price.
        last_size: Size of last trade.
        volume: Total volume traded today.
        timestamp: Quote timestamp (KST by default).
        lot_size: Minimum tradeable unit (shares) for this ticker. KRX board
            lot is 1 share for regular KOSPI/KOSDAQ/KONEX equities.
        tick_size: Minimum price increment (KRW) at `last_price`/`bid`/`ask`,
            per KRX's price-band 호가가격단위 table (see
            `daytrade_it.domain.services.tick_size.tick_size_for`).
            Informational only -- not enforced by this dataclass.
    """

    ticker: str
    bid: Decimal
    ask: Decimal
    bid_size: int = 0
    ask_size: int = 0
    last_price: Decimal | None = None
    last_size: int = 0
    volume: int = 0
    timestamp: datetime = field(default_factory=now_kst)
    lot_size: int = 1
    tick_size: Decimal | None = None

    def __post_init__(self) -> None:
        """Convert numeric fields to Decimal if needed."""
        if not isinstance(self.bid, Decimal):
            self.bid = Decimal(str(self.bid))
        if not isinstance(self.ask, Decimal):
            self.ask = Decimal(str(self.ask))
        if self.last_price is not None and not isinstance(self.last_price, Decimal):
            self.last_price = Decimal(str(self.last_price))
        if self.tick_size is not None and not isinstance(self.tick_size, Decimal):
            self.tick_size = Decimal(str(self.tick_size))

    @property
    def mid_price(self) -> Decimal:
        """Mid of bid/ask, or the one populated side of a one-sided book.

        A KRX stock locked at upper limit (상한가) has no sellers, so Kiwoom
        reports ask=0; one locked at lower limit has bid=0. Averaging with
        the zero side halves the price -- a limit-up stock would read as a
        -50% crash. Returns 0 only when both sides are empty.
        """
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return max(self.bid, self.ask)

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> Decimal:
        """Calculate spread as percentage of mid price."""
        mid = self.mid_price
        if mid == 0:
            return Decimal("0")
        return (self.spread / mid) * 100


class BrokerInterface(ABC):
    """Abstract base class for broker implementations.

    All broker adapters must implement this interface to ensure consistent
    behavior across different brokers (paper, Alpaca, IBKR, etc.).
    """

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the broker.

        Returns:
            True if connection successful, False otherwise.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the broker.

        Cleans up any resources and closes connections.
        """
        ...

    @abstractmethod
    def get_account(self) -> Account:
        """Get current account information.

        Returns:
            Account object with current balances and equity.

        Raises:
            BrokerError: If unable to retrieve account information.
        """
        ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get all open positions.

        Returns:
            List of Position objects for all open positions.

        Raises:
            BrokerError: If unable to retrieve positions.
        """
        ...

    @abstractmethod
    def place_order(self, order: Order) -> OrderResult:
        """Place an order with the broker.

        Args:
            order: Order object with order details.

        Returns:
            OrderResult with order ID and initial status.

        Raises:
            BrokerError: If order placement fails.
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Broker-assigned order ID.

        Returns:
            True if cancellation successful, False otherwise.

        Raises:
            BrokerError: If cancellation fails.
        """
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current status of an order.

        Args:
            order_id: Broker-assigned order ID.

        Returns:
            OrderStatus with current fill status.

        Raises:
            BrokerError: If unable to retrieve order status.
        """
        ...

    @abstractmethod
    def get_quote(self, ticker: str) -> Quote:
        """Get current market quote for a ticker.

        Args:
            ticker: Stock/asset ticker symbol.

        Returns:
            Quote with current bid/ask and last price.

        Raises:
            BrokerError: If unable to retrieve quote.
        """
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if broker is currently connected.

        Returns:
            True if connected, False otherwise.
        """
        ...
