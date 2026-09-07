"""Paper trading broker implementation for simulated trading.

This module provides a fully functional paper trading broker that simulates
realistic order execution with configurable slippage and fees.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from gpt_quant_v2.domain.entities.stock import Exchange
from gpt_quant_v2.domain.services.cost_model import CostModelConfig, KoreanCostModel
from gpt_quant_v2.infrastructure.brokers.base import (
    Account,
    BrokerInterface,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderStatusEnum,
    OrderType,
    Position,
    Quote,
    now_kst,
)

if TYPE_CHECKING:
    from gpt_quant_v2.shared.config.settings import Settings


@dataclass
class PaperBrokerConfig:
    """Configuration for paper trading broker.

    Attributes:
        initial_cash: Starting cash balance.
        cost_model_config: Rates for the shared
            :class:`~gpt_quant_v2.domain.services.cost_model.KoreanCostModel`
            (증권거래세 + 위탁수수료 + slippage) that this broker uses for
            every fill -- see ``domain/services/cost_model.py`` for the
            researched/placeholder rate sources (Task 13). Defaults to
            ``CostModelConfig()`` (the module's default rates) when not
            given.
        market: Exchange tier used to pick the cost model's tax/slippage
            rate. ``PaperBroker`` has no ticker-to-exchange mapping wired
            (see ``cost_model.py``'s module docstring), so this single
            value applies to every ticker traded through this broker
            instance. Defaults to KOSPI.
        simulate_partial_fills: Whether to simulate partial fills.
        fill_probability: Probability of limit order fill (0.0 to 1.0).
    """

    initial_cash: Decimal = field(default_factory=lambda: Decimal("100000"))
    cost_model_config: CostModelConfig = field(default_factory=CostModelConfig)
    market: Exchange = Exchange.KOSPI
    simulate_partial_fills: bool = False
    fill_probability: Decimal = field(default_factory=lambda: Decimal("0.95"))

    def __post_init__(self) -> None:
        """Convert numeric fields to Decimal if needed."""
        if not isinstance(self.initial_cash, Decimal):
            self.initial_cash = Decimal(str(self.initial_cash))
        if not isinstance(self.fill_probability, Decimal):
            self.fill_probability = Decimal(str(self.fill_probability))


@dataclass
class _OrderRecord:
    """Internal order record for tracking."""

    order: Order
    result: OrderResult
    created_at: datetime = field(default_factory=now_kst)


class PaperBroker(BrokerInterface):
    """Paper trading broker for simulated order execution.

    This broker simulates realistic trading with:
    - Virtual cash and position tracking
    - Configurable slippage on market orders
    - Commission calculation
    - Market and limit order support
    - Order history tracking

    Example:
        >>> config = PaperBrokerConfig(initial_cash=Decimal("50000"))
        >>> broker = PaperBroker(config)
        >>> broker.connect()
        True
        >>> order = Order(ticker="AAPL", quantity=Decimal("100"), side=OrderSide.BUY)
        >>> result = broker.place_order(order)
        >>> print(result.status)
        OrderStatusEnum.FILLED
    """

    def __init__(
        self,
        config: PaperBrokerConfig | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize paper broker.

        Args:
            config: Paper broker configuration.
            settings: Application settings (optional, for price feeds).
        """
        self._config = config or PaperBrokerConfig()
        self._settings = settings
        self._connected = False
        self._cost_model = KoreanCostModel(self._config.cost_model_config)

        # Account state
        self._account_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"
        self._cash = self._config.initial_cash
        self._positions: dict[str, Position] = {}

        # Order tracking
        self._orders: dict[str, _OrderRecord] = {}
        self._order_history: list[_OrderRecord] = []

        # Simulated price cache (ticker -> price)
        self._price_cache: dict[str, Decimal] = {}

    def connect(self) -> bool:
        """Establish connection (always succeeds for paper trading)."""
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Disconnect from paper broker."""
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if broker is connected."""
        return self._connected

    def get_account(self) -> Account:
        """Get current account information."""
        self._ensure_connected()
        equity = self._calculate_equity()
        return Account(
            account_id=self._account_id,
            cash=self._cash,
            buying_power=self._cash,
            equity=equity,
            currency="KRW",
        )

    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        self._ensure_connected()
        # Update current prices before returning
        positions = []
        for ticker, pos in self._positions.items():
            if pos.quantity != 0:
                # Update with latest price if available
                if ticker in self._price_cache:
                    pos.current_price = self._price_cache[ticker]
                    pos.market_value = abs(pos.quantity) * pos.current_price
                    pos.unrealized_pnl = (pos.current_price - pos.average_price) * pos.quantity
                positions.append(pos)
        return positions

    def place_order(self, order: Order) -> OrderResult:
        """Place an order and simulate execution."""
        self._ensure_connected()

        order_id = f"PO-{uuid.uuid4().hex[:12].upper()}"
        client_order_id = order.client_order_id or f"CO-{uuid.uuid4().hex[:8].upper()}"

        # Get current price for the ticker
        try:
            quote = self.get_quote(order.ticker)
            current_price = quote.last_price or quote.mid_price
        except ValueError:
            # No price available, use limit price for limit orders or reject
            if order.order_type == OrderType.LIMIT and order.limit_price:
                current_price = order.limit_price
            else:
                return self._create_rejected_result(
                    order_id, client_order_id, order, "No price available for ticker"
                )

        # Check if order can be filled
        if order.order_type == OrderType.LIMIT:
            fill_result = self._check_limit_fill(order, current_price)
            if not fill_result["can_fill"]:
                # Limit order not filled - add to pending orders
                result = OrderResult(
                    order_id=order_id,
                    client_order_id=client_order_id,
                    ticker=order.ticker,
                    quantity=order.quantity,
                    filled_quantity=Decimal("0"),
                    side=order.side,
                    order_type=order.order_type,
                    status=OrderStatusEnum.SUBMITTED,
                    limit_price=order.limit_price,
                )
                self._orders[order_id] = _OrderRecord(order=order, result=result)
                return result
            current_price = fill_result["fill_price"]

        # Calculate fill price (with slippage) + commission + tax via the
        # shared KoreanCostModel (Task 13) -- the same model the backtest
        # engine uses, so paper trading and backtesting never drift into
        # two hand-maintained cost numbers.
        trade_cost = self._cost_model.cost_of_trade(
            price=current_price,
            quantity=order.quantity,
            side=order.side.value,
            market=self._config.market,
        )
        fill_price = trade_cost.fill_price

        # Calculate total cost (증권거래세 is sell-only, so `tax` is 0 on buys)
        total_cost = fill_price * order.quantity + trade_cost.commission + trade_cost.tax

        # Check buying power for buys
        if order.side == OrderSide.BUY and total_cost > self._cash:
            return self._create_rejected_result(
                order_id, client_order_id, order, "Insufficient buying power"
            )

        # Execute the order
        self._execute_fill(order, fill_price, trade_cost.commission, trade_cost.tax)

        # Create result
        now = now_kst()
        result = OrderResult(
            order_id=order_id,
            client_order_id=client_order_id,
            ticker=order.ticker,
            quantity=order.quantity,
            filled_quantity=order.quantity,
            side=order.side,
            order_type=order.order_type,
            status=OrderStatusEnum.FILLED,
            limit_price=order.limit_price,
            filled_avg_price=fill_price,
            submitted_at=now,
            filled_at=now,
        )

        # Record order
        record = _OrderRecord(order=order, result=result)
        self._orders[order_id] = record
        self._order_history.append(record)

        return result

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        self._ensure_connected()

        if order_id not in self._orders:
            return False

        record = self._orders[order_id]
        if record.result.status not in (OrderStatusEnum.PENDING, OrderStatusEnum.SUBMITTED):
            return False

        # Update status to cancelled
        record.result.status = OrderStatusEnum.CANCELLED
        return True

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current status of an order."""
        self._ensure_connected()

        if order_id not in self._orders:
            raise ValueError(f"Order not found: {order_id}")

        record = self._orders[order_id]
        result = record.result

        return OrderStatus(
            order_id=order_id,
            status=result.status,
            filled_quantity=result.filled_quantity,
            remaining_quantity=result.quantity - result.filled_quantity,
            filled_avg_price=result.filled_avg_price,
        )

    def get_quote(self, ticker: str) -> Quote:
        """Get market quote for a ticker.

        For paper trading, returns cached prices or generates simulated ones.
        In production, this would connect to a real market data feed.
        """
        self._ensure_connected()

        if ticker in self._price_cache:
            price = self._price_cache[ticker]
        else:
            raise ValueError(f"No price available for ticker: {ticker}")

        # Simulate bid/ask spread (0.01% to 0.05%)
        spread_pct = Decimal("0.0003")
        half_spread = price * spread_pct / 2

        return Quote(
            ticker=ticker,
            bid=price - half_spread,
            ask=price + half_spread,
            last_price=price,
            bid_size=100,
            ask_size=100,
        )

    def set_price(self, ticker: str, price: Decimal) -> None:
        """Set simulated price for a ticker.

        This method allows test code to inject prices for simulation.

        Args:
            ticker: Stock ticker symbol.
            price: Price to set.
        """
        if not isinstance(price, Decimal):
            price = Decimal(str(price))
        self._price_cache[ticker] = price

        # Update position if exists
        if ticker in self._positions:
            pos = self._positions[ticker]
            pos.current_price = price
            pos.market_value = abs(pos.quantity) * price
            pos.unrealized_pnl = (price - pos.average_price) * pos.quantity

    def get_order_history(self) -> list[OrderResult]:
        """Get history of all orders.

        Returns:
            List of OrderResult objects for all historical orders.
        """
        return [record.result for record in self._order_history]

    def reset(self) -> None:
        """Reset broker to initial state.

        Clears all positions, orders, and resets cash to initial amount.
        """
        self._cash = self._config.initial_cash
        self._positions.clear()
        self._orders.clear()
        self._order_history.clear()
        self._price_cache.clear()

    def _ensure_connected(self) -> None:
        """Raise error if not connected."""
        if not self._connected:
            raise RuntimeError("Broker not connected. Call connect() first.")

    def _calculate_equity(self) -> Decimal:
        """Calculate total account equity."""
        equity = self._cash
        for pos in self._positions.values():
            if pos.market_value is not None:
                equity += pos.market_value
        return equity

    def _check_limit_fill(self, order: Order, current_price: Decimal) -> dict[str, bool | Decimal]:
        """Check if a limit order should be filled.

        Returns dict with 'can_fill' bool and 'fill_price' if fillable.
        """
        if order.limit_price is None:
            return {"can_fill": False, "fill_price": Decimal("0")}

        # Check limit price conditions
        can_fill_price = False
        if order.side == OrderSide.BUY:
            # Buy limit fills if market price <= limit price
            can_fill_price = current_price <= order.limit_price
        else:
            # Sell limit fills if market price >= limit price
            can_fill_price = current_price >= order.limit_price

        if can_fill_price and random.random() < float(self._config.fill_probability):
            return {"can_fill": True, "fill_price": current_price}

        return {"can_fill": False, "fill_price": Decimal("0")}

    def _execute_fill(
        self,
        order: Order,
        fill_price: Decimal,
        commission: Decimal,
        tax: Decimal = Decimal("0"),
    ) -> None:
        """Execute a fill by updating positions and cash.

        `tax` is 증권거래세 (sell-side only, already 0 on buys per
        `KoreanCostModel.transaction_tax`).
        """
        ticker = order.ticker
        quantity = order.quantity

        if order.side == OrderSide.BUY:
            # Deduct cost from cash (tax is always 0 here, but included for
            # symmetry with the sell branch / the cost model's contract)
            cost = fill_price * quantity + commission + tax
            self._cash -= cost

            # Update or create position
            if ticker in self._positions:
                pos = self._positions[ticker]
                if pos.quantity > 0:
                    # Adding to long position - update average price
                    old_cost = pos.quantity * pos.average_price
                    new_cost = quantity * fill_price
                    new_quantity = pos.quantity + quantity
                    pos.average_price = (old_cost + new_cost) / new_quantity
                    pos.quantity = new_quantity
                else:
                    # Covering short position
                    if quantity >= abs(pos.quantity):
                        # Fully cover and possibly go long
                        remaining = quantity - abs(pos.quantity)
                        if remaining > 0:
                            pos.quantity = remaining
                            pos.average_price = fill_price
                        else:
                            pos.quantity = Decimal("0")
                    else:
                        pos.quantity += quantity
            else:
                # New long position
                self._positions[ticker] = Position(
                    ticker=ticker,
                    quantity=quantity,
                    average_price=fill_price,
                    current_price=fill_price,
                )

        else:  # SELL
            # Add proceeds to cash (minus commission and 증권거래세)
            proceeds = fill_price * quantity - commission - tax
            self._cash += proceeds

            # Update or create position
            if ticker in self._positions:
                pos = self._positions[ticker]
                if pos.quantity > 0:
                    # Reducing long position
                    if quantity >= pos.quantity:
                        # Fully close and possibly go short
                        remaining = quantity - pos.quantity
                        if remaining > 0:
                            pos.quantity = -remaining
                            pos.average_price = fill_price
                        else:
                            pos.quantity = Decimal("0")
                    else:
                        pos.quantity -= quantity
                else:
                    # Adding to short position
                    old_cost = abs(pos.quantity) * pos.average_price
                    new_cost = quantity * fill_price
                    new_quantity = abs(pos.quantity) + quantity
                    pos.average_price = (old_cost + new_cost) / new_quantity
                    pos.quantity = -new_quantity
            else:
                # New short position
                self._positions[ticker] = Position(
                    ticker=ticker,
                    quantity=-quantity,
                    average_price=fill_price,
                    current_price=fill_price,
                )

        # Update price cache
        self._price_cache[ticker] = fill_price

        # Clean up zero positions
        if ticker in self._positions and self._positions[ticker].quantity == 0:
            del self._positions[ticker]

    def _create_rejected_result(
        self,
        order_id: str,
        client_order_id: str,
        order: Order,
        message: str,
    ) -> OrderResult:
        """Create a rejected order result."""
        return OrderResult(
            order_id=order_id,
            client_order_id=client_order_id,
            ticker=order.ticker,
            quantity=order.quantity,
            filled_quantity=Decimal("0"),
            side=order.side,
            order_type=order.order_type,
            status=OrderStatusEnum.REJECTED,
            limit_price=order.limit_price,
            message=message,
        )
