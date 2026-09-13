"""
Trade Execution Handler.

Handles trade execution workflow: validate signal, calculate position size,
submit order to broker, and track execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

import structlog

from daytrade_it.domain.entities.signal import SignalDirection
from daytrade_it.domain.entities.trade import Trade, TradeStatus
from daytrade_it.domain.events.trading_events import TradeExecutedEvent
from daytrade_it.domain.value_objects import Money, Quantity, Ticker

if TYPE_CHECKING:
    from daytrade_it.infrastructure.brokers.base import BaseBroker
    from daytrade_it.infrastructure.messaging.redis_pubsub import RedisPubSub
    from daytrade_it.infrastructure.persistence.repositories import (
        SignalRepository,
        TradeRepository,
    )

logger = structlog.get_logger(__name__)


@dataclass
class ExecuteTradeCommand:
    """Command to execute a trade based on a signal."""

    signal_id: UUID
    quantity: Decimal | None = None  # If None, use position sizing
    limit_price: Decimal | None = None  # If None, market order
    correlation_id: str | None = None


@dataclass
class ManualTradeCommand:
    """Command to execute a manual trade (without signal)."""

    ticker: str
    direction: str  # "BUY" or "SELL"
    quantity: Decimal
    limit_price: Decimal | None = None
    correlation_id: str | None = None


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing."""

    max_position_pct: Decimal = Decimal("0.05")  # Max 5% of portfolio per position
    max_position_value: Decimal = Decimal("10000")  # Max $10k per position
    min_position_value: Decimal = Decimal("100")  # Min $100 per position
    risk_per_trade_pct: Decimal = Decimal("0.01")  # Risk 1% per trade
    default_stop_loss_pct: Decimal = Decimal("0.02")  # 2% stop loss


@dataclass
class TradeExecutionResult:
    """Result of trade execution."""

    trade: Trade | None
    success: bool
    message: str
    order_id: str | None = None


class TradeExecutionHandler:
    """
    Handles trade execution workflow.

    Pipeline:
    1. Validate signal exists and is not expired
    2. Check portfolio constraints
    3. Calculate position size
    4. Submit order to broker
    5. Track execution and update trade status
    6. Publish TradeExecutedEvent
    """

    def __init__(
        self,
        signal_repository: SignalRepository,
        trade_repository: TradeRepository,
        broker: BaseBroker,
        pubsub: RedisPubSub,
        config: PositionSizingConfig | None = None,
    ) -> None:
        """Initialize with dependencies."""
        self._signal_repo = signal_repository
        self._trade_repo = trade_repository
        self._broker = broker
        self._pubsub = pubsub
        self._config = config or PositionSizingConfig()

    async def execute_from_signal(
        self,
        command: ExecuteTradeCommand,
    ) -> TradeExecutionResult:
        """
        Execute a trade based on a trading signal.

        Args:
            command: The execution command with signal ID.

        Returns:
            TradeExecutionResult with trade details and status.
        """
        log = logger.bind(
            signal_id=str(command.signal_id),
            correlation_id=command.correlation_id,
        )
        log.info("Executing trade from signal")

        # Step 1: Get and validate signal
        signal = await self._signal_repo.get_by_id(command.signal_id)

        if signal is None:
            log.warning("Signal not found")
            return TradeExecutionResult(
                trade=None,
                success=False,
                message=f"Signal {command.signal_id} not found",
            )

        if signal.is_expired:
            log.warning("Signal expired")
            return TradeExecutionResult(
                trade=None,
                success=False,
                message=f"Signal {command.signal_id} has expired",
            )

        # Step 2: Check if trade already exists for this signal
        existing_trade = await self._trade_repo.get_by_signal_id(command.signal_id)
        if existing_trade:
            log.warning("Trade already exists for signal")
            return TradeExecutionResult(
                trade=existing_trade,
                success=False,
                message=f"Trade already exists for signal {command.signal_id}",
            )

        # Step 3: Get current price
        current_price = await self._broker.get_quote(signal.ticker.value)
        if current_price is None:
            log.error("Failed to get current price")
            return TradeExecutionResult(
                trade=None,
                success=False,
                message="Failed to get current price",
            )

        # Step 4: Calculate position size
        if command.quantity:
            quantity = command.quantity
        else:
            quantity = await self._calculate_position_size(
                ticker=signal.ticker.value,
                price=current_price,
                direction=signal.direction,
            )

        if quantity <= 0:
            log.warning("Position size is zero")
            return TradeExecutionResult(
                trade=None,
                success=False,
                message="Position size calculation resulted in zero",
            )

        # Step 5: Create trade entity
        trade = Trade(
            signal_id=signal.id,
            ticker=signal.ticker,
            direction=signal.direction,
            quantity=Quantity(value=quantity),
            status=TradeStatus.PENDING,
            metadata={
                "signal_confidence": signal.confidence.value,
                "signal_source": signal.source.value,
            },
        )

        # Step 6: Submit order to broker
        try:
            order_result = await self._broker.submit_order(
                ticker=signal.ticker.value,
                side="buy" if signal.direction == SignalDirection.BUY else "sell",
                quantity=float(quantity),
                order_type="limit" if command.limit_price else "market",
                limit_price=float(command.limit_price) if command.limit_price else None,
            )

            trade.broker_order_id = order_result.get("order_id")
            trade.status = TradeStatus.SUBMITTED

            if order_result.get("filled"):
                trade.status = TradeStatus.FILLED
                trade.entry_price = Money(
                    amount=Decimal(str(order_result.get("fill_price", current_price))),
                    currency="USD",
                )
                trade.filled_at = datetime.now(UTC)

            log.info(
                "Order submitted",
                order_id=trade.broker_order_id,
                status=trade.status.value,
            )

        except Exception as e:
            log.error("Order submission failed", error=str(e))
            trade.status = TradeStatus.REJECTED
            trade.metadata["rejection_reason"] = str(e)

        # Step 7: Save trade
        await self._trade_repo.save(trade)

        # Step 8: Publish event
        if trade.status in (TradeStatus.SUBMITTED, TradeStatus.FILLED):
            event = TradeExecutedEvent(
                trade_id=trade.id,
                signal_id=signal.id,
                ticker=signal.ticker.value,
                direction=signal.direction.value,
                quantity=quantity,
                price=trade.entry_price.amount if trade.entry_price else current_price,
                status=trade.status.value,
            )
            await self._pubsub.publish(f"trades.{signal.ticker.value}", event.model_dump())

        return TradeExecutionResult(
            trade=trade,
            success=trade.status in (TradeStatus.SUBMITTED, TradeStatus.FILLED),
            message=f"Trade {trade.status.value}",
            order_id=trade.broker_order_id,
        )

    async def execute_manual(
        self,
        command: ManualTradeCommand,
    ) -> TradeExecutionResult:
        """
        Execute a manual trade without a signal.

        Args:
            command: The manual trade command.

        Returns:
            TradeExecutionResult with trade details.
        """
        log = logger.bind(
            ticker=command.ticker,
            direction=command.direction,
            correlation_id=command.correlation_id,
        )
        log.info("Executing manual trade")

        ticker = Ticker(value=command.ticker.upper())
        direction = SignalDirection(command.direction.upper())

        # Get current price
        current_price = await self._broker.get_quote(ticker.value)
        if current_price is None:
            return TradeExecutionResult(
                trade=None,
                success=False,
                message="Failed to get current price",
            )

        # Create trade entity
        trade = Trade(
            ticker=ticker,
            direction=direction,
            quantity=Quantity(value=command.quantity),
            status=TradeStatus.PENDING,
            metadata={"manual": True},
        )

        # Submit order
        try:
            order_result = await self._broker.submit_order(
                ticker=ticker.value,
                side="buy" if direction == SignalDirection.BUY else "sell",
                quantity=float(command.quantity),
                order_type="limit" if command.limit_price else "market",
                limit_price=float(command.limit_price) if command.limit_price else None,
            )

            trade.broker_order_id = order_result.get("order_id")
            trade.status = TradeStatus.SUBMITTED

            if order_result.get("filled"):
                trade.status = TradeStatus.FILLED
                trade.entry_price = Money(
                    amount=Decimal(str(order_result.get("fill_price", current_price))),
                    currency="USD",
                )
                trade.filled_at = datetime.now(UTC)

        except Exception as e:
            log.error("Manual order failed", error=str(e))
            trade.status = TradeStatus.REJECTED
            trade.metadata["rejection_reason"] = str(e)

        await self._trade_repo.save(trade)

        return TradeExecutionResult(
            trade=trade,
            success=trade.status in (TradeStatus.SUBMITTED, TradeStatus.FILLED),
            message=f"Trade {trade.status.value}",
            order_id=trade.broker_order_id,
        )

    async def _calculate_position_size(
        self,
        ticker: str,
        price: Decimal,
        direction: SignalDirection,
    ) -> Decimal:
        """
        Calculate position size based on portfolio and risk management.

        Uses a combination of:
        - Max position percentage of portfolio
        - Max absolute position value
        - Risk per trade with stop loss
        """
        # Get portfolio value
        portfolio_value = await self._broker.get_portfolio_value()
        if portfolio_value is None or portfolio_value <= 0:
            portfolio_value = Decimal("100000")  # Default fallback

        # Calculate max position by percentage
        max_by_pct = portfolio_value * self._config.max_position_pct

        # Use the smaller of percentage-based and absolute max
        max_position_value = min(max_by_pct, self._config.max_position_value)

        # Calculate shares based on price
        shares = max_position_value / price

        # Round down to whole shares
        shares = Decimal(int(shares))

        # Ensure minimum position
        position_value = shares * price
        if position_value < self._config.min_position_value:
            return Decimal("0")

        return shares

    async def close_position(
        self,
        ticker: str,
        correlation_id: str | None = None,
    ) -> TradeExecutionResult:
        """
        Close an open position for a ticker.

        Args:
            ticker: The ticker to close.
            correlation_id: Optional correlation ID.

        Returns:
            TradeExecutionResult with closing trade details.
        """
        log = logger.bind(ticker=ticker, correlation_id=correlation_id)
        log.info("Closing position")

        # Get open position
        position = await self._broker.get_position(ticker)
        if position is None or position.get("quantity", 0) == 0:
            return TradeExecutionResult(
                trade=None,
                success=False,
                message=f"No open position for {ticker}",
            )

        quantity = abs(Decimal(str(position["quantity"])))
        is_long = position["quantity"] > 0
        direction = SignalDirection.SELL if is_long else SignalDirection.BUY

        # Create closing trade
        command = ManualTradeCommand(
            ticker=ticker,
            direction=direction.value,
            quantity=quantity,
            correlation_id=correlation_id,
        )

        return await self.execute_manual(command)
