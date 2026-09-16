"""`CorePaperBroker` -- paper trading on real quotes, fills from krx-quant-core.

`news_signal_poller.py --execute --paper` runs the same `AutoTrader` as live,
but hands it this broker instead of the live `KiwoomBroker`:

* **Market data is real and read-only**: `get_quote` / `get_minute_bars` go
  to the wrapped data broker (the live `KiwoomBroker`, which only *reads*
  here -- this class never calls its `place_order` / `cancel_order`).
* **Orders never leave the process**: `place_order` / `cancel_order` /
  `get_order_status` / `get_positions` are served by krx-quant-core's
  `execution.paper.PaperBroker`, the same fill model its replay engine uses.
* Every `get_quote(ticker)` is fed to `PaperBroker.on_quote` (bid, ask,
  last), so resting paper orders fill against real quotes -- as often as
  AutoTrader looks at the ticker (entries, exit checks each manage tick).
  `get_order_status` and `get_positions` first re-quote the tickers that
  have resting paper orders: AutoTrader checks a pending entry through those
  two calls without quoting it again, so without the refresh a paper entry
  could only ever be cancelled at its TTL.
* `get_account().cash` is a paper balance: the starting `cash` minus buy
  fills plus sell fills (costs are not deducted -- the paper fill model
  doesn't, and AutoTrader only uses cash to size a 1-share cap).

Mapping to `BrokerInterface`: `order_id` is the core `ord_no` ("P000001").
The core broker has no partial fills, so a status is SUBMITTED (resting,
nothing filled), FILLED (all filled) or CANCELLED. An unknown order id raises
`BrokerError`, like `KiwoomBroker` does for an order it can't find. Only
LIMIT orders are accepted (MARKET is refused, same as live execute_trade).
Limit prices are rounded to a valid tick exactly like `KiwoomBroker`
(`round_to_tick(price, KOSPI)`, rounded down) before they reach the paper book.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

from krx_quant_core.execution.events import Fill
from krx_quant_core.execution.orders import OrderIntent
from krx_quant_core.execution.paper import PaperBroker

from daytrade_it.domain.entities.stock import Exchange
from daytrade_it.domain.services.tick_size import round_to_tick
from daytrade_it.infrastructure.brokers.base import (
    Account,
    BrokerInterface,
    MinuteBar,
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
from daytrade_it.shared.errors import BrokerError, OrderRejectedError

__all__ = ["DEFAULT_PAPER_CASH", "CorePaperBroker"]

DEFAULT_PAPER_CASH = Decimal(10_000_000)
_BROKER_NAME = "core-paper"


@dataclass
class _Tracked:
    order: Order
    intent: OrderIntent
    submitted_at: dt.datetime
    fill: Fill | None = None
    cancelled: bool = False


class CorePaperBroker(BrokerInterface):
    """Real read-only market data + krx-quant-core paper fills."""

    def __init__(
        self,
        data_broker: BrokerInterface,
        *,
        cash: Decimal = DEFAULT_PAPER_CASH,
        paper: PaperBroker | None = None,
        clock: Callable[[], dt.datetime] = now_kst,
    ) -> None:
        self._data = data_broker
        self._paper = paper or PaperBroker(fill_basis="through", latency_sec=0.0)
        self._cash = Decimal(cash)
        self._clock = clock
        self._orders: dict[str, _Tracked] = {}

    # -- connection: the data broker's ----------------------------------------

    def connect(self) -> bool:
        return True if self._data.is_connected else self._data.connect()

    def disconnect(self) -> None:
        self._data.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._data.is_connected

    # -- market data: real, and fed to the paper broker ------------------------

    def get_quote(self, ticker: str) -> Quote:
        quote = self._data.get_quote(ticker)
        last = float(quote.last_price) if quote.last_price is not None else None
        fills = self._paper.on_quote(
            ticker, self._clock(), float(quote.bid), float(quote.ask), last=last
        )
        for fill in fills:
            self._record_fill(fill)
        return quote

    def get_minute_bars(self, ticker: str, day: dt.date | None = None) -> list[MinuteBar]:
        return self._data.get_minute_bars(ticker, day)

    # -- account / positions: paper -------------------------------------------

    def get_account(self) -> Account:
        return Account(
            account_id="PAPER", cash=self._cash, buying_power=self._cash, equity=self._cash
        )

    def get_positions(self) -> list[Position]:
        self._refresh_resting()
        return [
            Position(
                ticker=h.code, quantity=Decimal(h.qty), average_price=Decimal(str(h.avg_price))
            )
            for h in self._paper.holdings().values()
            if h.qty > 0
        ]

    # -- orders: paper ----------------------------------------------------------

    def place_order(self, order: Order) -> OrderResult:
        if order.order_type != OrderType.LIMIT or order.limit_price is None:
            raise OrderRejectedError(
                message="paper broker accepts LIMIT orders only",
                broker_name=_BROKER_NAME,
                symbol=order.ticker,
            )
        intent = OrderIntent(
            side="buy" if order.side == OrderSide.BUY else "sell",
            code=order.ticker,
            qty=int(order.quantity),
            # Same tick rounding the live KiwoomBroker applies to ord_uv, so a
            # paper limit rests at the price a live order would have used.
            price=int(round_to_tick(order.limit_price, Exchange.KOSPI)),
        )
        result = self._paper.submit(intent)
        if not result.ok or result.ord_no is None:
            raise OrderRejectedError(
                message=f"paper order rejected: {result.blocked_reason or result.return_msg}",
                broker_name=_BROKER_NAME,
                symbol=order.ticker,
            )
        now = self._clock()
        self._orders[result.ord_no] = _Tracked(order=order, intent=intent, submitted_at=now)
        return OrderResult(
            order_id=result.ord_no,
            client_order_id=order.client_order_id,
            ticker=order.ticker,
            quantity=order.quantity,
            filled_quantity=Decimal(0),
            side=order.side,
            order_type=order.order_type,
            status=OrderStatusEnum.SUBMITTED,
            limit_price=order.limit_price,
            submitted_at=now,
        )

    def cancel_order(self, order_id: str) -> bool:
        tracked = self._orders.get(order_id)
        if tracked is None or tracked.fill is not None or tracked.cancelled:
            return False
        result = self._paper.cancel(order_id, tracked.intent)
        if result.return_code != 0:
            return False
        tracked.cancelled = True
        return True

    def get_order_status(self, order_id: str) -> OrderStatus:
        tracked = self._orders.get(order_id)
        if tracked is None:
            raise BrokerError(f"Order not found: {order_id}", broker_name=_BROKER_NAME)
        if tracked.fill is None and not tracked.cancelled:
            self._refresh_resting(tracked.order.ticker)
        qty = tracked.order.quantity
        if tracked.fill is not None:
            return OrderStatus(
                order_id=order_id,
                status=OrderStatusEnum.FILLED,
                filled_quantity=Decimal(tracked.fill.qty),
                remaining_quantity=qty - tracked.fill.qty,
                filled_avg_price=Decimal(tracked.fill.price),
                updated_at=tracked.fill.ts,
            )
        if tracked.cancelled:
            return OrderStatus(
                order_id=order_id,
                status=OrderStatusEnum.CANCELLED,
                filled_quantity=Decimal(0),
                remaining_quantity=Decimal(0),
            )
        return OrderStatus(
            order_id=order_id,
            status=OrderStatusEnum.SUBMITTED,
            filled_quantity=Decimal(0),
            remaining_quantity=qty,
        )

    def _refresh_resting(self, ticker: str | None = None) -> None:
        """Quote each ticker with a resting paper order (or just `ticker`)."""
        codes = {o.code for o in self._paper.open_orders()}
        for code in sorted(codes if ticker is None else codes & {ticker}):
            self.get_quote(code)

    def _record_fill(self, fill: Fill) -> None:
        notional = Decimal(fill.price) * fill.qty
        self._cash += -notional if fill.side == "buy" else notional
        tracked = self._orders.get(fill.ord_no)
        if tracked is not None:
            tracked.fill = fill
