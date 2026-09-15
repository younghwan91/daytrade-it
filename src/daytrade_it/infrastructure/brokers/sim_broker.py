"""Minute-bar broker for replaying a past session through the live trader.

Fill rules are the spec's (docs/superpowers/specs/2026-09-15-bayes-pipeline-
optimization-design.md), with one parity-gate correction: the quote at minute
m is that bar's open (ask) with a one-tick spread before second 30 and its
close from second 30 (live asks late in a minute matched the close).
Marketable limits fill at once at the quote; resting limits fill on a later
bar that trades through them. Every public call first settles resting orders
against bars that have *finished* before the current minute, and
`get_minute_bars` hides the current minute. The only look-ahead is the quote's
use of the current bar's close, at most 30 s early.
"""

from __future__ import annotations

import datetime as dt
import itertools
from collections.abc import Callable
from decimal import Decimal

from krx_quant_core.market.ticks import tick_size

from daytrade_it.infrastructure.brokers.base import (
    KST,
    Account,
    BrokerInterface,
    MinuteBar,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderStatusEnum,
    Position,
    Quote,
)
from daytrade_it.shared.errors.exceptions import BrokerError

_UP_LOCK = Decimal("1.295")
_DOWN_LOCK = Decimal("0.705")


class SimBroker(BrokerInterface):
    def __init__(
        self,
        bars: dict[str, list[MinuteBar]],
        prev_close: dict[str, Decimal],
        clock: Callable[[], dt.datetime],
        cash: Decimal = Decimal("1100000"),
    ) -> None:
        self._bars = {t: sorted(b, key=lambda x: x.ts) for t, b in bars.items()}
        self._prev_close = prev_close
        self._clock = clock
        self._cash = cash
        self._held: dict[str, int] = {}
        self._avg: dict[str, Decimal] = {}
        self._open: dict[str, dict] = {}
        self._ids = itertools.count(1)
        self.fills: list[dict] = []

    # -- time --------------------------------------------------------------

    def _minute(self) -> dt.datetime:
        return self._clock().replace(tzinfo=None, second=0, microsecond=0)

    def _bar_at(self, ticker: str, minute: dt.datetime) -> MinuteBar | None:
        for b in self._bars.get(ticker, []):
            if b.ts == minute:
                return b
        return None

    def _locked(self, ticker: str, bar: MinuteBar) -> tuple[bool, bool]:
        prev = self._prev_close.get(ticker)
        if not prev or bar.high != bar.low:
            return False, False
        return bar.open >= prev * _UP_LOCK, bar.open <= prev * _DOWN_LOCK

    # -- settlement --------------------------------------------------------

    def _fill(self, oid: str, price: Decimal, ts: dt.datetime | None = None) -> None:
        o = self._open.pop(oid)
        t, qty = o["ticker"], o["qty"]
        if o["side"] == OrderSide.BUY:
            held = self._held.get(t, 0)
            self._avg[t] = (self._avg.get(t, Decimal(0)) * held + price * qty) / (held + qty)
            self._held[t] = held + qty
            self._cash -= price * qty
        else:
            self._held[t] = self._held.get(t, 0) - qty
            self._cash += price * qty
        self.fills.append(
            {
                "ts": ts if ts is not None else self._clock(),
                "ticker": t,
                "side": o["side"].value,
                "qty": qty,
                "price": price,
                "order_id": oid,
            }
        )

    def _sync(self) -> None:
        minute = self._minute()
        for oid, o in list(self._open.items()):
            for b in self._bars.get(o["ticker"], []):
                if b.ts <= o["checked"] or b.ts >= minute:
                    continue
                o["checked"] = b.ts
                up, down = self._locked(o["ticker"], b)
                bar_end = (b.ts + dt.timedelta(minutes=1)).replace(tzinfo=KST)
                if o["side"] == OrderSide.BUY and not up and b.low <= o["limit"]:
                    self._fill(oid, min(o["limit"], b.open), ts=bar_end)
                    break
                if o["side"] == OrderSide.SELL and not down and b.high >= o["limit"]:
                    self._fill(oid, max(o["limit"], b.open), ts=bar_end)
                    break

    # -- BrokerInterface ---------------------------------------------------

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        return None

    @property
    def is_connected(self) -> bool:
        return True

    def get_account(self) -> Account:
        self._sync()
        return Account(
            account_id="sim", cash=self._cash, buying_power=self._cash, equity=self._cash
        )

    def get_positions(self) -> list[Position]:
        self._sync()
        return [
            Position(ticker=t, quantity=Decimal(q), average_price=self._avg[t])
            for t, q in self._held.items()
            if q > 0
        ]

    def get_quote(self, ticker: str) -> Quote:
        self._sync()
        minute = self._minute()
        bar = self._bar_at(ticker, minute)
        if bar is not None:
            # Nearest bar boundary: the open for :00-:29, the close for :30-:59.
            # Live asks late in a minute matched its close, not its open (see
            # tests); this peeks at most 30 s into the current bar.
            price = bar.open if self._clock().second < 30 else bar.close
            up, down = self._locked(ticker, bar)
        else:
            done = [b for b in self._bars.get(ticker, []) if b.ts < minute]
            if not done:
                return Quote(ticker=ticker, bid=Decimal(0), ask=Decimal(0))
            price, up, down = done[-1].close, False, False
        bid = price - tick_size(price)
        return Quote(
            ticker=ticker,
            bid=Decimal(0) if down else bid,
            ask=Decimal(0) if up else price,
            last_price=price,
        )

    def place_order(self, order: Order) -> OrderResult:
        self._sync()
        if order.limit_price is None:
            raise BrokerError("SimBroker takes LIMIT orders only", broker_name="sim")
        quote = self.get_quote(order.ticker)
        oid = f"S{next(self._ids):07d}"
        qty = int(order.quantity)
        if order.side == OrderSide.SELL:
            reserved = sum(
                o["qty"]
                for o in self._open.values()
                if o["ticker"] == order.ticker and o["side"] == OrderSide.SELL
            )
            available = self._held.get(order.ticker, 0) - reserved
            if qty > available:
                raise BrokerError(f"no holdings to sell: {order.ticker}", broker_name="sim")
        self._open[oid] = {
            "ticker": order.ticker,
            "side": order.side,
            "qty": qty,
            "limit": order.limit_price,
            "checked": self._minute(),
        }
        filled_price: Decimal | None = None
        if order.side == OrderSide.BUY and quote.ask > 0 and order.limit_price >= quote.ask:
            filled_price = quote.ask
        if order.side == OrderSide.SELL and quote.bid > 0 and order.limit_price <= quote.bid:
            filled_price = quote.bid
        if filled_price is not None:
            self._fill(oid, filled_price)
        return OrderResult(
            order_id=oid,
            client_order_id=None,
            ticker=order.ticker,
            quantity=order.quantity,
            filled_quantity=Decimal(qty if filled_price is not None else 0),
            side=order.side,
            order_type=order.order_type,
            status=OrderStatusEnum.FILLED
            if filled_price is not None
            else OrderStatusEnum.SUBMITTED,
            limit_price=order.limit_price,
            filled_avg_price=filled_price,
        )

    def cancel_order(self, order_id: str) -> bool:
        self._sync()
        return self._open.pop(order_id, None) is not None

    def get_order_status(self, order_id: str) -> OrderStatus:
        self._sync()
        o = self._open.get(order_id)
        if o is None:
            raise BrokerError(f"Order not found: {order_id}", broker_name="sim")
        return OrderStatus(
            order_id=order_id,
            status=OrderStatusEnum.SUBMITTED,
            filled_quantity=Decimal(0),
            remaining_quantity=Decimal(o["qty"]),
        )

    def get_minute_bars(
        self,
        ticker: str,
        day: dt.date | None = None,  # noqa: ARG002 -- interface parity with BrokerInterface
    ) -> list[MinuteBar]:
        self._sync()
        minute = self._minute()
        return [b for b in self._bars.get(ticker, []) if b.ts < minute]
