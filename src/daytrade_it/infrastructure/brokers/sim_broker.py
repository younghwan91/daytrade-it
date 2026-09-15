"""Minute-bar broker for replaying a past session through the live trader.

Fill rules are the spec's (docs/superpowers/specs/2026-09-15-bayes-pipeline-
optimization-design.md): the quote at minute m is that bar's open (ask) with a
one-tick spread; marketable limits fill at once at the quote; resting limits
fill on a later bar that trades through them. Every public call first settles
resting orders against bars that have *finished* before the current minute,
and `get_minute_bars` hides the current minute -- nothing after "now" leaks.

Risk inputs are live-like: positions are marked at the price the broker
quotes now (this minute's open, else the last completed close), equity is
cash plus that value, buying power is cash (no margin), and a BUY whose fill
would exceed cash is rejected -- at placement if marketable, or cancelled
with a `rejections` record when a resting order would fill.
"""

from __future__ import annotations

import bisect
import datetime as dt
import itertools
from collections.abc import Callable
from decimal import Decimal

from krx_quant_core.market.limits import limit_down_price, limit_up_price
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


class SimBroker(BrokerInterface):
    def __init__(
        self,
        bars: dict[str, list[MinuteBar]],
        prev_close: dict[str, Decimal],
        clock: Callable[[], dt.datetime],
        cash: Decimal = Decimal("1100000"),
        limit_base: dict[str, Decimal] | None = None,
    ) -> None:
        self._bars = {t: sorted(b, key=lambda x: x.ts) for t, b in bars.items()}
        self._ts = {t: [b.ts for b in bs] for t, bs in self._bars.items()}
        self._by_minute = {t: {b.ts: b for b in bs} for t, bs in self._bars.items()}
        self._prev_close = prev_close
        self._limit_base = limit_base or {}
        self._clock = clock
        self._cash = cash
        self._held: dict[str, int] = {}
        self._avg: dict[str, Decimal] = {}
        self._open: dict[str, dict] = {}
        self._ids = itertools.count(1)
        self.fills: list[dict] = []
        #: Resting BUYs cancelled at fill time for lack of cash.
        self.rejections: list[dict] = []

    # -- time --------------------------------------------------------------

    def _minute(self) -> dt.datetime:
        return self._clock().replace(tzinfo=None, second=0, microsecond=0)

    def _bar_at(self, ticker: str, minute: dt.datetime) -> MinuteBar | None:
        return self._by_minute.get(ticker, {}).get(minute)

    def _done(self, ticker: str, minute: dt.datetime) -> list[MinuteBar]:
        """Bars whose minute ended at or before `minute` starts."""
        return self._bars.get(ticker, [])[: bisect.bisect_left(self._ts.get(ticker, []), minute)]

    def _limits(self, ticker: str) -> tuple[Decimal, Decimal] | None:
        prev = self._limit_base.get(ticker) or self._prev_close.get(ticker)
        if not prev:
            return None
        up = Decimal(str(limit_up_price(float(prev))))
        return up, Decimal(str(limit_down_price(float(prev))))

    def _open_locked(self, ticker: str, bar: MinuteBar) -> tuple[bool, bool]:
        """Quote-time lock: this minute's open sits at the price limit.

        Only the open is known during the minute; its high/low are not.
        """
        lim = self._limits(ticker)
        if lim is None:
            return False, False
        return bar.open >= lim[0], bar.open <= lim[1]

    def _bar_locked(self, ticker: str, bar: MinuteBar) -> tuple[bool, bool]:
        """Completed-bar lock for resting fills: pinned (high == low) at the limit."""
        lim = self._limits(ticker)
        if lim is None or bar.high != bar.low:
            return False, False
        return bar.open >= lim[0], bar.open <= lim[1]

    def _mark(self, ticker: str) -> Decimal | None:
        minute = self._minute()
        bar = self._bar_at(ticker, minute)
        if bar is not None:
            return bar.open
        done = self._done(ticker, minute)
        return done[-1].close if done else None

    # -- settlement --------------------------------------------------------

    def _fill(
        self, oid: str, price: Decimal, ts: dt.datetime | None = None, *, resting: bool = False
    ) -> None:
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
                "limit": o["limit"],
                "resting": resting,
            }
        )

    def _sync(self) -> None:
        minute = self._minute()
        for oid, o in list(self._open.items()):
            t = o["ticker"]
            ts = self._ts.get(t, [])
            lo, hi = bisect.bisect_right(ts, o["checked"]), bisect.bisect_left(ts, minute)
            for b in self._bars.get(t, [])[lo:hi]:
                o["checked"] = b.ts
                up, down = self._bar_locked(t, b)
                bar_end = (b.ts + dt.timedelta(minutes=1)).replace(tzinfo=KST)
                if o["side"] == OrderSide.BUY and not up and b.low <= o["limit"]:
                    price = min(o["limit"], b.open)
                    if price * o["qty"] > self._cash:
                        self._open.pop(oid)
                        self.rejections.append(
                            {
                                "ts": bar_end,
                                "ticker": t,
                                "order_id": oid,
                                "qty": o["qty"],
                                "price": price,
                                "reason": f"insufficient cash {self._cash} for fill",
                            }
                        )
                    else:
                        self._fill(oid, price, ts=bar_end, resting=True)
                    break
                if o["side"] == OrderSide.SELL and not down and b.high >= o["limit"]:
                    self._fill(oid, max(o["limit"], b.open), ts=bar_end, resting=True)
                    break

    def has_bars(self, ticker: str) -> bool:
        return bool(self._bars.get(ticker))

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
        value = sum(
            (Decimal(q) * (self._mark(t) or self._avg[t]) for t, q in self._held.items() if q > 0),
            Decimal(0),
        )
        return Account(
            account_id="sim", cash=self._cash, buying_power=self._cash, equity=self._cash + value
        )

    def get_positions(self) -> list[Position]:
        self._sync()
        return [
            Position(
                ticker=t,
                quantity=Decimal(q),
                average_price=self._avg[t],
                current_price=self._mark(t) or self._avg[t],
            )
            for t, q in self._held.items()
            if q > 0
        ]

    def get_quote(self, ticker: str) -> Quote:
        self._sync()
        minute = self._minute()
        bar = self._bar_at(ticker, minute)
        if bar is not None:
            price = bar.open
            up, down = self._open_locked(ticker, bar)
        else:
            done = self._done(ticker, minute)
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
        if (
            order.side == OrderSide.BUY
            and filled_price is not None
            and filled_price * qty > self._cash
        ):
            self._open.pop(oid)
            raise BrokerError(
                f"insufficient cash {self._cash} for {qty} x {filled_price} {order.ticker}",
                broker_name="sim",
            )
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
        return list(self._done(ticker, self._minute()))
