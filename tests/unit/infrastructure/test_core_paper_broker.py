"""CorePaperBroker: real read-only quotes, krx-quant-core PaperBroker fills.

The data broker is a fake `BrokerInterface`-shaped object -- no Kiwoom call.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from daytrade_it.infrastructure.brokers.base import (
    KST,
    Order,
    OrderSide,
    OrderStatusEnum,
    OrderType,
    Quote,
)
from daytrade_it.infrastructure.brokers.core_paper_broker import CorePaperBroker
from daytrade_it.shared.errors import BrokerError, OrderRejectedError


class _FakeData:
    """Read-only data broker: quotes set by the test, orders must never be called."""

    def __init__(self) -> None:
        self.quotes: dict[str, Quote] = {}
        self.connected = True

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        self.connected = False

    @property
    def is_connected(self) -> bool:
        return self.connected

    def get_quote(self, ticker: str) -> Quote:
        return self.quotes[ticker]

    def get_minute_bars(self, ticker: str, day: dt.date | None = None) -> list:
        return ["bar"]

    def place_order(self, order: Order) -> None:  # pragma: no cover - must not happen
        raise AssertionError("paper mode must never place a real order")

    def cancel_order(self, order_id: str) -> None:  # pragma: no cover
        raise AssertionError("paper mode must never cancel a real order")


class _Clock:
    def __init__(self) -> None:
        self.now = dt.datetime(2026, 9, 17, 9, 1, tzinfo=KST)

    def __call__(self) -> dt.datetime:
        self.now += dt.timedelta(seconds=1)
        return self.now


def _quote(bid: int, ask: int, last: int) -> Quote:
    return Quote(ticker="005930", bid=Decimal(bid), ask=Decimal(ask), last_price=Decimal(last))


def _limit(side: OrderSide, price: int, qty: int = 1) -> Order:
    return Order(
        ticker="005930",
        quantity=Decimal(qty),
        side=side,
        order_type=OrderType.LIMIT,
        limit_price=Decimal(price),
    )


@pytest.fixture
def data() -> _FakeData:
    return _FakeData()


@pytest.fixture
def broker(data: _FakeData) -> CorePaperBroker:
    return CorePaperBroker(data, cash=Decimal(10_000_000), clock=_Clock())  # type: ignore[arg-type]


def test_buy_rests_then_fills_on_a_later_quote(data: _FakeData, broker: CorePaperBroker) -> None:
    data.quotes["005930"] = _quote(70_000, 70_100, 70_000)
    broker.get_quote("005930")  # the pre-order quote (AutoTrader reads one first)

    result = broker.place_order(_limit(OrderSide.BUY, 70_000))  # below the ask: rests
    assert result.status == OrderStatusEnum.SUBMITTED
    assert result.order_id
    status = broker.get_order_status(result.order_id)
    assert status.status == OrderStatusEnum.SUBMITTED
    assert status.remaining_quantity == 1
    assert broker.get_positions() == []

    data.quotes["005930"] = _quote(69_900, 70_000, 70_000)
    broker.get_quote("005930")

    status = broker.get_order_status(result.order_id)
    assert status.status == OrderStatusEnum.FILLED
    assert status.filled_quantity == 1
    assert status.remaining_quantity == 0
    assert status.filled_avg_price == Decimal(70_000)
    [pos] = broker.get_positions()
    assert (pos.ticker, pos.quantity, pos.average_price) == ("005930", 1, Decimal(70_000))
    assert broker.get_account().cash == Decimal(10_000_000 - 70_000)


def test_buy_below_the_ask_keeps_resting(data: _FakeData, broker: CorePaperBroker) -> None:
    data.quotes["005930"] = _quote(70_000, 70_100, 70_000)
    broker.get_quote("005930")
    result = broker.place_order(_limit(OrderSide.BUY, 69_000))
    broker.get_quote("005930")
    assert broker.get_order_status(result.order_id).status == OrderStatusEnum.SUBMITTED


def test_cancel_removes_resting_order(data: _FakeData, broker: CorePaperBroker) -> None:
    data.quotes["005930"] = _quote(70_000, 70_100, 70_000)
    broker.get_quote("005930")
    result = broker.place_order(_limit(OrderSide.BUY, 69_000))

    assert broker.cancel_order(result.order_id) is True
    status = broker.get_order_status(result.order_id)
    assert status.status == OrderStatusEnum.CANCELLED
    assert status.remaining_quantity == 0
    # Cancelled: a later crossing quote must not fill it.
    data.quotes["005930"] = _quote(68_000, 68_500, 68_500)
    broker.get_quote("005930")
    assert broker.get_positions() == []
    assert broker.cancel_order(result.order_id) is False


def test_sell_more_than_held_is_rejected(data: _FakeData, broker: CorePaperBroker) -> None:
    data.quotes["005930"] = _quote(70_000, 70_100, 70_000)
    broker.get_quote("005930")
    with pytest.raises(OrderRejectedError):
        broker.place_order(_limit(OrderSide.SELL, 70_000))


def test_sell_after_fill_realizes_cash(data: _FakeData, broker: CorePaperBroker) -> None:
    data.quotes["005930"] = _quote(70_000, 70_100, 70_000)
    broker.get_quote("005930")
    buy = broker.place_order(_limit(OrderSide.BUY, 70_100))
    broker.get_quote("005930")
    assert broker.get_order_status(buy.order_id).status == OrderStatusEnum.FILLED

    sell = broker.place_order(_limit(OrderSide.SELL, 70_000))
    data.quotes["005930"] = _quote(70_200, 70_300, 70_200)
    broker.get_quote("005930")
    status = broker.get_order_status(sell.order_id)
    assert status.status == OrderStatusEnum.FILLED
    assert status.filled_avg_price == Decimal(70_200)  # taker at the bid
    assert broker.get_positions() == []
    assert broker.get_account().cash == Decimal(10_000_000 - 70_100 + 70_200)


def test_market_orders_are_rejected(broker: CorePaperBroker) -> None:
    with pytest.raises(OrderRejectedError):
        broker.place_order(Order(ticker="005930", quantity=Decimal(1), side=OrderSide.BUY))


def test_unknown_order_status_raises_like_kiwoom(broker: CorePaperBroker) -> None:
    with pytest.raises(BrokerError):
        broker.get_order_status("P999999")


def test_minute_bars_and_connection_come_from_the_data_broker(
    data: _FakeData, broker: CorePaperBroker
) -> None:
    assert broker.get_minute_bars("005930") == ["bar"]
    assert broker.is_connected
    assert broker.connect() is True


def test_status_and_positions_requote_resting_orders(
    data: _FakeData, broker: CorePaperBroker
) -> None:
    """AutoTrader checks a pending entry via get_order_status/get_positions
    without quoting it again -- those calls must look at a fresh quote, or a
    paper entry could only ever be cancelled at its TTL."""
    data.quotes["005930"] = _quote(70_000, 70_100, 70_000)
    broker.get_quote("005930")
    result = broker.place_order(_limit(OrderSide.BUY, 70_100))

    data.quotes["005930"] = _quote(70_000, 70_100, 70_100)
    assert broker.get_order_status(result.order_id).status == OrderStatusEnum.FILLED

    sell = broker.place_order(_limit(OrderSide.SELL, 70_000))
    data.quotes["005930"] = _quote(70_000, 70_100, 70_000)
    assert broker.get_positions() == []
    assert broker.get_order_status(sell.order_id).status == OrderStatusEnum.FILLED


def test_limit_is_rounded_to_a_valid_tick_like_kiwoom_broker(
    data: _FakeData, broker: CorePaperBroker
) -> None:
    data.quotes["005930"] = _quote(70_000, 70_100, 70_000)
    broker.get_quote("005930")
    result = broker.place_order(_limit(OrderSide.BUY, 69_950))  # not a 100-won tick
    # KiwoomBroker sends round_to_tick(limit, KOSPI): rounded down.
    assert [o.price for o in broker._paper.open_orders()] == [69_900]
    assert result.limit_price == Decimal(69_950)  # the order as AutoTrader sent it
