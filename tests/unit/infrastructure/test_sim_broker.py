"""SimBroker: minute-bar fills with no look-ahead."""

import datetime as dt
from decimal import Decimal

import pytest

from daytrade_it.infrastructure.brokers.base import KST, MinuteBar, Order, OrderSide, OrderType
from daytrade_it.infrastructure.brokers.sim_broker import SimBroker
from daytrade_it.shared.errors.exceptions import BrokerError


def _bar(hm: str, o: int, h: int, l: int, c: int, v: int = 100) -> MinuteBar:  # noqa: E741
    hh, mm = map(int, hm.split(":"))
    return MinuteBar(
        ts=dt.datetime(2026, 9, 15, hh, mm),
        open=Decimal(o),
        high=Decimal(h),
        low=Decimal(l),
        close=Decimal(c),
        volume=v,
    )


class _Clock:
    def __init__(self) -> None:
        self.now = dt.datetime(2026, 9, 15, 9, 1, 10, tzinfo=KST)

    def __call__(self) -> dt.datetime:
        return self.now


BARS = {
    "000001": [
        _bar("09:00", 1000, 1010, 990, 1000),
        _bar("09:01", 1005, 1020, 1000, 1010),
        _bar("09:02", 1010, 1015, 980, 990),
        _bar("09:03", 995, 1000, 970, 975),
    ]
}


def _broker(clock: _Clock) -> SimBroker:
    return SimBroker(BARS, {"000001": Decimal(1000)}, clock, cash=Decimal(1_000_000))


def _buy(limit: int) -> Order:
    return Order(
        ticker="000001",
        quantity=Decimal(1),
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=Decimal(limit),
    )


def _sell(limit: int) -> Order:
    return Order(
        ticker="000001",
        quantity=Decimal(1),
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        limit_price=Decimal(limit),
    )


def test_quote_uses_current_minute_open_one_tick_spread() -> None:
    q = _broker(_Clock()).get_quote("000001")
    assert (q.ask, q.bid, q.last_price) == (Decimal(1005), Decimal(1004), Decimal(1005))


def test_marketable_buy_fills_at_ask_and_updates_positions_and_cash() -> None:
    c = _Clock()
    b = _broker(c)
    r = b.place_order(_buy(1005))
    assert r.filled_quantity == 1 and r.filled_avg_price == Decimal(1005)
    assert [(p.ticker, p.quantity) for p in b.get_positions()] == [("000001", Decimal(1))]
    assert b.get_account().cash == Decimal(1_000_000 - 1005)
    with pytest.raises(BrokerError):
        b.get_order_status(r.order_id)


def test_resting_buy_fills_only_on_a_later_completed_bar() -> None:
    c = _Clock()
    b = _broker(c)
    r = b.place_order(_buy(985))
    assert b.get_order_status(r.order_id).remaining_quantity == 1
    c.now = dt.datetime(2026, 9, 15, 9, 2, 30, tzinfo=KST)  # 09:02 bar not finished
    assert b.get_order_status(r.order_id).remaining_quantity == 1
    c.now = dt.datetime(2026, 9, 15, 9, 3, 0, tzinfo=KST)  # 09:02 bar (low 980) done
    with pytest.raises(BrokerError):
        b.get_order_status(r.order_id)
    assert b.fills[-1]["price"] == Decimal(985)


def test_cancel_open_order_true_then_false() -> None:
    b = _broker(_Clock())
    r = b.place_order(_buy(900))
    assert b.cancel_order(r.order_id) is True
    assert b.cancel_order(r.order_id) is False


def test_marketable_sell_fills_at_bid() -> None:
    c = _Clock()
    b = _broker(c)
    b.place_order(_buy(1005))
    r = b.place_order(_sell(990))
    assert r.filled_avg_price == Decimal(1004)
    assert b.get_positions() == []


def test_minute_bars_hide_current_and_future_minutes() -> None:
    c = _Clock()
    assert [x.ts.minute for x in _broker(c).get_minute_bars("000001")] == [0]


def test_limit_up_lock_has_no_ask_and_never_fills_buys() -> None:
    bars = {
        "000002": [_bar("09:00", 1300, 1300, 1300, 1300), _bar("09:01", 1300, 1300, 1300, 1300)]
    }
    c = _Clock()
    b = SimBroker(bars, {"000002": Decimal(1000)}, c)
    assert b.get_quote("000002").ask == 0
    r = b.place_order(
        Order(
            ticker="000002",
            quantity=Decimal(1),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal(1300),
        )
    )
    c.now = dt.datetime(2026, 9, 15, 9, 5, tzinfo=KST)
    assert b.get_order_status(r.order_id).remaining_quantity == 1
