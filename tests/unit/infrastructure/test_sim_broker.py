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


@pytest.mark.parametrize("second", [0, 10, 29, 30, 45, 59])
def test_quote_never_uses_current_minute_close_high_or_low(second: int) -> None:
    """No look-ahead: at any second of 09:01 the quote is that bar's open.

    The 09:01 bar's close (1010), high (1020) and low (1000) are only known
    when the minute ends; a quote or marketable fill using them would peek.
    """
    c = _Clock()
    c.now = dt.datetime(2026, 9, 15, 9, 1, second, tzinfo=KST)
    b = _broker(c)
    q = b.get_quote("000001")
    assert (q.ask, q.bid, q.last_price) == (Decimal(1005), Decimal(1004), Decimal(1005))
    r = b.place_order(_buy(1020))
    assert r.filled_avg_price == Decimal(1005)


def test_marketable_buy_fills_at_ask_and_updates_positions_and_cash() -> None:
    c = _Clock()
    b = _broker(c)
    r = b.place_order(_buy(1005))
    assert r.filled_quantity == 1 and r.filled_avg_price == Decimal(1005)
    assert [(p.ticker, p.quantity) for p in b.get_positions()] == [("000001", Decimal(1))]
    assert b.get_account().buying_power == Decimal(1_000_000 - 1005)
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


def test_resting_fill_ts_is_the_fill_bars_end_not_discovery_time() -> None:
    c = _Clock()
    b = _broker(c)
    r = b.place_order(_buy(985))
    c.now = dt.datetime(2026, 9, 15, 9, 5, 0, tzinfo=KST)  # discovered well after fill bar
    with pytest.raises(BrokerError):
        b.get_order_status(r.order_id)  # settled during _sync -> triggers the fill
    assert b.fills[-1]["ts"] == dt.datetime(2026, 9, 15, 9, 3, 0, tzinfo=KST)


def test_second_sell_order_blocked_by_shares_reserved_in_first_open_sell() -> None:
    c = _Clock()
    b = _broker(c)
    b.place_order(_buy(1005))  # marketable buy -> 1 share held
    b.place_order(_sell(2000))  # resting sell reserves the 1 share
    with pytest.raises(BrokerError):
        b.place_order(_sell(990))


def test_sell_succeeds_after_cancelling_the_reserving_order() -> None:
    c = _Clock()
    b = _broker(c)
    b.place_order(_buy(1005))
    r1 = b.place_order(_sell(2000))
    b.cancel_order(r1.order_id)
    r2 = b.place_order(_sell(990))
    assert r2.filled_avg_price == Decimal(1004)


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


# -- risk realism (final review I3) ----------------------------------------


def test_positions_and_equity_are_marked_at_the_current_minute_open() -> None:
    c = _Clock()
    b = _broker(c)
    b.place_order(_buy(1005))
    c.now = dt.datetime(2026, 9, 15, 9, 3, 20, tzinfo=KST)  # 09:02 closed 990, 09:03 opens 995
    [p] = b.get_positions()
    assert (p.current_price, p.market_value) == (Decimal(995), Decimal(995))
    acct = b.get_account()
    assert acct.buying_power == Decimal(1_000_000 - 1005)
    assert acct.equity == acct.buying_power + Decimal(995)


def test_marketable_buy_beyond_cash_is_rejected_at_placement() -> None:
    b = SimBroker(BARS, {"000001": Decimal(1000)}, _Clock(), cash=Decimal(1000))
    with pytest.raises(BrokerError):
        b.place_order(_buy(1005))
    assert b.fills == [] and b.get_positions() == []
    assert b.get_account().cash == Decimal(1000)


def test_resting_buy_is_cancelled_with_a_record_when_cash_is_short_at_fill() -> None:
    c = _Clock()
    b = SimBroker(BARS, {"000001": Decimal(1000)}, c, cash=Decimal(1500))
    r = b.place_order(_buy(985))  # rests; 985 <= cash now
    b.place_order(_buy(1005))  # marketable, cash 495 left
    c.now = dt.datetime(2026, 9, 15, 9, 3, 0, tzinfo=KST)  # 09:02 bar trades through 985
    with pytest.raises(BrokerError):
        b.get_order_status(r.order_id)
    assert all(f["order_id"] != r.order_id for f in b.fills)
    [rej] = b.rejections
    assert rej["order_id"] == r.order_id and "cash" in rej["reason"]
    assert b.get_account().buying_power == Decimal(495)


def test_fills_record_limit_and_whether_the_order_rested() -> None:
    c = _Clock()
    b = _broker(c)
    b.place_order(_buy(1005))
    b.place_order(_buy(985))
    c.now = dt.datetime(2026, 9, 15, 9, 3, 0, tzinfo=KST)
    b.get_positions()
    assert [(f["limit"], f["resting"]) for f in b.fills] == [
        (Decimal(1005), False),
        (Decimal(985), True),
    ]


# -- price-limit lock without look-ahead (final review minor) ----------------


def test_lock_ignores_current_minute_high_low_and_uses_limit_price_from_prev_close() -> None:
    """09:01 opens at the upper limit (1300 over 1000) but trades to 1350 later in the
    minute: the high is unknown at 09:01:10, so the quote must not depend on it."""
    bars = {
        "000002": [_bar("09:00", 1200, 1200, 1200, 1200), _bar("09:01", 1300, 1350, 1300, 1320)]
    }
    b = SimBroker(bars, {"000002": Decimal(1000)}, _Clock())
    assert b.get_quote("000002").ask == 0


def test_open_below_the_limit_is_not_locked_even_if_bar_is_flat() -> None:
    bars = {"000002": [_bar("09:01", 1200, 1200, 1200, 1200)]}
    b = SimBroker(bars, {"000002": Decimal(1000)}, _Clock())
    assert b.get_quote("000002").ask == Decimal(1200)


def test_limit_lock_uses_limit_base_not_prev_close() -> None:
    # After-market-inclusive close 2900 but KRX regular close 2705: upper limit is ~3515 (2705 * 1.3), not 3770.
    bars = {
        "000009": [_bar("09:00", 3515, 3515, 3515, 3515), _bar("09:01", 3515, 3515, 3515, 3515)]
    }
    c = _Clock()
    b = SimBroker(bars, {"000009": Decimal(2900)}, c, limit_base={"000009": Decimal(2705)})
    assert b.get_quote("000009").ask == 0
    b2 = SimBroker(bars, {"000009": Decimal(2900)}, c)
    assert b2.get_quote("000009").ask == 3515


# -- deposit cash is not reduced by today's buys (replay-parity 2026-10-05 #13) --


def test_account_cash_is_the_deposit_not_reduced_by_todays_fills() -> None:
    """Kiwoom's 예수금 settles T+2: a buy today leaves `cash` where it was.

    AutoTrader subtracts its own `_committed` limit value from `cash` before
    each entry/add-on. A broker that also took the fill out of `cash` made the
    trader pay twice: on 09-16 cash 544,370 minus committed 555,630 refused two
    add-ons live had placed. `buying_power` still tells what can really fill.
    """
    c = _Clock()
    b = _broker(c)
    b.place_order(_buy(1005))
    acct = b.get_account()
    assert acct.cash == Decimal(1_000_000)
    assert acct.buying_power == Decimal(1_000_000 - 1005)
    b.place_order(_sell(990))  # marketable sell at the bid 1004
    acct = b.get_account()
    assert acct.cash == Decimal(1_000_000)
    assert acct.buying_power == Decimal(1_000_000 - 1005 + 1004)
    assert acct.equity == acct.buying_power


def test_fill_affordability_still_uses_the_real_balance() -> None:
    """The deposit is what the trader sees; the fills still can't exceed real cash."""
    c = _Clock()
    b = SimBroker(BARS, {"000001": Decimal(1000)}, c, cash=Decimal(1500))
    b.place_order(_buy(1005))
    assert b.get_account().cash == Decimal(1500)
    with pytest.raises(BrokerError):
        b.place_order(_buy(1005))
