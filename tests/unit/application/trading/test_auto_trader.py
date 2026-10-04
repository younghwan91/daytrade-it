"""AutoTrader: 1-share entries through execute_trade, flat by the close."""

import datetime as dt
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest

from daytrade_it.application.trading import auto_trader
from daytrade_it.application.trading.auto_trader import (
    AutoTrader,
    TradingLimits,
    limits_from_strings,
    limits_to_strings,
    rank_batch,
)
from daytrade_it.domain.entities.signal import SignalDirection, SignalSource, TradingSignal
from daytrade_it.domain.value_objects import Confidence, Ticker
from daytrade_it.infrastructure.brokers.base import (
    KST,
    Account,
    MinuteBar,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderStatusEnum,
    OrderType,
    Position,
    Quote,
)
from daytrade_it.shared.errors import BrokerError, OrderRejectedError


class Clock:
    def __init__(self, hhmm: str) -> None:
        h, m = map(int, hhmm.split(":"))
        self.now = dt.datetime(2026, 9, 14, h, m, tzinfo=KST)

    def __call__(self) -> dt.datetime:
        return self.now

    def advance(self, **kw: int) -> None:
        self.now += dt.timedelta(**kw)


@dataclass
class FakeBroker:
    bid: str = "70000"
    ask: str = "70100"

    def __post_init__(self) -> None:
        self.positions: dict[str, int] = {}
        self.orders: list = []
        self.unfilled: dict[str, OrderStatus] = {}
        self.filled: dict[str, OrderStatus] = {}
        self.cancelled: list[str] = []
        self.place_error: Exception | None = None
        self._seq = 0
        self.bars: dict[str, list[MinuteBar]] | None = None  # None: no chart feed
        self.bar_calls = 0

    def get_minute_bars(self, ticker: str, day: dt.date | None = None) -> list[MinuteBar]:
        self.bar_calls += 1
        if self.bars is None:
            raise NotImplementedError
        return self.bars.get(ticker, [])

    def get_account(self) -> Account:
        return Account(
            account_id="a",
            cash=Decimal("10000000"),
            buying_power=Decimal("10000000"),
            equity=Decimal("10000000"),
        )

    def get_positions(self) -> list[Position]:
        return [
            Position(ticker=t, quantity=Decimal(q), average_price=Decimal("70000"))
            for t, q in self.positions.items()
            if q > 0
        ]

    def get_quote(self, ticker: str) -> Quote:
        return Quote(ticker=ticker, bid=Decimal(self.bid), ask=Decimal(self.ask))

    def place_order(self, order) -> OrderResult:  # noqa: ANN001
        if self.place_error:
            raise self.place_error
        self._seq += 1
        order_id = f"{self._seq:07d}"
        self.orders.append(order)
        self.unfilled[order_id] = OrderStatus(
            order_id=order_id,
            status=OrderStatusEnum.SUBMITTED,
            filled_quantity=Decimal(0),
            remaining_quantity=order.quantity,
        )
        return OrderResult(
            order_id=order_id,
            client_order_id=None,
            ticker=order.ticker,
            quantity=order.quantity,
            filled_quantity=Decimal(0),
            side=order.side,
            order_type=order.order_type,
            status=OrderStatusEnum.SUBMITTED,
        )

    def fill(self, order_id: str, ticker: str, price: str | None = None) -> None:
        """Fill one share. With `price`, the status stays readable like the paper
        broker's (FILLED + filled_avg_price); without it the order vanishes like
        Kiwoom's unfilled list."""
        status = self.unfilled.pop(order_id)
        delta = 1
        for o in reversed(self.orders):  # the latest order on the ticker decides the side
            if o.ticker == ticker:
                delta = -int(status.remaining_quantity) if o.side == OrderSide.SELL else 1
                break
        self.positions[ticker] = self.positions.get(ticker, 0) + delta
        if price is not None:
            self.filled[order_id] = OrderStatus(
                order_id=order_id,
                status=OrderStatusEnum.FILLED,
                filled_quantity=status.remaining_quantity,
                remaining_quantity=Decimal(0),
                filled_avg_price=Decimal(price),
            )

    def get_order_status(self, order_id: str) -> OrderStatus:
        if order_id in self.filled:
            return self.filled[order_id]
        if order_id not in self.unfilled:
            raise BrokerError(f"Order not found: {order_id}", broker_name="fake")
        return self.unfilled[order_id]

    def cancel_order(self, order_id: str) -> bool:
        self.cancelled.append(order_id)
        return self.unfilled.pop(order_id, None) is not None


def _signal(
    ticker: str = "005930", direction: SignalDirection = SignalDirection.BUY
) -> TradingSignal:
    return TradingSignal(
        ticker=Ticker(value=ticker),
        direction=direction,
        confidence=Confidence(value=0.8),
        source=SignalSource.COMPOSITE,
        expires_at=dt.datetime.now(dt.UTC) + dt.timedelta(hours=2),
    )


@pytest.fixture(autouse=True)
def _no_kill_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from daytrade_it.interfaces.mcp import tools

    monkeypatch.setattr(tools, "_KILL_SWITCH_PATH", tmp_path / "no_such_live_STOP")
    from daytrade_it.application.trading import auto_trader

    monkeypatch.setattr(auto_trader, "_disclosure_blocks", lambda _ticker: False)


def _trader(
    tmp_path: Path,
    broker: FakeBroker,
    clock: Clock,
    *,
    live: bool = True,
    limits: TradingLimits | None = None,
) -> AutoTrader:
    if limits is None:
        return AutoTrader(broker, tmp_path / "ledger", live=live, clock=clock)
    return AutoTrader(broker, tmp_path / "ledger", live=live, clock=clock, limits=limits)


def _rows(tmp_path: Path) -> list[dict]:
    path = tmp_path / "ledger" / "2026-09-14.jsonl"
    return [json.loads(line) for line in path.open()] if path.exists() else []


def _events(tmp_path: Path) -> list[str]:
    return [r["event"] for r in _rows(tmp_path)]


async def test_live_entry_is_one_share_limit_at_ask(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock("10:00")
    await _trader(tmp_path, broker, clock).enter(_signal())
    assert len(broker.orders) == 1
    order = broker.orders[0]
    assert (order.side, order.order_type, order.quantity) == (OrderSide.BUY, OrderType.LIMIT, 1)
    assert order.limit_price == Decimal("70100")
    assert _events(tmp_path) == ["baseline", "entry_submitted"]


async def test_dry_run_places_nothing(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock("10:00")
    await _trader(tmp_path, broker, clock, live=False).enter(_signal())
    assert broker.orders == []
    assert "entry_dry_run" in _events(tmp_path)


@pytest.mark.parametrize("hhmm", ["08:59", "14:30", "15:00"])
async def test_no_entries_outside_window(tmp_path: Path, hhmm: str) -> None:
    broker = FakeBroker()
    await _trader(tmp_path, broker, Clock(hhmm)).enter(_signal())
    assert broker.orders == []


async def test_sell_signals_never_enter(tmp_path: Path) -> None:
    broker = FakeBroker()
    await _trader(tmp_path, broker, Clock("10:00")).enter(_signal(direction=SignalDirection.SELL))
    assert broker.orders == []


async def test_one_entry_per_ticker_per_day(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock("10:00")
    trader = _trader(tmp_path, broker, clock)
    await trader.enter(_signal())
    await trader.enter(_signal())
    assert len(broker.orders) == 1


async def test_never_buys_a_ticker_the_user_already_holds(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.positions["005930"] = 10
    await _trader(tmp_path, broker, Clock("10:00")).enter(_signal())
    assert broker.orders == []


async def test_no_ask_or_over_notional_cap_is_skipped(tmp_path: Path) -> None:
    limit_up = FakeBroker(bid="91000", ask="0")
    await _trader(tmp_path, limit_up, Clock("10:00")).enter(_signal())
    assert limit_up.orders == []

    pricey = FakeBroker(bid="500000", ask="500500")
    await _trader(tmp_path / "b", pricey, Clock("10:00")).enter(_signal())
    assert pricey.orders == []


async def test_concurrent_cap_counts_pending_and_held(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock("10:00")
    trader = AutoTrader(
        broker,
        tmp_path / "ledger",
        live=True,
        clock=clock,
        limits=TradingLimits(max_open_positions=2),
    )
    for t in ("005930", "000660", "035420"):
        await trader.enter(_signal(t))
    assert [o.ticker for o in broker.orders] == ["005930", "000660"]


async def test_refusal_and_broker_error_do_not_raise(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock("10:00")
    broker.place_error = OrderRejectedError(message="잔고부족", broker_name="fake", symbol="005930")
    trader = _trader(tmp_path, broker, clock)
    await trader.enter(_signal())
    assert _events(tmp_path)[-1] == "entry_unknown"
    broker.place_error = None
    await trader.enter(_signal())  # same ticker: never retried after an unknown outcome
    assert broker.orders == []


async def test_stale_entry_order_is_cancelled_after_ttl(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock("10:00")
    trader = _trader(tmp_path, broker, clock)
    await trader.enter(_signal())
    trader.manage()
    assert broker.cancelled == []
    clock.advance(minutes=5)
    trader.manage()
    assert broker.cancelled == ["0000001"]


async def test_flatten_sells_only_the_traders_share(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock("10:00")
    broker.positions["000660"] = 7  # user's own, untouched
    trader = _trader(tmp_path, broker, clock)
    await trader.enter(_signal("005930"))
    broker.fill("0000001", "005930")

    clock.now = clock.now.replace(hour=15, minute=10)
    trader.manage()

    sells = [o for o in broker.orders if o.side == OrderSide.SELL]
    assert [(o.ticker, o.quantity, o.order_type, o.limit_price) for o in sells] == [
        ("005930", 1, OrderType.LIMIT, Decimal("70000"))
    ]


async def test_flatten_never_sells_user_shares_of_an_unfilled_entry(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock("10:00")
    trader = _trader(tmp_path, broker, clock)
    await trader.enter(_signal("005930"))
    clock.advance(minutes=5)
    trader.manage()  # cancelled with zero fill
    broker.positions["005930"] = 3  # user buys by hand later

    clock.now = clock.now.replace(hour=15, minute=10)
    trader.manage()
    assert [o for o in broker.orders if o.side == OrderSide.SELL] == []


async def test_exit_retries_after_delay_and_not_every_tick(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock("10:00")
    trader = _trader(tmp_path, broker, clock)
    await trader.enter(_signal())
    broker.fill("0000001", "005930")
    clock.now = clock.now.replace(hour=15, minute=10)
    trader.manage()
    clock.advance(seconds=30)
    trader.manage()
    assert sum(o.side == OrderSide.SELL for o in broker.orders) == 1
    clock.advance(minutes=3)
    trader.manage()  # still held: cancel the stale exit, resend at the bid
    assert sum(o.side == OrderSide.SELL for o in broker.orders) == 2
    assert "0000002" in broker.cancelled


async def test_restart_resumes_from_ledger(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock("10:00")
    await _trader(tmp_path, broker, clock).enter(_signal("005930"))
    broker.fill("0000001", "005930")

    restarted = _trader(tmp_path, broker, clock)
    await restarted.enter(_signal("005930"))
    assert len(broker.orders) == 1  # no re-entry after restart

    clock.now = clock.now.replace(hour=15, minute=10)
    restarted.manage()
    sells = [o for o in broker.orders if o.side == OrderSide.SELL]
    assert len(sells) == 1  # baseline kept from ledger, not re-snapshotted with the bot's share


async def test_exit_is_not_blocked_by_kill_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daytrade_it.interfaces.mcp import tools

    broker, clock = FakeBroker(), Clock("10:00")
    trader = _trader(tmp_path, broker, clock)
    await trader.enter(_signal())
    broker.fill("0000001", "005930")

    kill = tmp_path / "live_STOP"
    kill.write_text("stop")
    monkeypatch.setattr(tools, "_KILL_SWITCH_PATH", kill)
    clock.now = clock.now.replace(hour=15, minute=10)
    trader.manage()
    assert any(o.side == OrderSide.SELL for o in broker.orders)


async def test_entry_needs_deposit_cash_not_margin(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock("10:00")
    low_cash = Account(
        account_id="a",
        cash=Decimal("50000"),  # deposit: less than one 70,100 share
        buying_power=Decimal("13000000"),  # margin-inflated orderable amount
        equity=Decimal("50000"),
    )
    broker.get_account = lambda: low_cash  # type: ignore[method-assign]
    await _trader(tmp_path, broker, clock).enter(_signal())
    assert broker.orders == []


async def test_cash_check_counts_todays_open_entries(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock("10:00")
    cash = Account(
        account_id="a",
        cash=Decimal("120000"),
        buying_power=Decimal("120000"),
        equity=Decimal("120000"),
    )
    broker.get_account = lambda: cash  # type: ignore[method-assign]
    trader = _trader(tmp_path, broker, clock)
    await trader.enter(_signal("005930"))  # 70,100 committed
    await trader.enter(_signal("000660"))  # 120,000 - 70,100 < 70,801 -> refused
    assert [o.ticker for o in broker.orders] == ["005930"]


async def test_flatten_starts_at_1500_and_retries_every_minute_with_deeper_limits(
    tmp_path: Path,
) -> None:
    broker, clock = FakeBroker(), Clock("10:00")
    trader = _trader(tmp_path, broker, clock)
    await trader.enter(_signal())
    broker.fill("0000001", "005930")

    clock.now = clock.now.replace(hour=14, minute=59)
    trader.manage()
    assert [o for o in broker.orders if o.side == OrderSide.SELL] == []

    clock.now = clock.now.replace(hour=15, minute=0)
    for _ in range(6):  # still held every time: each minute resends lower
        trader.manage()
        clock.advance(minutes=1)
    limits = [o.limit_price for o in broker.orders if o.side == OrderSide.SELL]
    assert limits == [
        Decimal("70000"),
        Decimal("70000") * Decimal("0.995"),
        Decimal("70000") * Decimal("0.990"),
        Decimal("70000") * Decimal("0.985"),
        Decimal("70000") * Decimal("0.980"),
        Decimal("70000") * Decimal("0.980"),  # capped at 2% below bid
    ]


async def test_leftover_position_at_shutdown_is_recorded(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock("10:00")
    trader = _trader(tmp_path, broker, clock)
    await trader.enter(_signal())
    broker.fill("0000001", "005930")
    clock.now = clock.now.replace(hour=15, minute=30)
    assert trader.report_leftovers() == {"005930": 1}
    assert _events(tmp_path)[-1] == "overnight_positions"


async def test_no_leftover_report_when_flat(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock("10:00")
    trader = _trader(tmp_path, broker, clock)
    await trader.enter(_signal())
    clock.advance(minutes=5)
    trader.manage()  # entry cancelled unfilled
    clock.now = clock.now.replace(hour=15, minute=30)
    assert trader.report_leftovers() == {}
    assert "overnight_positions" not in _events(tmp_path)


def _bars(*spec: tuple[str, int, int]) -> list[MinuteBar]:
    """(HH:MM, close, volume) -> bars on the Clock's day, tz-naive like Kiwoom."""
    out = []
    for hhmm, close, vol in spec:
        h, m = map(int, hhmm.split(":"))
        c = Decimal(close)
        out.append(MinuteBar(dt.datetime(2026, 9, 14, h, m), c, c, c, c, vol))
    return out


async def _held_from_0901(
    tmp_path: Path, broker: FakeBroker, limits: TradingLimits | None = None
) -> tuple[AutoTrader, Clock]:
    clock = Clock("09:01")
    trader = _trader(tmp_path, broker, clock, limits=limits)
    await trader.enter(_signal("005930"))  # limit at ask 70100
    broker.fill("0000001", "005930")
    return trader, clock


# 진입 70100. 73000에서 거래가 몰려 VWAP≈73000, 이후 71600: 진입 대비 +2.1%, VWAP 대비 -1.9%.
_UP_THEN_BREAK = ("09:01", 73000, 10000), ("09:05", 73000, 10000), ("09:11", 71600, 100)


async def test_vwap_break_after_gain_sells_before_the_close(tmp_path: Path) -> None:
    broker = FakeBroker()
    trader, clock = await _held_from_0901(tmp_path, broker)
    broker.bars = {"005930": _bars(*_UP_THEN_BREAK)}
    clock.now = clock.now.replace(minute=12)
    trader.manage()
    sells = [o for o in broker.orders if o.side == OrderSide.SELL]
    assert [(o.ticker, o.quantity, o.limit_price) for o in sells] == [
        ("005930", 1, Decimal("70000"))
    ]
    assert "exit_signal" in _events(tmp_path)


async def test_no_vwap_exit_when_gain_below_minimum(tmp_path: Path) -> None:
    broker = FakeBroker()
    trader, clock = await _held_from_0901(tmp_path, broker)
    # +1.3% only: below VWAP but the position never reached +2%.
    broker.bars = {"005930": _bars(("09:01", 73000, 10000), ("09:11", 71000, 100))}
    clock.now = clock.now.replace(minute=12)
    trader.manage()
    assert [o for o in broker.orders if o.side == OrderSide.SELL] == []


async def test_vwap_exit_ignores_the_forming_minute(tmp_path: Path) -> None:
    broker = FakeBroker()
    trader, clock = await _held_from_0901(tmp_path, broker)
    broker.bars = {"005930": _bars(*_UP_THEN_BREAK)}
    clock.now = clock.now.replace(minute=11, second=30)  # 09:11 bar still forming
    trader.manage()
    assert [o for o in broker.orders if o.side == OrderSide.SELL] == []
    clock.advance(seconds=60)  # 09:12:30, 09:11 closed, check interval passed
    trader.manage()
    assert len([o for o in broker.orders if o.side == OrderSide.SELL]) == 1


async def test_vwap_check_is_rate_limited_per_ticker(tmp_path: Path) -> None:
    broker = FakeBroker()
    trader, clock = await _held_from_0901(tmp_path, broker)
    broker.bars = {"005930": _bars(("09:01", 73000, 10000))}
    clock.now = clock.now.replace(minute=12)
    trader.manage()
    clock.advance(seconds=20)
    trader.manage()
    assert broker.bar_calls == 1


async def test_no_chart_feed_means_hold_to_flatten(tmp_path: Path) -> None:
    broker = FakeBroker()  # bars None -> NotImplementedError
    trader, clock = await _held_from_0901(tmp_path, broker)
    clock.now = clock.now.replace(minute=30)
    trader.manage()
    assert [o for o in broker.orders if o.side == OrderSide.SELL] == []


async def test_unfilled_entry_is_not_checked_for_vwap_exit(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock("09:01")
    trader = _trader(tmp_path, broker, clock)
    await trader.enter(_signal("005930"))  # never filled
    broker.bars = {"005930": _bars(*_UP_THEN_BREAK)}
    clock.now = clock.now.replace(minute=3)
    trader.manage()
    assert broker.bar_calls == 0


async def test_unfilled_early_exit_is_repriced_after_retry_interval(tmp_path: Path) -> None:
    broker = FakeBroker()
    trader, clock = await _held_from_0901(tmp_path, broker)
    broker.bars = {"005930": _bars(*_UP_THEN_BREAK)}
    clock.now = clock.now.replace(minute=12)
    trader.manage()
    clock.advance(minutes=1)
    trader.manage()
    sells = [o for o in broker.orders if o.side == OrderSide.SELL]
    assert [o.limit_price for o in sells] == [Decimal("70000"), Decimal("69650")]
    assert broker.cancelled[-1] == "0000002"


# -- fill ledger -------------------------------------------------------------------


def _fills(tmp_path: Path) -> list[tuple]:
    return [
        (r["role"], r["ticker"], r["order_id"], r["side"], r["quantity"], r["avg_price"])
        for r in _rows(tmp_path)
        if r["event"] == "fill"
    ]


async def test_entry_fill_is_recorded_with_broker_avg_price(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock("10:00")
    trader = _trader(tmp_path, broker, clock)
    await trader.enter(_signal())
    trader.manage()
    assert _fills(tmp_path) == []
    broker.fill("0000001", "005930", price="70050")
    clock.advance(minutes=1)
    trader.manage()
    assert _fills(tmp_path) == [("entry", "005930", "0000001", "BUY", 1, "70050")]
    assert _events(tmp_path)[-1] == "entry_resolved"
    clock.advance(minutes=5)
    trader.manage()  # resolved: no TTL cancel, no second fill row
    assert broker.cancelled == []
    assert len(_fills(tmp_path)) == 1


async def test_exit_fill_is_recorded_and_polling_stops(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock("10:00")
    trader = _trader(tmp_path, broker, clock)
    await trader.enter(_signal())
    broker.fill("0000001", "005930")  # vanished like Kiwoom: no price, no fill row
    clock.now = clock.now.replace(hour=15, minute=0)
    trader.manage()  # exit 0000002 submitted
    broker.fill("0000002", "005930", price="69990")
    clock.advance(minutes=1)
    trader.manage()
    assert _fills(tmp_path) == [("exit", "005930", "0000002", "SELL", 1, "69990")]
    assert [r for r in _rows(tmp_path) if r["event"] == "exit_resolved"][0]["order_id"] == (
        "0000002"
    )
    clock.advance(minutes=1)
    trader.manage()
    assert len(_fills(tmp_path)) == 1
    assert sum(o.side == OrderSide.SELL for o in broker.orders) == 1


async def test_vanished_exit_order_is_resolved_without_a_fill_row(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock("10:00")
    trader = _trader(tmp_path, broker, clock)
    await trader.enter(_signal())
    broker.fill("0000001", "005930")
    clock.now = clock.now.replace(hour=15, minute=0)
    trader.manage()
    broker.fill("0000002", "005930")  # Kiwoom: gone from the unfilled list
    clock.advance(minutes=1)
    trader.manage()
    assert _fills(tmp_path) == []
    assert "exit_resolved" in _events(tmp_path)


async def test_fill_poll_is_rate_limited_and_never_resolves_entries_on_error(
    tmp_path: Path,
) -> None:
    broker, clock = FakeBroker(), Clock("10:00")
    trader = _trader(tmp_path, broker, clock)
    await trader.enter(_signal())
    calls = 0
    real = broker.get_order_status

    def counting(order_id: str) -> OrderStatus:
        nonlocal calls
        calls += 1
        return real(order_id)

    broker.get_order_status = counting  # type: ignore[method-assign]
    trader.manage()
    clock.advance(seconds=20)
    trader.manage()
    assert calls == 1
    broker.fill("0000001", "005930")  # status now raises
    clock.advance(minutes=1)
    trader.manage()
    assert "entry_resolved" not in _events(tmp_path)  # TTL path still owns that call
    clock.advance(minutes=5)
    trader.manage()
    assert _events(tmp_path)[-1] == "entry_resolved"


async def test_add_on_fill_is_recorded(tmp_path: Path) -> None:
    broker = FakeBroker()
    trader, clock = await _held_from_0901(tmp_path, broker, _PYRAMID)
    broker.bars = {"005930": _rising()}
    clock.now = clock.now.replace(minute=26)
    trader.manage()
    broker.fill("0000002", "005930", price="70100")
    clock.advance(minutes=1)
    trader.manage()
    assert _fills(tmp_path) == [("add_on", "005930", "0000002", "BUY", 1, "70100")]
    assert "add_on_resolved" in _events(tmp_path)


def test_rank_batch_orders_by_surprise_then_score() -> None:
    a, b, c = _signal("000001"), _signal("000002"), _signal("000003")
    ranked = rank_batch([(a, 0, 0.9), (b, 1, 0.2), (c, 1, 0.5)])
    assert [s.ticker.value for s in ranked] == ["000003", "000002", "000001"]


# -- pyramiding (불타기) ----------------------------------------------------------


def _rising(until_minute: int = 25, step: int = 50) -> list[MinuteBar]:
    """09:01..09:MM closes 70000 +50/min, even volume: last close above VWAP, MA5 > MA20."""
    return _bars(*[(f"09:{m:02d}", 70000 + step * m, 100) for m in range(1, until_minute + 1)])


def _buys(broker: FakeBroker) -> list:
    return [o for o in broker.orders if o.side == OrderSide.BUY]


#: 불타기는 기본값으로 꺼져 있다. 이 절의 테스트는 켠 상태의 동작을 검증한다.
_PYRAMID = TradingLimits(add_on_min_gain=Decimal("0.01"))


async def test_add_on_is_off_by_default(tmp_path: Path) -> None:
    broker = FakeBroker()
    trader, clock = await _held_from_0901(tmp_path, broker)
    broker.bars = {"005930": _rising()}
    clock.now = clock.now.replace(minute=26)
    trader.manage()
    assert len(_buys(broker)) == 1
    assert "add_on_submitted" not in _events(tmp_path)


async def test_add_on_buys_one_share_when_trend_holds(tmp_path: Path) -> None:
    broker = FakeBroker()
    trader, clock = await _held_from_0901(tmp_path, broker, _PYRAMID)
    broker.bars = {"005930": _rising()}  # last close 71250: +1.6% vs 70100
    clock.now = clock.now.replace(minute=26)
    trader.manage()
    buys = _buys(broker)
    assert [(o.quantity, o.order_type, o.limit_price) for o in buys[1:]] == [
        (1, OrderType.LIMIT, Decimal("70100"))
    ]
    assert "add_on_submitted" in _events(tmp_path)


async def test_add_on_happens_at_most_once_per_ticker(tmp_path: Path) -> None:
    broker = FakeBroker()
    trader, clock = await _held_from_0901(tmp_path, broker, _PYRAMID)
    broker.bars = {"005930": _rising()}
    clock.now = clock.now.replace(minute=26)
    trader.manage()
    broker.fill("0000002", "005930")
    broker.bars = {"005930": _rising(28)}
    clock.advance(minutes=2)
    trader.manage()
    assert len(_buys(broker)) == 2


async def test_no_add_on_below_minimum_gain(tmp_path: Path) -> None:
    broker = FakeBroker()
    trader, clock = await _held_from_0901(tmp_path, broker, _PYRAMID)
    broker.bars = {"005930": _rising(step=20)}  # last close 70500: +0.6%
    clock.now = clock.now.replace(minute=26)
    trader.manage()
    assert len(_buys(broker)) == 1


async def test_no_add_on_with_kill_switch(tmp_path: Path) -> None:
    from daytrade_it.interfaces.mcp import tools

    broker = FakeBroker()
    trader, clock = await _held_from_0901(tmp_path, broker, _PYRAMID)
    tools._KILL_SWITCH_PATH.touch()
    broker.bars = {"005930": _rising()}
    clock.now = clock.now.replace(minute=26)
    trader.manage()
    assert len(_buys(broker)) == 1


async def test_no_add_on_past_exposure_cap(tmp_path: Path) -> None:
    broker = FakeBroker()
    clock = Clock("09:01")
    trader = AutoTrader(
        broker,
        tmp_path / "ledger",
        live=True,
        clock=clock,
        limits=TradingLimits(
            max_trader_exposure=Decimal("100000"), add_on_min_gain=Decimal("0.01")
        ),
    )
    await trader.enter(_signal("005930"))
    broker.fill("0000001", "005930")
    broker.bars = {"005930": _rising()}
    clock.now = clock.now.replace(minute=26)
    trader.manage()
    assert len(_buys(broker)) == 1


async def test_flatten_sells_both_shares_after_filled_add_on(tmp_path: Path) -> None:
    broker = FakeBroker()
    trader, clock = await _held_from_0901(tmp_path, broker, _PYRAMID)
    broker.bars = {"005930": _rising()}
    clock.now = clock.now.replace(minute=26)
    trader.manage()
    broker.fill("0000002", "005930")
    clock.now = clock.now.replace(hour=15, minute=0)
    trader.manage()
    sells = [o for o in broker.orders if o.side == OrderSide.SELL]
    assert [(o.ticker, o.quantity) for o in sells] == [("005930", 2)]


async def test_unfilled_add_on_is_cancelled_and_original_share_still_sold(tmp_path: Path) -> None:
    broker = FakeBroker()
    trader, clock = await _held_from_0901(tmp_path, broker, _PYRAMID)
    broker.bars = {"005930": _rising()}
    clock.now = clock.now.replace(minute=26)
    trader.manage()
    clock.advance(minutes=5)
    trader.manage()
    assert "0000002" in broker.cancelled
    assert "add_on_cancelled" in _events(tmp_path)
    clock.now = clock.now.replace(hour=15, minute=0)
    trader.manage()
    sells = [o for o in broker.orders if o.side == OrderSide.SELL]
    assert [(o.ticker, o.quantity) for o in sells] == [("005930", 1)]


async def test_vwap_exit_cancels_pending_add_on_first(tmp_path: Path) -> None:
    broker = FakeBroker()
    trader, clock = await _held_from_0901(tmp_path, broker, _PYRAMID)
    broker.bars = {"005930": _rising()}
    clock.now = clock.now.replace(minute=26)
    trader.manage()  # add-on 0000002 pending
    # 급등 후 VWAP 아래로: 진입 +2% 이상, VWAP -1% 이탈
    broker.bars = {
        "005930": _bars(("09:01", 73000, 10000), ("09:05", 73000, 10000), ("09:27", 71600, 100))
    }
    clock.advance(minutes=2)
    trader.manage()
    assert "0000002" in broker.cancelled
    assert [o.side for o in broker.orders][-1] == OrderSide.SELL


# -- baseline metadata: what the replay needs to rerun the day (2026-10-05) ----


def test_limits_round_trip_through_strings() -> None:
    limits = TradingLimits(
        max_open_positions=2,
        vwap_exit_gap=None,
        add_on_min_gain=Decimal("0.01"),
        entry_order_ttl=dt.timedelta(minutes=5),
        entry_start=dt.time(9, 0, 30),
    )
    raw = limits_to_strings(limits)
    assert all(isinstance(v, str) for v in raw.values())
    assert raw["max_open_positions"] == "2" and raw["vwap_exit_gap"] == "None"
    assert raw["add_on_min_gain"] == "0.01" and raw["entry_start"] == "09:00:30"
    assert limits_from_strings(raw) == limits
    assert limits_from_strings(limits_to_strings(TradingLimits())) == TradingLimits()


def test_limits_from_strings_tolerates_a_newer_ledger_with_unknown_keys() -> None:
    raw = {**limits_to_strings(TradingLimits()), "future_knob": "1"}
    assert limits_from_strings(raw) == TradingLimits()


async def test_baseline_records_git_sha_limits_and_cash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(auto_trader, "_git_sha", lambda: "deadbeef")
    limits = TradingLimits(max_open_positions=3, add_on_min_gain=Decimal("0.01"))
    broker, clock = FakeBroker(), Clock("10:00")
    await _trader(tmp_path, broker, clock, limits=limits).enter(_signal())
    [baseline] = [r for r in _rows(tmp_path) if r["event"] == "baseline"]
    assert baseline["positions"] == {}
    assert baseline["git_sha"] == "deadbeef"
    assert baseline["cash"] == "10000000"
    assert limits_from_strings(baseline["limits"]) == limits


async def test_baseline_without_account_or_git_still_lets_the_entry_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(auto_trader, "_git_sha", lambda: None)

    class _NoAccountOnce(FakeBroker):
        calls = 0

        def get_account(self) -> Account:
            _NoAccountOnce.calls += 1
            if _NoAccountOnce.calls == 1:
                raise BrokerError("account down", broker_name="fake")
            return super().get_account()

    broker, clock = _NoAccountOnce(), Clock("10:00")
    await _trader(tmp_path, broker, clock).enter(_signal())
    [baseline] = [r for r in _rows(tmp_path) if r["event"] == "baseline"]
    assert baseline["cash"] is None and baseline["git_sha"] is None
    assert _events(tmp_path) == ["baseline", "entry_submitted"]


def test_git_sha_is_a_commit_hash_or_none() -> None:
    sha = auto_trader._git_sha()
    assert sha is None or (len(sha) == 40 and int(sha, 16) >= 0)
