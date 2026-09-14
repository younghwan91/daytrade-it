"""AutoTrader: 1-share entries through execute_trade, flat by the close."""

import datetime as dt
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest

from daytrade_it.application.trading.auto_trader import AutoTrader, TradingLimits, rank_batch
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

    def fill(self, order_id: str, ticker: str) -> None:
        self.unfilled.pop(order_id)
        self.positions[ticker] = self.positions.get(ticker, 0) + 1

    def get_order_status(self, order_id: str) -> OrderStatus:
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


def _trader(tmp_path: Path, broker: FakeBroker, clock: Clock, *, live: bool = True) -> AutoTrader:
    return AutoTrader(broker, tmp_path / "ledger", live=live, clock=clock)


def _events(tmp_path: Path) -> list[str]:
    path = tmp_path / "ledger" / "2026-09-14.jsonl"
    return [json.loads(line)["event"] for line in path.open()] if path.exists() else []


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


async def _held_from_0901(tmp_path: Path, broker: FakeBroker) -> tuple[AutoTrader, Clock]:
    clock = Clock("09:01")
    trader = _trader(tmp_path, broker, clock)
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


def test_rank_batch_orders_by_surprise_then_score() -> None:
    a, b, c = _signal("000001"), _signal("000002"), _signal("000003")
    ranked = rank_batch([(a, 0, 0.9), (b, 1, 0.2), (c, 1, 0.5)])
    assert [s.ticker.value for s in ranked] == ["000003", "000002", "000001"]
