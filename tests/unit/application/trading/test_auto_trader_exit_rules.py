"""Stop-loss and profit-taking exit rules (`exit_rule_hit`), all off by default."""

import datetime as dt
import json
from decimal import Decimal
from pathlib import Path

from daytrade_it.application.trading.auto_trader import (
    AutoTrader,
    TradingLimits,
    exit_rule_hit,
    vwap_exit_hit,
)
from daytrade_it.infrastructure.brokers.base import KST, MinuteBar, OrderSide
from tests.unit.application.trading.test_auto_trader import (
    Clock,
    FakeBroker,
    _bars,
    _no_kill_switch,  # noqa: F401  (autouse fixture)
    _signal,
)

_OFF = {"vwap_exit_gap": None}


def _bar(
    hm: str,
    close: float,
    *,
    low: float | None = None,
    high: float | None = None,
    volume: int = 100,
) -> MinuteBar:
    h, m = map(int, hm.split(":"))
    c = Decimal(str(close))
    lo = Decimal(str(low)) if low is not None else c
    hi = Decimal(str(high)) if high is not None else c
    return MinuteBar(dt.datetime(2026, 9, 14, h, m), c, hi, lo, c, volume)


def _t(hms: str) -> dt.datetime:
    h, m, s = map(int, hms.split(":"))
    return dt.datetime(2026, 9, 14, h, m, s, tzinfo=KST)


E = Decimal(100)


def test_stop_loss_fires_at_threshold_on_completed_bar_only() -> None:
    lim = TradingLimits(stop_loss_pct=Decimal("0.02"), **_OFF)
    bars = [_bar("09:00", 100), _bar("09:01", 98)]
    entered = _t("09:00:30")
    assert exit_rule_hit(bars, E, entered, _t("09:01:59"), lim) is None  # 09:01 not complete
    assert exit_rule_hit(bars, E, entered, _t("09:02:00"), lim) == "stop_loss"
    above = [_bar("09:00", 100), _bar("09:01", 98.01)]
    assert exit_rule_hit(above, E, entered, _t("09:02:00"), lim) is None


def test_stop_vwap_needs_close_below_entry_and_below_vwap_gap() -> None:
    lim = TradingLimits(stop_vwap_gap=Decimal("0.02"), **_OFF)
    entered = _t("09:00:30")
    # VWAP ~101.97; 99 < entry and < 101.97 * 0.98 = 99.93.
    bars = [_bar("09:00", 102, volume=10000), _bar("09:01", 99)]
    assert exit_rule_hit(bars, E, entered, _t("09:02:00"), lim) == "stop_vwap"
    # 100 is well below VWAP but not below entry.
    at_entry = [_bar("09:00", 102, volume=10000), _bar("09:01", 100)]
    assert exit_rule_hit(at_entry, E, entered, _t("09:02:00"), lim) is None
    # Below entry but within the gap of VWAP (VWAP ~99.99, 99.5 > 97.99).
    near = [_bar("09:00", 100, volume=10000), _bar("09:01", 99.5)]
    assert exit_rule_hit(near, E, entered, _t("09:02:00"), lim) is None


def test_stop_time_fires_after_holding_period_without_min_gain() -> None:
    lim = TradingLimits(
        stop_time=dt.timedelta(minutes=30), stop_time_min_gain=Decimal("0.01"), **_OFF
    )
    entered = _t("09:00:30")
    bars = [_bar(f"09:{m:02d}", 100.5) for m in range(30)]
    assert exit_rule_hit(bars, E, entered, _t("09:30:29"), lim) is None  # 29m59s held
    assert exit_rule_hit(bars, E, entered, _t("09:30:30"), lim) == "stop_time"
    gained = [_bar(f"09:{m:02d}", 101) for m in range(30)]
    assert exit_rule_hit(gained, E, entered, _t("09:30:30"), lim) is None


def test_stop_orb_fires_on_close_below_opening_range_low_after_window() -> None:
    lim = TradingLimits(stop_orb=dt.timedelta(minutes=5), **_OFF)
    entered = _t("09:00:30")
    window = [_bar(f"09:0{m}", 100, low=99) for m in range(5)]
    bars = [*window, _bar("09:05", 98.5)]
    assert exit_rule_hit(bars, E, entered, _t("09:05:59"), lim) is None  # 09:05 forming
    assert exit_rule_hit(bars, E, entered, _t("09:06:00"), lim) == "stop_orb"
    held = [*window, _bar("09:05", 99)]
    assert exit_rule_hit(held, E, entered, _t("09:06:00"), lim) is None  # not below 99
    # Before the window has ended nothing fires, even on a low close.
    early = [_bar("09:00", 100, low=99), _bar("09:01", 99, low=98), _bar("09:02", 98)]
    assert exit_rule_hit(early, E, entered, _t("09:03:00"), lim) is None


def test_take_profit_fires_at_threshold() -> None:
    lim = TradingLimits(take_profit_pct=Decimal("0.03"), **_OFF)
    entered = _t("09:00:30")
    below = [_bar("09:00", 100), _bar("09:01", 102.99)]
    assert exit_rule_hit(below, E, entered, _t("09:02:00"), lim) is None
    at = [_bar("09:00", 100), _bar("09:01", 103)]
    assert exit_rule_hit(at, E, entered, _t("09:02:00"), lim) == "take_profit"


def test_trailing_uses_peak_after_entry_minute() -> None:
    lim = TradingLimits(trail_after_gain=Decimal("0.02"), trail_gap=Decimal("0.01"), **_OFF)
    entered = _t("09:00:30")
    base = [_bar("09:00", 100), _bar("09:01", 102), _bar("09:02", 101)]
    assert exit_rule_hit(base, E, entered, _t("09:03:00"), lim) is None  # 101 > 100.98
    fell = [*base, _bar("09:03", 100.98)]
    assert exit_rule_hit(fell, E, entered, _t("09:04:00"), lim) == "trailing"
    # A peak before the entry minute does not arm the trail.
    late = [_bar("09:00", 105), _bar("09:01", 101), _bar("09:02", 100)]
    assert exit_rule_hit(late, E, _t("09:01:10"), _t("09:03:00"), lim) is None


def test_vwap_break_is_the_existing_rule() -> None:
    lim = TradingLimits()
    bars = _bars(("09:01", 73000, 10000), ("09:05", 73000, 10000), ("09:11", 71600, 100))
    entry = Decimal(70100)
    assert exit_rule_hit(bars, entry, _t("09:01:00"), _t("09:12:00"), lim) == "vwap_break"
    assert exit_rule_hit(bars, entry, _t("09:01:00"), _t("09:11:30"), lim) is None


def test_rules_are_checked_in_order_stops_first() -> None:
    lim = TradingLimits(stop_loss_pct=Decimal("0.02"), stop_vwap_gap=Decimal("0.01"), **_OFF)
    bars = [_bar("09:00", 102, volume=10000), _bar("09:01", 97)]
    assert exit_rule_hit(bars, E, _t("09:00:30"), _t("09:02:00"), lim) == "stop_loss"


def test_default_limits_only_ever_match_vwap_exit_hit() -> None:
    lim = TradingLimits()
    assert lim.vwap_exit_gap is not None
    scenarios = [
        [_bar("09:00", 100), _bar("09:01", 50)],  # crash: no stop by default
        [_bar("09:00", 100), _bar("09:01", 200)],  # moon: no take-profit by default
        [_bar("09:00", 103, volume=10000), _bar("09:01", 102)],  # vwap break
        [_bar("08:59", 103, volume=10000), _bar("09:00", 102)],  # pre-market ignored
        [_bar("09:00", 102, volume=0)],
        [],
    ]
    for bars in scenarios:
        for now in (_t("09:01:30"), _t("09:02:00"), _t("15:00:00")):
            hit, _, _ = vwap_exit_hit(bars, E, now, lim.vwap_exit_gap, lim.vwap_exit_min_gain)
            got = exit_rule_hit(bars, E, _t("09:00:10"), now, lim)
            assert got == ("vwap_break" if hit else None)


# -- AutoTrader wiring ----------------------------------------------------------


def _rows(tmp_path: Path) -> list[dict]:
    path = tmp_path / "ledger" / "2026-09-14.jsonl"
    return [json.loads(line) for line in path.open()]


async def _held(tmp_path: Path, limits: TradingLimits) -> tuple[AutoTrader, FakeBroker, Clock]:
    broker, clock = FakeBroker(), Clock("09:01")
    trader = AutoTrader(broker, tmp_path / "ledger", live=True, limits=limits, clock=clock)
    await trader.enter(_signal("005930"))  # limit at ask 70100
    broker.fill("0000001", "005930")
    return trader, broker, clock


async def test_stop_loss_exit_records_rule_name_and_sells(tmp_path: Path) -> None:
    lim = TradingLimits(stop_loss_pct=Decimal("0.02"), vwap_exit_gap=None, add_on_min_gain=None)
    trader, broker, clock = await _held(tmp_path, lim)
    broker.bars = {"005930": _bars(("09:01", 70000, 100), ("09:05", 68000, 100))}
    clock.now = clock.now.replace(minute=6)
    trader.manage()
    signal = [r for r in _rows(tmp_path) if r["event"] == "exit_signal"]
    assert [r["reason"] for r in signal] == ["stop_loss"]
    assert signal[0]["close"] == "68000"
    assert [o.quantity for o in broker.orders if o.side == OrderSide.SELL] == [1]


async def test_stop_time_uses_entry_submitted_ts(tmp_path: Path) -> None:
    lim = TradingLimits(stop_time=dt.timedelta(minutes=10), vwap_exit_gap=None)
    trader, broker, clock = await _held(tmp_path, lim)
    broker.bars = {"005930": _bars(("09:01", 70000, 100), ("09:10", 70000, 100))}
    clock.now = clock.now.replace(minute=10)  # held 9m
    trader.manage()
    assert "exit_signal" not in [r["event"] for r in _rows(tmp_path)]
    clock.now = clock.now.replace(minute=11, second=0)
    trader.manage()
    assert [r["reason"] for r in _rows(tmp_path) if r["event"] == "exit_signal"] == ["stop_time"]


async def test_no_rule_enabled_fetches_no_bars(tmp_path: Path) -> None:
    lim = TradingLimits(vwap_exit_gap=None)
    trader, broker, clock = await _held(tmp_path, lim)
    broker.bars = {"005930": _bars(("09:01", 10000, 100))}
    clock.now = clock.now.replace(minute=30)
    trader.manage()
    assert broker.bar_calls == 0


async def test_default_vwap_break_ledger_row_is_unchanged(tmp_path: Path) -> None:
    trader, broker, clock = await _held(tmp_path, TradingLimits())
    broker.bars = {
        "005930": _bars(("09:01", 73000, 10000), ("09:05", 73000, 10000), ("09:11", 71600, 100))
    }
    clock.now = clock.now.replace(minute=12)
    trader.manage()
    (row,) = [r for r in _rows(tmp_path) if r["event"] == "exit_signal"]
    assert set(row) == {"ts", "event", "live", "ticker", "reason", "close", "vwap"}
    assert (row["reason"], row["close"]) == ("vwap_break", "71600")
    _, close, vwap = vwap_exit_hit(
        broker.bars["005930"], Decimal(70100), clock.now, Decimal("0.01"), Decimal("0.02")
    )
    assert (row["close"], row["vwap"]) == (str(close), str(vwap))


def test_trailing_ignores_entry_minute_high_before_entry() -> None:
    """Ruling S3: the entry-minute bar's high can predate the entry."""
    lim = TradingLimits(trail_after_gain=Decimal("0.02"), trail_gap=Decimal("0.01"), **_OFF)
    entered = _t("09:00:30")
    bars = [_bar("09:00", 100, high=103), _bar("09:01", 100.5)]
    assert exit_rule_hit(bars, E, entered, _t("09:02:00"), lim) is None
