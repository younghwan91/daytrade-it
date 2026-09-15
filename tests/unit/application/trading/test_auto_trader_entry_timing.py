"""Entry-timing rules (delay, pullback, VWAP reclaim, limit ticks), all off by default."""

import datetime as dt
import json
from decimal import Decimal
from pathlib import Path

import pytest
import time_machine

from daytrade_it.application.replay.day_replay import snapshot_dart
from daytrade_it.application.trading.auto_trader import AutoTrader, TradingLimits, entry_ready
from daytrade_it.domain.entities.signal import SignalDirection, SignalSource, TradingSignal
from daytrade_it.domain.value_objects import Confidence, Ticker
from daytrade_it.infrastructure.brokers.base import KST, MinuteBar
from daytrade_it.infrastructure.brokers.sim_broker import SimBroker
from tests.unit.application.replay.test_day_replay import _dart

DAY = dt.date(2026, 9, 15)
T = "000001"
_OFF = {"vwap_exit_gap": None, "add_on_min_gain": None}


def _t(hms: str) -> dt.datetime:
    h, m, s = map(int, hms.split(":"))
    return dt.datetime(2026, 9, 15, h, m, s, tzinfo=KST)


def _bar(hm: str, close: float, volume: int = 100) -> MinuteBar:
    h, m = map(int, hm.split(":"))
    c = Decimal(str(close))
    return MinuteBar(dt.datetime(2026, 9, 15, h, m), c, c, c, c, volume)


# -- entry_ready (pure) ---------------------------------------------------------


def test_entry_ready_delay_boundary() -> None:
    lim = TradingLimits(entry_delay=dt.timedelta(minutes=5))
    sig = _t("09:05:10")
    assert not entry_ready([], Decimal(100), sig, _t("09:10:09"), lim)
    assert entry_ready([], Decimal(100), sig, _t("09:10:10"), lim)


def test_entry_ready_pullback_uses_completed_bars_after_signal() -> None:
    lim = TradingLimits(entry_pullback=Decimal("0.01"))
    sig = _t("09:05:10")
    ref = Decimal(100)
    bars = [_bar("09:04", 90), _bar("09:05", 100), _bar("09:06", 99)]
    # 09:06 is not complete at 09:06:59; 09:04 (pre-signal) never counts.
    assert not entry_ready(bars[:2], ref, sig, _t("09:06:00"), lim)
    assert not entry_ready(bars, ref, sig, _t("09:06:59"), lim)
    assert entry_ready(bars, ref, sig, _t("09:07:00"), lim)  # 99 <= 100 x 0.99
    above = [*bars[:2], _bar("09:06", 99.01)]
    assert not entry_ready(above, ref, sig, _t("09:07:00"), lim)


def test_entry_ready_vwap_reclaim_needs_a_close_below_then_above() -> None:
    lim = TradingLimits(entry_vwap_reclaim=True)
    sig = _t("09:02:10")
    ref = Decimal(100)
    base = [_bar("09:00", 100, 1000), _bar("09:01", 100, 1000)]
    only_above = [*base, _bar("09:02", 101), _bar("09:03", 102)]
    assert not entry_ready(only_above, ref, sig, _t("09:04:00"), lim)
    dip = [*base, _bar("09:02", 99), _bar("09:03", 100.5)]
    assert not entry_ready(dip, ref, sig, _t("09:03:59"), lim)  # 09:03 not complete
    assert entry_ready(dip, ref, sig, _t("09:04:00"), lim)
    # A dip before the signal minute does not count.
    early = [_bar("09:00", 100, 1000), _bar("09:01", 99, 1000), _bar("09:02", 101)]
    assert not entry_ready(early, ref, sig, _t("09:03:00"), lim)


def test_entry_ready_rules_are_anded() -> None:
    lim = TradingLimits(entry_delay=dt.timedelta(minutes=5), entry_pullback=Decimal("0.01"))
    sig = _t("09:05:10")
    bars = [_bar("09:05", 100), _bar("09:06", 98)]
    assert not entry_ready(bars, Decimal(100), sig, _t("09:07:00"), lim)  # delay not met
    assert entry_ready([*bars, _bar("09:10", 98)], Decimal(100), sig, _t("09:11:00"), lim)


# -- AutoTrader through SimBroker ------------------------------------------------


def _session(price_at: dict[str, int] | None = None) -> list[MinuteBar]:
    """Flat 10,100 bars from 09:00; `price_at` switches the price from HH:MM on."""
    out, t, p = [], dt.datetime(2026, 9, 15, 9, 0), Decimal(10100)
    while t.time() <= dt.time(15, 30):
        hm = t.strftime("%H:%M")
        if price_at and hm in price_at:
            p = Decimal(price_at[hm])
        out.append(MinuteBar(ts=t, open=p, high=p, low=p, close=p, volume=1000))
        t += dt.timedelta(minutes=1)
    return out


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    from daytrade_it.interfaces.mcp import tools

    monkeypatch.setattr(tools, "_DART_DB_PATH", snapshot_dart(_dart(tmp_path), DAY, tmp_path / "s"))
    monkeypatch.setattr(tools, "_KILL_SWITCH_PATH", tmp_path / "no_kill_switch")
    with time_machine.travel(_t("09:05:00"), tick=False) as traveller:
        yield traveller


def _broker(bars: list[MinuteBar] | None = None) -> SimBroker:
    return SimBroker({T: bars or _session()}, {T: Decimal(10000)}, lambda: dt.datetime.now(KST))


def _trader(tmp_path: Path, broker: SimBroker, **limits: object) -> AutoTrader:
    return AutoTrader(
        broker,
        tmp_path / "ledger",
        live=True,
        limits=TradingLimits(**{**_OFF, **limits}),
        clock=lambda: dt.datetime.now(KST),
    )


def _signal() -> TradingSignal:
    return TradingSignal(
        ticker=Ticker(value=T),
        direction=SignalDirection.BUY,
        confidence=Confidence(value=0.8),
        source=SignalSource.COMPOSITE,
        expires_at=dt.datetime.now(dt.UTC) + dt.timedelta(hours=2),
        metadata={"prev_close": "10000"},
    )


def _ledger(tmp_path: Path) -> list[dict]:
    path = tmp_path / "ledger" / f"{DAY}.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def _events(tmp_path: Path) -> list[str]:
    return [r["event"] for r in _ledger(tmp_path)]


async def test_default_limits_enter_at_once(tmp_path: Path, env) -> None:  # noqa: ANN001
    trader = _trader(tmp_path, _broker())
    await trader.enter(_signal())
    assert _events(tmp_path) == ["baseline", "entry_submitted"]
    assert _ledger(tmp_path)[-1]["limit"] == "10100"


async def test_entry_delay_defers_then_enters_on_a_later_tick(tmp_path: Path, env) -> None:  # noqa: ANN001
    trader = _trader(tmp_path, _broker(), entry_delay=dt.timedelta(minutes=5))
    sig = _signal()
    await trader.enter(sig)
    assert _events(tmp_path) == ["baseline", "entry_deferred"]
    row = _ledger(tmp_path)[-1]
    assert (row["ticker"], row["signal_id"], row["ref_price"]) == (T, str(sig.id), "10100")
    assert row["signaled_at"].startswith("2026-09-15T09:05:00")
    env.move_to(_t("09:09:59"))
    trader.manage()
    assert _events(tmp_path)[-1] == "entry_deferred"
    env.move_to(_t("09:10:00"))
    trader.manage()
    assert _events(tmp_path)[-1] == "entry_submitted"
    trader.manage()
    assert _events(tmp_path).count("entry_submitted") == 1


async def test_pullback_waits_for_completed_bar_via_manage(tmp_path: Path, env) -> None:  # noqa: ANN001
    broker = _broker(_session({"09:07": 9990}))
    trader = _trader(tmp_path, broker, entry_pullback=Decimal("0.01"))
    await trader.enter(_signal())
    env.move_to(_t("09:07:30"))
    trader.manage()  # 09:07 bar (9,990) not complete yet
    assert _events(tmp_path)[-1] == "entry_deferred"
    env.move_to(_t("09:08:30"))
    trader.manage()
    assert _events(tmp_path)[-1] == "entry_submitted"
    assert _ledger(tmp_path)[-1]["limit"] == "9990"


async def test_max_wait_expires_the_deferred_entry(tmp_path: Path, env) -> None:  # noqa: ANN001
    trader = _trader(tmp_path, _broker(), entry_pullback=Decimal("0.05"))
    await trader.enter(_signal())
    env.move_to(_t("09:35:00"))
    trader.manage()
    assert _events(tmp_path)[-1] == "entry_deferred"
    env.move_to(_t("09:35:30"))
    trader.manage()
    assert _events(tmp_path)[-1] == "entry_expired"
    assert _ledger(tmp_path)[-1]["reason"] == "max_wait"
    env.move_to(_t("09:36:30"))
    trader.manage()
    assert _events(tmp_path).count("entry_expired") == 1


async def test_no_chase_is_reapplied_at_deferred_entry(tmp_path: Path, env) -> None:  # noqa: ANN001
    broker = _broker(_session({"09:08": 10400}))  # 10,400 vs prev close 10,000 = +4%
    trader = _trader(tmp_path, broker, entry_delay=dt.timedelta(minutes=5))
    await trader.enter(_signal())
    env.move_to(_t("09:10:00"))
    trader.manage()
    assert _events(tmp_path)[-1] == "entry_expired"
    assert _ledger(tmp_path)[-1]["reason"] == "extended_at_entry"
    assert not broker.fills


async def test_no_chase_reads_the_module_global_at_call_time(
    tmp_path: Path,
    env,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daytrade_it.application.signals import entry_filter

    monkeypatch.setattr(entry_filter, "MAX_PRE_ENTRY_MOVE", Decimal("0.05"))
    broker = _broker(_session({"09:08": 10400}))
    trader = _trader(tmp_path, broker, entry_delay=dt.timedelta(minutes=5))
    await trader.enter(_signal())
    env.move_to(_t("09:10:00"))
    trader.manage()
    assert _events(tmp_path)[-1] == "entry_submitted"


async def test_limit_ticks_price_one_tick_below_the_ask(tmp_path: Path, env) -> None:  # noqa: ANN001
    broker = _broker()
    trader = _trader(tmp_path, broker, entry_limit_ticks=1)
    await trader.enter(_signal())
    assert _events(tmp_path) == ["baseline", "entry_submitted"]
    assert _ledger(tmp_path)[-1]["limit"] == "10090"  # tick 10 at 10,100
    assert broker._open  # resting below the ask


async def test_restart_closes_deferred_entries_without_a_signal(tmp_path: Path, env) -> None:  # noqa: ANN001
    broker = _broker()
    first = _trader(tmp_path, broker, entry_delay=dt.timedelta(minutes=5))
    await first.enter(_signal())
    second = _trader(tmp_path, broker, entry_delay=dt.timedelta(minutes=5))
    env.move_to(_t("09:10:00"))
    second.manage()
    assert _events(tmp_path)[-2:] == ["entry_deferred", "entry_expired"]
    assert _ledger(tmp_path)[-1]["reason"] == "restart"
    second.manage()
    assert "entry_submitted" not in _events(tmp_path)


async def test_a_second_signal_while_deferred_is_skipped(tmp_path: Path, env) -> None:  # noqa: ANN001
    trader = _trader(tmp_path, _broker(), entry_delay=dt.timedelta(minutes=5))
    await trader.enter(_signal())
    await trader.enter(_signal())
    assert _events(tmp_path) == ["baseline", "entry_deferred"]
