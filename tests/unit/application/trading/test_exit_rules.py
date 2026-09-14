"""exit_rules 시뮬레이터 — 체결 가정은 스펙 '시뮬레이터' 절."""

from datetime import datetime, time

import pytest

from daytrade_it.application.trading.exit_rules import (
    Bar,
    Baseline,
    MomentumFade,
    StopTake,
    TimeExit,
    Trailing,
    net_return,
    simulate_exit,
)


def bars_from(
    closes: list[float], start: str = "2026-09-01T09:00", vols: list[int] | None = None
) -> list[Bar]:
    t0 = datetime.fromisoformat(start)
    out = []
    for i, c in enumerate(closes):
        prev = closes[i - 1] if i else c
        out.append(
            Bar(
                ts=t0.replace(hour=t0.hour + (t0.minute + i) // 60, minute=(t0.minute + i) % 60),
                open=prev,
                high=max(prev, c),
                low=min(prev, c),
                close=c,
                volume=(vols[i] if vols else 1000),
            )
        )
    return out


def test_net_return_includes_tax_and_fees() -> None:
    assert net_return(100.0, 100.0) == pytest.approx((1 - 0.002 - 0.00015) / (1 + 0.00015) - 1)


def test_baseline_exits_at_first_bar_at_or_after_flatten() -> None:
    bars = bars_from([100.0] * 10, start="2026-09-01T14:55")
    r = simulate_exit(bars, 0, 100.0, Baseline())
    assert r.exit_ts.time() == time(15, 0)
    assert r.reason == "flatten"


def test_data_end_before_flatten_exits_at_last_close() -> None:
    bars = bars_from([100.0, 101.0, 102.0])
    r = simulate_exit(bars, 0, 100.0, Baseline())
    assert (r.exit_price, r.reason) == (102.0, "data_end")


def test_trailing_triggers_on_close_and_fills_next_open() -> None:
    # 고점 110, 종가 107 ≤ 110×0.98 → 다음 봉 시가(=107)
    bars = bars_from([100.0, 110.0, 107.0, 90.0])
    r = simulate_exit(bars, 0, 100.0, Trailing(0.02))
    assert r.reason == "trailing"
    assert r.exit_ts == bars[3].ts
    assert r.exit_price == bars[3].open


def test_stop_and_take_in_same_bar_takes_stop() -> None:
    b = Bar(datetime(2026, 9, 1, 9, 1), 100.0, 106.0, 96.0, 100.0, 1000)
    bars = [Bar(datetime(2026, 9, 1, 9, 0), 100.0, 100.0, 100.0, 100.0, 1000), b]
    r = simulate_exit(bars, 0, 100.0, StopTake(0.03, 0.05))
    assert (r.reason, r.exit_price) == ("stop", 97.0)


def test_stop_gap_down_fills_at_open() -> None:
    bars = [
        Bar(datetime(2026, 9, 1, 9, 0), 100.0, 100.0, 100.0, 100.0, 1000),
        Bar(datetime(2026, 9, 1, 9, 1), 90.0, 91.0, 89.0, 90.0, 1000),
    ]
    assert simulate_exit(bars, 0, 100.0, StopTake(0.03, None)).exit_price == 90.0


def test_take_profit_fills_at_target() -> None:
    bars = bars_from([100.0, 106.0, 106.0])
    r = simulate_exit(bars, 0, 100.0, StopTake(0.03, 0.05))
    assert (r.reason, r.exit_price) == ("take", 105.0)


def test_time_exit_after_n_minutes() -> None:
    bars = bars_from([100.0] * 40)
    r = simulate_exit(bars, 0, 100.0, TimeExit(30))
    assert (r.reason, r.exit_ts) == ("time", bars[30].ts)


def test_momentum_fade_needs_gain_stall_and_volume_drop() -> None:
    closes = [100.0, 102.0, 104.0] + [103.0] * 6
    vols = [1000, 1000, 1000, 1000, 1000, 1000, 100, 100, 100]
    bars = bars_from(closes, vols=vols)
    r = simulate_exit(bars, 0, 100.0, MomentumFade(0.03, 3))
    # 고점 bar2(104). bar7 종가 시점: 고점 뒤 5봉 신고가 없음, 최근 3봉 거래량 1200 < 0.5×3000 → bar8 시가
    assert (r.reason, r.exit_ts) == ("fade", bars[8].ts)


def test_momentum_fade_not_triggered_without_min_gain() -> None:
    closes = [100.0, 101.0] + [100.5] * 10
    vols = [1000] * 6 + [10] * 6
    r = simulate_exit(bars_from(closes, vols=vols), 0, 100.0, MomentumFade(0.03, 3))
    assert r.reason == "data_end"
