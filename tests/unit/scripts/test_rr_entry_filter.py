"""rr_entry_filter.simulate: 사전등록 규칙(2026-10-08-rr-entry-prereg.json) 그대로인가."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from rr_entry_filter import _start_minute, simulate  # noqa: E402

C = 0.004


def bars(*rows):
    """(분, open, high, low, close) — 분은 09:00 기준."""
    return np.asarray(rows, float)


BASE = [
    (0, 100, 110, 98, 100),  # 고가 110, 저가 98
    (1, 99, 99.5, 99, 99),  # t=1 진입: E=99, S=98, T=110 → rr ≈ 7.6
]


def test_target_hit_after_entry():
    r = simulate(bars(*BASE, (2, 99, 111, 99, 110)), 1, float("nan"), 100, C, C)
    assert r["entry_min"] == 1 and r["E"] == 99 and r["S"] == 98 and r["T"] == 110
    assert r["reason"] == "target"
    assert r["net_bp"] == pytest.approx((110 / 99 - 1 - C) * 1e4)


def test_stop_and_target_same_bar_is_stop():
    r = simulate(bars(*BASE, (2, 99, 111, 97, 100)), 1, float("nan"), 100, C, C)
    assert r["reason"] == "stop"
    assert r["net_bp"] < 0


def test_stop_uses_only_completed_bars():
    """진입 봉 자신의 저가는 손절가 계산에 안 들어가고, 그 봉에서 손절이 난다."""
    r = simulate(bars(BASE[0], (1, 99, 99.5, 97, 98)), 1, float("nan"), 100, C, C)
    assert r["S"] == 98
    assert r["reason"] == "stop" and r["exit_min"] == 1


def test_gap_below_stop_fills_at_open():
    r = simulate(bars(*BASE, (2, 95, 96, 94, 95)), 1, float("nan"), 100, C, C)
    assert r["reason"] == "gap_stop"
    assert r["net_bp"] == pytest.approx((95 / 99 - 1 - C) * 1e4)


def test_no_resistance_above_means_no_setup():
    b = bars((0, 100, 101, 98, 101), (1, 102, 103, 102, 103))
    assert simulate(b, 1, 101.0, 100, C, C) is None


def test_prev_high_is_nearest_resistance():
    r = simulate(bars(*BASE, (2, 99, 105, 99, 105)), 1, 104.0, 100, C, C)
    assert r["T"] == 104 and r["reason"] == "target"


def test_rr_below_two_is_skipped():
    # E=105, S=98, T=110 → (110−105−0.42)/(105−98+0.42) ≈ 0.62
    b = bars((0, 100, 110, 98, 105), (1, 105, 106, 104, 105))
    assert simulate(b, 1, float("nan"), 100, C, C) is None


def test_start_minute_respected():
    b = bars(*BASE, (2, 99, 99, 99, 99), (3, 99, 99, 99, 99))
    r = simulate(b, 3, float("nan"), 100, C, C)
    assert r["entry_min"] == 3


def test_flat_at_1500_open():
    b = bars(*BASE, (2, 100, 101, 99.5, 101), (360, 101, 102, 100, 101))
    r = simulate(b, 1, float("nan"), 100, C, C)
    assert r["reason"] == "time" and r["exit_min"] == 360
    assert r["net_bp"] == pytest.approx((101 / 99 - 1 - C) * 1e4)


def test_scan_once_gives_up_after_first_minute():
    b = bars((0, 100, 110, 98, 105), (1, 105, 106, 104, 105), (2, 99, 99, 99, 99))
    assert simulate(b, 1, float("nan"), 100, C, C)["entry_min"] == 2
    assert simulate(b, 1, float("nan"), 100, C, C, scan_once=True) is None


def test_limit_up_entry_skipped():
    # 전일 종가 100 → 상한가 130. E=130 이면 매수 불가.
    b = bars((0, 125, 140, 120, 130), (1, 130, 130, 130, 130))
    assert simulate(b, 1, 200.0, 100, C, C) is None


def test_seam_excluded():
    b = bars((0, 50, 51, 49, 50), (1, 50, 51, 49, 50))
    assert simulate(b, 1, float("nan"), 100, C, C) == {"excluded": True}


def test_start_minute_ceils_and_clamps():
    assert _start_minute(pd.Timestamp("2026-04-07 13:16:10")) == 13 * 60 + 17 - 540
    assert _start_minute(pd.Timestamp("2026-04-07 13:16:00")) == 13 * 60 + 16 - 540
    assert _start_minute(pd.Timestamp("2026-04-07 08:30:00")) == 1
