"""volume_entry.simulate: 사전등록 규칙(2026-10-09-volume-entry-prereg.json) 그대로인가."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from volume_entry import minute_base, simulate  # noqa: E402

C = 0.004
BASE = 1_000.0  # 평균 1분 거래대금. 5배 = 5,000


def bars(*rows):
    """(분, open, high, low, close, volume)."""
    return np.asarray(rows, float)


QUIET = (0, 100, 100, 100, 100, 1)
SPIKE_UP = (1, 100, 101, 100, 101, 60)  # 101 × 60 = 6,060 ≥ 5,000, 양봉


def test_enters_next_bar_open_after_spike():
    r = simulate(bars(QUIET, SPIKE_UP, (2, 102, 102, 102, 102, 1)), 1, BASE, 100, C, C)
    assert r["trigger_min"] == 1 and r["entry_min"] == 2 and r["E"] == 102


def test_target_plus_4pct():
    r = simulate(
        bars(QUIET, SPIKE_UP, (2, 100, 100, 100, 100, 1), (3, 100, 104.5, 100, 104, 1)),
        1,
        BASE,
        100,
        C,
        C,
    )
    assert r["reason"] == "target"
    assert r["net_bp"] == pytest.approx((0.04 - C) * 1e4)


def test_stop_minus_2pct_one_tick_worse():
    r = simulate(
        bars(QUIET, SPIKE_UP, (2, 100, 100, 100, 100, 1), (3, 99, 99, 97.9, 98, 1)),
        1,
        BASE,
        100,
        C,
        C,
    )
    assert r["reason"] == "stop"
    # S = 98, 1틱(1원) 아래 97
    assert r["net_bp"] == pytest.approx((97 / 100 - 1 - C) * 1e4)


def test_same_bar_stop_wins():
    r = simulate(bars(QUIET, SPIKE_UP, (2, 100, 105, 97, 100, 1)), 1, BASE, 100, C, C)
    assert r["reason"] == "stop"


def test_down_bar_spike_ignored():
    b = bars(QUIET, (1, 101, 101, 100, 100, 60), (2, 100, 100, 100, 100, 1))
    assert simulate(b, 1, BASE, 100, C, C) is None


def test_below_threshold_ignored():
    b = bars(QUIET, (1, 100, 101, 100, 101, 40), (2, 100, 100, 100, 100, 1))  # 4,040 < 5,000
    assert simulate(b, 1, BASE, 100, C, C) is None


def test_open_auction_bar_is_not_a_trigger():
    b = bars((0, 100, 101, 100, 101, 1000), (1, 101, 101, 101, 101, 1))
    assert simulate(b, 1, BASE, 100, C, C) is None


def test_spike_before_decide_ignored():
    b = bars(QUIET, SPIKE_UP, (2, 100, 100, 100, 100, 1), (3, 100, 100, 100, 100, 1))
    assert simulate(b, 3, BASE, 100, C, C) is None  # 트리거는 ts >= start-1 = 2 부터


def test_limit_up_entry_skips_to_next_trigger():
    b = bars(
        (0, 125, 125, 125, 125, 1),
        (1, 125, 130, 125, 130, 60),
        (2, 130, 130, 129, 129.5, 1),  # 상한가 130 진입 불가
        (3, 128, 129, 128, 129, 60),
        (4, 128, 128, 128, 128, 1),
    )
    r = simulate(b, 1, BASE, 100, C, C)
    assert r["trigger_min"] == 3 and r["E"] == 128


def test_no_entry_after_1430():
    b = bars(QUIET, (330, 100, 101, 100, 101, 60), (331, 101, 101, 101, 101, 1))
    assert simulate(b, 1, BASE, 100, C, C) is None


def test_flat_at_1500():
    b = bars(QUIET, SPIKE_UP, (2, 100, 101, 99, 100, 1), (360, 101, 101, 101, 101, 1))
    r = simulate(b, 1, BASE, 100, C, C)
    assert r["reason"] == "time" and r["exit_min"] == 360


def test_no_baseline_no_trade():
    assert (
        simulate(bars(QUIET, SPIKE_UP, (2, 100, 100, 100, 100, 1)), 1, float("nan"), 100, C, C)
        is None
    )


def test_minute_base_converts_million_won():
    # 일 거래대금 7,600 백만원(76억) → 1분 평균 2,000만원
    assert minute_base(7_600) == pytest.approx(20_000_000)
