"""The research cost rate must stay equal to the broker's own model.

Research netted out a hand-written `COST = 0.00215` for months while
`PaperBroker` charged 15+ bp more, and nothing failed. These tests make that
divergence fail here instead of quietly inflating a backtest.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "eval"))

from costs import MEASURED_SPREAD, round_trip  # noqa: E402

DAY = dt.date(2026, 9, 15)
#: the literal that was wrong -- 0.20% sell tax + one side of commission, no spread
RETIRED_CONSTANT = 0.00215


@pytest.mark.parametrize("market", ["거래소", "코스닥"])
@pytest.mark.parametrize("at_open", [True, False])
def test_round_trip_exceeds_the_retired_constant(market: str, at_open: bool) -> None:
    """Whatever the model says, it must not silently collapse back to 0.00215."""
    assert round_trip(DAY, market, at_open=at_open) > RETIRED_CONSTANT + 0.001


def test_components_add_up() -> None:
    """tax 0.20% + commission 0.015% x 2 + the measured spread, and nothing else."""
    for (market, at_open), spread in MEASURED_SPREAD.items():
        expected = 0.0020 + 2 * 0.00015 + spread
        assert round_trip(DAY, market, at_open=at_open) == pytest.approx(expected, abs=1e-9)


def test_the_open_costs_more_than_the_rest_of_the_day() -> None:
    for market in ("거래소", "코스닥"):
        assert round_trip(DAY, market, at_open=True) > round_trip(DAY, market, at_open=False)


def test_kosdaq_costs_more_than_kospi() -> None:
    for at_open in (True, False):
        assert round_trip(DAY, "코스닥", at_open=at_open) > round_trip(
            DAY, "거래소", at_open=at_open
        )


def test_accepts_an_iso_string() -> None:
    assert round_trip("2026-09-15", "거래소") == round_trip(DAY, "거래소")


def test_unknown_market_falls_back_without_raising() -> None:
    assert round_trip(DAY, "코넥스") == pytest.approx(round_trip(DAY, "거래소"), abs=1e-9)
