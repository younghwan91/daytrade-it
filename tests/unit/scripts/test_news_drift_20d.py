"""news_drift_20d.entry_day: 기사 이후 첫 정규장 시가가 서는 거래일."""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from news_drift_20d import entry_day  # noqa: E402

# 2026-10-02(금), 10-05(월), 10-06(화) 거래일. 10-03·04 주말.
DAYS = np.array(["2026-10-02", "2026-10-05", "2026-10-06"], dtype="datetime64[D]")


def _e(s):
    return entry_day(pd.Series(pd.to_datetime([s])), DAYS).iloc[0]


def test_premarket_same_day():
    assert _e("2026-10-05 08:59:59") == dt.date(2026, 10, 5)


def test_intraday_next_day():
    assert _e("2026-10-05 09:00:00") == dt.date(2026, 10, 6)


def test_after_close_next_day():
    assert _e("2026-10-02 18:30:00") == dt.date(2026, 10, 5)


def test_weekend_next_trading_day():
    assert _e("2026-10-04 13:00:00") == dt.date(2026, 10, 5)


def test_past_last_day_is_missing():
    assert pd.isna(_e("2026-10-06 10:00:00"))
