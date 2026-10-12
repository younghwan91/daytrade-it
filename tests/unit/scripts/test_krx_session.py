"""krx_session: 2026-09-14 이후 일봉을 정규장 값으로 바꾸는 정문."""

import datetime as dt
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from krx_quant_core.market.limits import limit_up_price  # noqa: E402
from krx_session import AFTERMARKET_FROM, last_regular_closes, patch_regular  # noqa: E402

D = dt.date


def _daily() -> pd.DataFrame:
    # 005930 종가는 실제 값(09-17 일봉 256,000 vs 정규장 252,500), 고가·거래대금은 예시.
    return pd.DataFrame(
        [
            ("005930", D(2026, 9, 11), 263000, 261500, 258000, 259500, 1_000),
            ("005930", D(2026, 9, 16), 251000, 254000, 249000, 253500, 2_000),
            ("005930", D(2026, 9, 17), 255000, 259000, 251500, 256000, 3_000),
            ("000001", D(2026, 9, 17), 1000, 1100, 900, 1050, 10),  # 정규장 분봉 없음
        ],
        columns=["code", "date", "open", "high", "low", "close", "trade_value"],
    )


def _session() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("005930", D(2026, 9, 16), 254000.0, 249000.0, 253500.0, 1_900.0),
            ("005930", D(2026, 9, 17), 258500.0, 251500.0, 252500.0, 2_700.0),
        ],
        columns=["code", "date", "high", "low", "close", "trade_value"],
    )


def test_aftermarket_rows_take_regular_session_values_and_open_is_kept() -> None:
    out = patch_regular(_daily(), _session()).set_index(["code", "date"])
    r = out.loc[("005930", D(2026, 9, 17))]
    assert (r["close"], r["high"], r["trade_value"]) == (252500, 258500, 2_700)
    assert r["open"] == 255000 and r["regular"]


def test_rows_before_aftermarket_are_untouched() -> None:
    out = patch_regular(_daily(), _session()).set_index(["code", "date"])
    r = out.loc[("005930", D(2026, 9, 11))]
    assert (r["close"], r["high"], r["trade_value"]) == (259500, 261500, 1_000)
    assert r["regular"]
    assert D(2026, 9, 11) < AFTERMARKET_FROM


def test_missing_regular_bars_give_nan_not_the_contaminated_daily_value() -> None:
    out = patch_regular(_daily(), _session()).set_index(["code", "date"])
    r = out.loc[("000001", D(2026, 9, 17))]
    assert math.isnan(r["close"]) and math.isnan(r["high"]) and not r["regular"]
    assert r["open"] == 1000


def test_limit_up_base_is_the_regular_close() -> None:
    """09-18 상한가: 일봉 종가 256,000 기준이면 332,500, 정규장 252,500 기준이면 328,000."""
    out = patch_regular(_daily(), _session()).set_index(["code", "date"])
    base = out.loc[("005930", D(2026, 9, 17)), "close"]
    assert limit_up_price(base) == 328000
    assert limit_up_price(256000) == 332500


def test_last_regular_closes_skip_aftermarket_bars_and_the_day_itself() -> None:
    rows = [
        ("005930", dt.datetime(2026, 9, 17, 15, 30), 252500.0),
        ("005930", dt.datetime(2026, 9, 17, 19, 59), 256000.0),  # 애프터마켓
        ("005930", dt.datetime(2026, 9, 18, 9, 0), 255000.0),  # 당일
        ("000660", dt.datetime(2026, 9, 17, 15, 32), 1745000.0),  # VI 연장 종가 단일가
    ]
    assert last_regular_closes(rows, D(2026, 9, 18)) == {
        "005930": (252500.0, D(2026, 9, 17)),
        "000660": (1745000.0, D(2026, 9, 17)),
    }
