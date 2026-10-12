"""정규장(KRX) 일봉 정문 — 2026-09-14 부터 일봉에 섞인 KRX 애프터마켓을 걷어낸다.

2026-09-14 부터 KRX 애프터마켓(16:00~20:00 연속매매, ETF·ETN 제외) 체결이 키움 일봉 → ``daily_bars``·
``daily_bars_adjusted``·``supply_demand.close`` 의 고가·저가·종가·거래량(거래대금)과 ``minute_bars`` 16:00~19:59 봉에
섞였다. 시가만 정상이다. 예: 005930 2026-09-17 일봉 종가 256,000 vs 정규장 마지막 봉 252,500,
247540 2026-09-18 일봉 고가 108,600 vs 정규장 고가 105,600.

다음 날 가격제한폭(상·하한가)의 기준가는 전일 **정규장** 종가다. 연구 코드가 상한가·추격·이음매·전일 고가·거래대금을
일봉에서 바로 읽으면 2026-09-14 이후 날짜가 오염된다. 연구 스크립트는 일봉을 이 모듈로 읽는다:

- :func:`regular_daily` — ``daily_bars`` 를 읽되 :data:`AFTERMARKET_FROM` 이후 행의 고가·저가·종가·거래대금을
  ``minute_bars`` 정규장(:data:`REGULAR_OPEN` <= ts < :data:`REGULAR_END`) 집계로 바꾼다. 그날 정규장 분봉이 없으면
  그 열들은 NaN 이다(일봉 값으로 돌아가지 않는다). 검산(2026-10-12): 09-10~11 은 집계 종가·고가·저가가 일봉과
  99.9% 같고 거래대금 비 중앙값 0.999, 09-16~17 은 종가 일치 14% 뿐이다.
- :func:`last_regular_closes` — 분봉 행에서 종목별 직전 정규장 종가. 리플레이(``replay_parity`` R18, 가격제한 기준가)가
  같은 함수를 쓴다.

``tests/unit/scripts/test_research_price_gate.py`` 가 ``scripts/`` 에서 ``limit_up_price``·``limit_down_price`` 를
부르면서 ``daily_bars`` 를 직접 SELECT 하는 파일을 막는다.

``supply_demand`` 로 기준가를 만들 수도 있다: 그날 ``close / (1 + flu_rt / 1e4)`` 가 그날의 기준가(전일 정규장 종가)다
(``flu_rt`` 는 % × 100. 005930 09-17: 256,000 / 1.0099 = 253,490 ≈ 09-16 정규장 종가 253,500). 반올림 오차가 한 틱을
넘을 수 있어 여기서는 쓰지 않는다.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

#: 일봉에 KRX 애프터마켓 체결이 섞이기 시작한 첫 거래일.
AFTERMARKET_FROM = dt.date(2026, 9, 14)
#: 정규장 + 종가 단일가 봉: 보통 15:30, 변동성 완화장치로 연장되면 15:32
#: (09-01..09-15: 15:30 봉 2,895, 15:32 봉 22, 15:31/15:33/15:34 없음). 그 뒤 봉은 시간외·애프터마켓이다.
REGULAR_OPEN, REGULAR_END = dt.time(9, 0), dt.time(15, 33)
#: 정규장으로 다시 만드는 열. 시가는 오염되지 않았다.
PATCHED = ("high", "low", "close", "trade_value")


def last_regular_closes(
    rows: list[tuple[str, dt.datetime, float]], day: dt.date
) -> dict[str, tuple[float, dt.date]]:
    """``(code, ts, close)`` 분봉 행에서 종목별로 `day` 이전 마지막 정규장 봉의 (종가, 날짜). ts 는 KST naive."""
    best: dict[str, tuple[dt.datetime, float]] = {}
    for code, ts, close in rows:
        if ts.date() >= day or not (REGULAR_OPEN <= ts.time() < REGULAR_END):
            continue
        if code not in best or ts > best[code][0]:
            best[code] = (ts, close)
    return {c: (cl, ts.date()) for c, (ts, cl) in best.items()}


def regular_session_bars(con, first: dt.date, last: dt.date) -> pd.DataFrame:  # noqa: ANN001
    """``minute_bars`` 정규장 집계 (code, date, high, low, close, trade_value[백만원]), `first`..`last` 포함.

    ``ts`` 는 시간대 없는 KST 다 — ``at time zone`` 을 걸지 않는다.
    """
    q = (
        "SELECT code, ts::date AS date, max(high) AS high, min(low) AS low, "
        "(array_agg(close ORDER BY ts DESC))[1] AS close, sum(trade_value) / 1e6 AS trade_value "
        "FROM minute_bars WHERE ts >= %s AND ts < %s AND ts::time >= %s AND ts::time < %s "
        "GROUP BY code, ts::date"
    )
    params = (first, last + dt.timedelta(days=1), REGULAR_OPEN, REGULAR_END)
    out = pd.read_sql(q, con, params=params)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    for col in PATCHED:
        out[col] = out[col].astype(float)
    return out


def patch_regular(daily: pd.DataFrame, session: pd.DataFrame) -> pd.DataFrame:
    """`daily` 의 :data:`AFTERMARKET_FROM` 이후 행에서 :data:`PATCHED` 열을 `session` 값으로 바꾼 사본.

    `session` 에 없는 오염 행은 그 열들을 NaN 으로 둔다. 열 ``regular`` = 그 행의 값이 정규장 값인가.
    """
    d = daily.copy()
    for col in PATCHED:
        if col in d.columns:
            d[col] = d[col].astype(float)
    late = (d["date"] >= AFTERMARKET_FROM).to_numpy()
    d["regular"] = ~late
    if not late.any():
        return d
    s = session.set_index(["code", "date"])
    key = pd.MultiIndex.from_arrays([d.loc[late, "code"], d.loc[late, "date"]])
    sv = s.reindex(key)
    for col in PATCHED:
        if col in d.columns:
            d.loc[late, col] = sv[col].to_numpy()
    d.loc[late, "regular"] = np.isfinite(sv["close"].to_numpy(dtype=float))
    return d


def regular_daily(con, first: dt.date, last: dt.date) -> pd.DataFrame:  # noqa: ANN001
    """``daily_bars`` (code, date, open, high, low, close, trade_value[백만원], regular), 정규장 기준.

    가격제한 기준가 = 전일 행의 ``close``. ``regular`` 가 False 인 행은 고가·저가·종가·거래대금이 NaN 이다.
    """
    daily = pd.read_sql(
        "SELECT code, date, open, high, low, close, trade_value FROM daily_bars "
        "WHERE date >= %s AND date <= %s",
        con,
        params=(first, last),
    )
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    if last < AFTERMARKET_FROM:
        return patch_regular(daily, pd.DataFrame(columns=["code", "date", *PATCHED]))
    session = regular_session_bars(con, max(first, AFTERMARKET_FROM), last)
    return patch_regular(daily, session)
