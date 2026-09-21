#!/usr/bin/env python
"""수집한 Yahoo 60분봉을 parquet 로 굳히고, 빠진 장 마감 구간을 일봉에서 복원한다.

Yahoo 는 09:00~14:00 여섯 봉만 준다 — **15:00 이후가 통째로 없다**(종가 단일가
매매 포함. 그래서 Yahoo 일 거래량이 우리 일봉의 83% 다). ``daily_bars_adjusted``
에서 시간봉 합을 빼면 그 구간의 **종가는 정확히** 복원된다.

복원한 봉에 고가·저가는 **넣지 않는다(NaN)**. 15:00~15:30 이 시가·종가 밖으로
움직인 날이 압도적이라(실측 93%) 근사값을 넣으면 손절 백테스트가 에러 없이
조용히 틀린 답을 낸다. 없는 값은 없는 채로 둬야 쓰는 쪽이 걸려 넘어진다.

거래량도 정확하지 않다. 일봉 거래량은 분봉 합보다 평균 5~6% 크고(장외 대량매매
등), 잔여분에는 시간외 단일가까지 섞인다. ``volume`` 은 **15:00 이후 잔여
거래량의 상한 추정치**다 — 측정값은 README 에 박는다.

겹치는 구간(2025-08-01~)의 진짜 ``minute_bars`` 와 대조해서 전부 실측한다.
kr_quant 은 읽기 전용으로만 쓴다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

AUCTION_HOUR = 15  # 15:00 봉 = Yahoo 가 안 주는 15:00 이후 전부

DEFECTS = {
    "missing_close_session": (
        "Yahoo 는 09:00~14:00 여섯 봉만 준다. 15:00 봉(source=daily_recovered, "
        "synthetic=True)은 daily_bars_adjusted 에서 복원한 것이다. "
        "open 은 14:00 봉 종가, close 는 **그날의 환산비로 Yahoo 기준에 맞춘** "
        "일봉 종가, **high·low 는 NaN** — 근사값을 넣으면 손절 백테스트가 조용히 "
        "틀린다. volume 은 15:00 이후 잔여 거래량의 상한 추정치다(시간외·장외분 포함). "
        "환산해도 일봉 고저 범위를 벗어나거나 14:00 대비 15% 넘게 튀면 "
        "``recovery_ok=False`` 로 두고 close·volume 을 비운다."
    ),
    "recovery_ok": (
        "``recovery_ok=False`` 인 15:00 봉은 close·volume 이 NaN 이다. 복원이 "
        "말이 안 되는 날이라 그럴듯한 숫자를 넣는 대신 비웠다. Yahoo 원본 봉은 "
        "언제나 True 다(복원한 게 아니니까). **하루의 마지막 유효 종가를 쓰려면 "
        "NaN 을 확인하고 14:00 봉으로 물러나라** — dropna 없이 평균을 내면 "
        "그 날이 조용히 빠진다."
    ),
    "survivorship_bias": (
        "상장폐지 종목이 전혀 없다. Yahoo 는 폐지된 KRX 티커를 404 로 지운다"
        "(054340·101990 확인). 이 데이터로 잰 수익률에는 생존편향 상향 편의가 "
        "있고 고칠 방법이 없다. 수치를 인용할 때 반드시 같이 적어라."
    ),
    "adjusted_prices": (
        "수정주가다(무상증자·액면분할·배당 소급 반영). kr_quant minute_bars 는 "
        "원주가로 보인다. 한 백테스트에서 두 시리즈를 섞지 마라."
    ),
    "rolling_window": (
        "Yahoo interval=1h 는 730 거래일만 준다. 이 스냅샷보다 과거는 다시 못 받는다."
    ),
    "not_minute_bars": (
        "60분봉이다. minute_bars 테이블에 넣으면 안 된다 — 거기 쓰는 모든 쿼리가 "
        "'한 행 = 1분'을 가정하고 있어서, 에러 없이 조용히 틀린 답이 나온다."
    ),
    "dividend_adjusted": (
        "**Yahoo 는 배당까지 소급 반영하고, 우리 daily_bars_adjusted 는 안 한다.** "
        "전 종목 대조에서 드러났다 — 불연속 925건의 배당락일 쏠림이 1월 340건, "
        "6월 186건, 9월 188건으로 한국 배당락 시점과 정확히 겹치고, 계단 크기 "
        "중앙값이 2.4%(58%가 3% 미만)다. 나머지 큰 계단(p95 = 90%)은 액면분할이다. "
        "``adjustment_step=True`` 가 그 종목들이다(2,089 중 315개). "
        "**하루 안에서는 상수라 장중 분석에는 영향이 없다.** 배당락일을 가로지르는 "
        "수익률만 그만큼 틀어진다 — 여러 날 보유를 재려면 사건일에서 구간을 끊어라. "
        "추정 사건일은 _MANIFEST.json 의 adjustment_breaks 에 있다."
    ),
    "unverified_identity": (
        "``identity_verified=False`` 인 코드는 받아 온 봉이 정말 그 종목인지 일봉으로 "
        "확인하지 못한 것이다. 틀린 접미사가 404 를 주리라 믿으면 안 된다 — "
        "``247540.KS`` 는 728봉짜리 *다른 종목*을 조용히 돌려준다. 기본적으로 걸러라."
    ),
    "irregular_sessions": (
        "개장시간이 평소와 다른 날은 ``irregular_session=True`` 로 표시했다. "
        "목록을 들고 다니지 않고 데이터에서 찾아낸다 — 그날 전 종목의 최빈 "
        "(첫 봉, 마지막 봉, 봉 개수)이 (09시, 14시, 6개)와 다르면 이상으로 본다. "
        "예: 2025-11-13 수능일은 10:00 개장이라 첫 봉이 10시다. 이런 날은 15:00 "
        "복원봉이 실제 종가 시각과 어긋나므로 빼고 보라. "
        "규칙이 잡아내는 건 '평소와 다르다'까지고, 왜 다른지는 알려주지 않는다."
    ),
}


def load_raw(raw_dir: str) -> pd.DataFrame:
    # 900만 행짜리 JSONL 이라 pandas 의 json 파서로는 한참 걸린다.
    import pyarrow.json as paj

    path = os.path.join(raw_dir, "bars.jsonl")
    opts = paj.ParseOptions(
        explicit_schema=pa.schema(
            [
                ("code", pa.string()),
                ("ts", pa.string()),
                ("open", pa.float64()),
                ("high", pa.float64()),
                ("low", pa.float64()),
                ("close", pa.float64()),
                ("volume", pa.int64()),
                ("source", pa.string()),
            ]
        )
    )
    df = paj.read_json(path, parse_options=opts).to_pandas()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["ts"] = pd.to_datetime(df["ts"])
    before = len(df)
    df = df.drop_duplicates(subset=["code", "ts"]).sort_values(["code", "ts"])
    if before != len(df):
        print(f"  중복 {before - len(df):,}행 제거", file=sys.stderr)
    return df.reset_index(drop=True)


def read_sql(dsn: str, sql: str, params: tuple) -> pd.DataFrame:
    import psycopg2

    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


NORMAL_SESSION = (9, 14, 6)  # (첫 봉 시각, 마지막 봉 시각, 봉 개수) — 평소 장


def session_profile(bars: pd.DataFrame) -> pd.DataFrame:
    """날짜별 장 모양을 데이터에서 직접 읽는다 — 예외일 목록을 들고 다니지 않는다.

    그날 전 종목의 최빈 (첫 봉, 마지막 봉, 봉 개수)가 평소와 다르면 이상한 날이다.
    수능일(10시 개장)처럼 개장시간이 밀린 날이 여기 걸린다. 마지막 날짜는 장중에
    받았으면 15:00 봉이 있어 당연히 달라 보이므로 제외한다.
    """
    b = bars[~bars["synthetic"]].copy()
    b["date"] = b["ts"].dt.normalize()
    b["hour"] = b["ts"].dt.hour
    per = b.groupby(["date", "code"])["hour"].agg(["min", "max", "count"])
    prof = per.groupby("date").agg(lambda s: s.mode().iat[0] if len(s.mode()) else np.nan)
    prof.columns = ["first_hour", "last_hour", "bars"]
    prof["irregular"] = ~(
        (prof["first_hour"] == NORMAL_SESSION[0])
        & (prof["last_hour"] == NORMAL_SESSION[1])
        & (prof["bars"] == NORMAL_SESSION[2])
    )
    if len(prof):  # 오늘치는 아직 장이 안 끝났을 수 있다
        prof.loc[prof.index.max(), "irregular"] = False
    return prof.reset_index()


# 복원 게이트 — 하나라도 어기면 그 봉의 종가/거래량을 비운다(NaN)
MAX_CLOSE_MOVE = 0.15  # 15:00~15:30 에 이만큼 넘게 움직였다면 가격 기준이 어긋난 것이다
VOL_SHARE_RANGE = (0.3, 1.0)  # Yahoo 시간봉 합 / 일봉 거래량 이 이 밖이면 기준이 다르다


def recover_close_session(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Yahoo 가 안 주는 15:00 이후를 일봉에서 되살린다.

    **일봉 종가를 그대로 쓰면 안 된다.** Yahoo 는 배당까지 소급 반영하고
    ``daily_bars_adjusted`` 는 안 해서 둘이 다른 가격 기준 위에 있다. 그대로
    붙이면 14:00 봉과 15:00 봉 사이에 기준 차이만큼 계단이 생긴다 — 실측으로
    평균 |점프| 1,453bp, 심한 종목은 +573% 였다. 한 시간 만에 5배 오른 게
    아니라 자가 만든 흉터다.

    그래서 **그날의 환산비**(Yahoo 시가 / 일봉 시가)로 일봉 종가를 Yahoo 기준
    위로 옮긴 뒤에 붙인다. 비는 날마다 다시 재므로 배당락·분할이 언제 있었든
    저절로 따라간다.

    환산해도 말이 안 되는 날은 종가·거래량을 **비운다(NaN)**. 고저가와 같은
    이유다 — 그럴듯한 틀린 숫자는 조용히 틀리고, 빈 값은 쓰는 쪽에서 걸린다.
    """
    b = bars.copy()
    b["date"] = b["ts"].dt.normalize()
    sess = (
        b.groupby(["code", "date"])
        .agg(
            hourly_vol=("volume", "sum"),
            first_open=("open", "first"),
            last_close=("close", "last"),
            last_hour=("ts", "max"),
        )
        .reset_index()
    )

    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    for col in ("open", "high", "low", "close", "volume"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    m = sess.merge(d, on=["code", "date"], how="inner", suffixes=("", "_d"))
    m = m[m["last_hour"].dt.hour < AUCTION_HOUR]  # 이미 15:00 봉이 있는 날은 건드리지 않는다
    m = m.dropna(subset=["open", "close", "volume"])
    m = m[(m["open"] > 0) & (m["close"] > 0)]

    # 그날의 가격 기준 환산비. 시가끼리 비교한다 — 양쪽 다 같은 시가 단일가다.
    scale = (m["first_open"] / m["open"]).to_numpy()
    close = m["close"].to_numpy() * scale
    c14 = m["last_close"].to_numpy()
    hi = m["high"].to_numpy() * scale
    lo = m["low"].to_numpy() * scale

    ok = (
        np.isfinite(scale)
        & (scale > 0)
        & (close >= lo * (1 - 1e-9))
        & (close <= hi * (1 + 1e-9))
        & (np.abs(close / c14 - 1.0) <= MAX_CLOSE_MOVE)
    )

    vol_share = m["hourly_vol"].to_numpy() / m["volume"].to_numpy()
    vol_ok = ok & (vol_share >= VOL_SHARE_RANGE[0]) & (vol_share <= VOL_SHARE_RANGE[1])
    volume = np.where(
        vol_ok, np.clip(m["volume"].to_numpy() - m["hourly_vol"].to_numpy(), 0, None), np.nan
    )

    return pd.DataFrame(
        {
            "code": m["code"].to_numpy(),
            "ts": (m["date"] + pd.Timedelta(hours=AUCTION_HOUR)).to_numpy(),
            "open": c14,
            "high": np.nan,
            "low": np.nan,
            "close": np.where(ok, close, np.nan),
            "volume": pd.array(volume, dtype="Float64").astype("Int64"),
            "source": "daily_recovered",
            "synthetic": True,
            "recovery_ok": ok,
        }
    )


def seam_check(full: pd.DataFrame) -> dict:
    """14:00 봉과 15:00 복원봉 사이에 계단이 있는지 **산출물 자체로** 확인한다.

    앞선 판(2026-09-22 01차)은 일봉 종가를 환산 없이 붙여서 평균 |점프| 1,453bp,
    최악 +573% 짜리 흉터를 남겼다. 그걸 겹치는 구간 표본 검증은 못 잡았다 —
    최근 13개월은 두 기준이 대체로 같아서 통과해 버렸기 때문이다. 이 점검은
    표본이 아니라 **전 구간·전 종목**을 보고, 다음에 같은 실수를 하면 즉시 튄다.
    """
    d = full.copy()
    d["date"] = d["ts"].dt.normalize()
    c14 = d[(~d["synthetic"]) & (d["ts"].dt.hour == 14)][["code", "date", "close"]]
    c15 = d[d["synthetic"]][["code", "date", "close", "recovery_ok"]]
    j = c14.merge(c15, on=["code", "date"], suffixes=("_14", "_15")).dropna(
        subset=["close_14", "close_15"]
    )
    if j.empty:
        return {"pairs": 0}
    jump = (j["close_15"] / j["close_14"] - 1.0).abs()
    bad = jump > 0.02
    per_code = bad.groupby(j["code"]).mean()
    return {
        "pairs": int(len(j)),
        "mean_abs_jump_bp": round(float(jump.mean()) * 1e4, 1),
        "p99_abs_jump_bp": round(float(jump.quantile(0.99)) * 1e4, 1),
        "max_abs_jump_bp": round(float(jump.max()) * 1e4, 1),
        "pairs_over_2pct": int(bad.sum()),
        "codes_bad_over_half_of_days": int((per_code >= 0.5).sum()),
        "codes_bad_every_day": int((per_code >= 0.99).sum()),
    }


def validate_recovery(dsn: str, recovered: pd.DataFrame, sample: int) -> dict:
    """겹치는 구간의 진짜 분봉 15:00~장 끝과 대조한다 — 거기만 정답을 안다."""
    overlap = recovered[recovered["ts"] >= pd.Timestamp("2025-08-01")]
    if overlap.empty:
        return {"compared": 0, "note": "겹치는 구간 없음"}
    codes = sorted(overlap["code"].unique())[:sample]
    lo = pd.Timestamp(overlap["ts"].min()).strftime("%Y-%m-%d")
    truth = read_sql(
        dsn,
        "select code, ts::date d, "
        "       (array_agg(open order by ts))[1] open_true, "
        "       (array_agg(close order by ts desc) filter (where ts::time < '15:31'))[1] close_true, "
        "       sum(volume) vol_post15, "
        "       sum(volume) filter (where ts::time < '15:31') vol_auction, "
        "       max(ts::time) filter (where ts::time < '15:31') last_minute "
        "from minute_bars "
        "where code = any(%s) and ts >= %s and ts::time >= '15:00' "
        "group by 1, 2",
        (codes, lo),
    )
    if truth.empty:
        return {"compared": 0, "note": "분봉 쪽에 15:00 이후가 없음"}
    truth["code"] = truth["code"].astype(str)
    truth["ts"] = pd.to_datetime(truth["d"]) + pd.Timedelta(hours=AUCTION_HOUR)
    for col in ("open_true", "close_true", "vol_post15", "vol_auction"):
        truth[col] = pd.to_numeric(truth[col], errors="coerce")
    j = overlap.merge(truth, on=["code", "ts"])
    j = j.dropna(subset=["close_true", "open_true"])
    if j.empty:
        return {"compared": 0, "note": "겹치는 (종목,일자) 없음"}

    def frac_within(a: pd.Series, b: pd.Series, tol: float) -> float:
        rel = (a - b).abs() / b.abs().where(b.abs() > 0)
        return round(float(rel.le(tol).mean()), 4)

    def ratio_stats(num: pd.Series, den: pd.Series) -> dict:
        r = (num / den.where(den > 0)).replace([np.inf, -np.inf], np.nan).dropna()
        if r.empty:
            return {}
        return {
            "median": round(float(r.median()), 4),
            "p05": round(float(r.quantile(0.05)), 4),
            "p95": round(float(r.quantile(0.95)), 4),
        }

    # 우리 minute_bars 가 15:30 까지 안 찬 날은 '정답' 쪽이 틀린 것이라 따로 센다.
    clean = j[j["last_minute"].astype(str).ge("15:30")]
    return {
        "compared": int(len(j)),
        "codes": len(codes),
        "close_exact_frac": round(float((j["close"] - j["close_true"]).abs().lt(1e-6).mean()), 4),
        "close_within_0p1pct_frac": frac_within(j["close"], j["close_true"], 0.001),
        "open_within_0p5pct_frac": frac_within(j["open"], j["open_true"], 0.005),
        "volume_vs_post15": ratio_stats(j["volume"], j["vol_post15"]),
        "volume_vs_auction_only": ratio_stats(j["volume"], j["vol_auction"]),
        # minute_bars 가 15:30 까지 온전한 날만 — 복원 자체의 정확도
        "clean_days": int(len(clean)),
        "clean_close_exact_frac": (
            round(float((clean["close"] - clean["close_true"]).abs().lt(1e-6).mean()), 4)
            if len(clean)
            else None
        ),
    }


def adjustment_breaks(bars: pd.DataFrame, daily_open: pd.DataFrame) -> pd.DataFrame:
    """Yahoo 시리즈에 **수정주가 불연속**이 있는 종목을 전 종목에서 찾아낸다.

    수집기의 동일종목 판정은 최근 54거래일만 본다. 2024년에 액면분할한 종목은
    그 검사를 멀쩡히 통과하면서도 분할일 앞뒤로 가격이 5배 튀어 있을 수 있다 —
    그 구간을 가로지르는 수익률은 -80% 로 나온다. 조용히 틀리는 자리다.

    시가비(Yahoo 시가 / 우리 수정주가 시가)를 날짜별로 본다. 같은 종목이고
    기준이 같으면 이 비는 계속 1이다. 기준이 다르면 **계단**이 생기고, 계단이
    곧 사건 날짜다. 비가 한두 값에 몰리는지로 '같은 종목인지'와 '계단이 있는지'가
    동시에 갈린다.
    """
    b = bars[~bars["synthetic"]].copy()
    b["date"] = b["ts"].dt.normalize()
    y = b.groupby(["code", "date"])["open"].first().rename("y").reset_index()
    d = daily_open.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["open"] = pd.to_numeric(d["open"], errors="coerce")
    j = y.merge(d, on=["code", "date"])
    j = j[(j["open"] > 0) & (j["y"] > 0)]
    j["ratio"] = j["y"] / j["open"]

    out = []
    for code, g in j.groupby("code", sort=True):
        out.append(_classify_ratio(code, g.sort_values("date")))
    return pd.DataFrame(out)


# 비가 이만큼 안쪽이면 같은 값으로 본다 (호가단위 반올림·수정계수 반올림 흡수)
PLATEAU_TOL = 0.01


def _classify_ratio(code: str, g: pd.DataFrame) -> dict:
    """한 종목의 시가비 시계열을 평탄 구간으로 쪼갠다.

    값을 반올림해 세면 1.0 근처에 퍼진 잡음이 여러 값으로 갈라져 멀쩡한 종목이
    '계단 있음'으로 찍힌다. 그래서 **중앙값 대비 상대거리**로 묶는다.
    """
    if len(g) < 20:
        return {"code": code, "days": len(g), "verdict": "unverifiable"}
    r = g["ratio"].to_numpy()
    med = float(np.median(r))
    near = float(np.mean(np.abs(r / med - 1.0) <= PLATEAU_TOL))
    rec = {
        "code": code,
        "days": int(len(g)),
        "main_ratio": round(med, 4),
        "share_at_main": round(near, 3),
    }
    if near >= 0.98:
        rec["verdict"] = "identical"
        return rec
    # 남은 날들이 다시 한두 개 평탄 구간을 이루는지 본다
    levels = [med]
    rest = r[np.abs(r / med - 1.0) > PLATEAU_TOL]
    for _ in range(2):
        if not len(rest):
            break
        m2 = float(np.median(rest))
        levels.append(m2)
        rest = rest[np.abs(rest / m2 - 1.0) > PLATEAU_TOL]
    covered = 1.0 - (len(rest) / len(r))
    rec["plateaus"] = [round(x, 4) for x in levels]
    rec["covered"] = round(covered, 3)
    if covered < 0.98:
        rec["verdict"] = "unresolved"
        return rec
    rec["verdict"] = "adjustment_step"
    # 평탄 구간이 바뀌는 자리 = 사건일. 날짜순으로 라벨이 바뀌는 지점을 잡는다.
    lab = np.array([int(np.argmin([abs(x / lv - 1.0) for lv in levels])) for x in r])
    dates = g["date"].to_numpy()
    rec["breaks"] = [
        {
            "between": str(pd.Timestamp(dates[i - 1]).date()),
            "and": str(pd.Timestamp(dates[i]).date()),
            "ratio_from": round(float(levels[lab[i - 1]]), 4),
            "ratio_to": round(float(levels[lab[i]]), 4),
        }
        for i in range(1, len(lab))
        if lab[i] != lab[i - 1]
    ][:5]
    return rec


def classify_suspects(suspects: list, bars: pd.DataFrame, daily_close: pd.DataFrame) -> list:
    """의심 종목이 '다른 종목'인지 '수정계수만 다른 같은 종목'인지 가른다.

    수집기의 판정(일봉 고저 범위 포함)은 같은 종목이라도 수정주가 기준이 다르면
    통째로 떨어진다 — 인적분할 같은 걸 겪은 종목이 그렇다. 그런데 그건 위험하지
    않고, **다른 종목이 딸려 온 것**만 위험하다. 둘은 일별 가격비로 갈린다:
    같은 종목이면 비가 거의 일정하고(변동계수 작음), 다른 종목이면 제멋대로다.
    """
    if not suspects:
        return []
    b = bars[~bars["synthetic"]].copy()
    b["date"] = b["ts"].dt.normalize()
    last = b.groupby(["code", "date"])["close"].last().rename("y_close").reset_index()
    ref = daily_close.copy()
    ref["date"] = pd.to_datetime(ref["date"])
    ref["close"] = pd.to_numeric(ref["close"], errors="coerce")
    j = last.merge(ref, on=["code", "date"])
    out = []
    for s in suspects:
        g = j[j["code"] == s["code"]]
        r = (
            (g["y_close"] / g["close"].where(g["close"] > 0))
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        rec = dict(s)
        rec["ratio_days"] = int(len(r))
        if len(r) < 20:
            rec["verdict"] = "unverifiable"  # 대조할 일봉이 없다
        else:
            cv = float(r.std() / abs(r.mean())) if r.mean() else float("inf")
            rec["ratio_median"] = round(float(r.median()), 4)
            rec["ratio_cv"] = round(cv, 4)
            rec["verdict"] = "scale_mismatch" if cv < 0.02 else "likely_wrong_instrument"
        out.append(rec)
    return out


def collect_fetch_log(raw_dir: str) -> tuple[list, list, list]:
    fallbacks, failures, suspects = [], [], []
    with open(os.path.join(raw_dir, "done.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("suffix_fallback"):
                fallbacks.append(
                    {
                        "code": r["code"],
                        "stocks_market": r["market"],
                        "actual_suffix": r["suffix"],
                        "match_score": r.get("match_score"),
                    }
                )
            if not r.get("bars"):
                failures.append({"code": r["code"], "market": r["market"], "error": r.get("error")})
            elif r.get("suspect"):
                suspects.append(
                    {
                        "code": r["code"],
                        "market": r["market"],
                        "suffix": r["suffix"],
                        "bars": r["bars"],
                        "match_score": r.get("match_score"),
                        "checked_days": r.get("checked_days"),
                    }
                )
    return fallbacks, failures, suspects


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Yahoo 60분봉 → parquet + 마감 구간 복원")
    ap.add_argument("--out", default="/home/young/git/daytrade-it/data/yahoo_hourly")
    ap.add_argument("--dsn", default=os.environ.get("KR_QUANT_DB"))
    ap.add_argument("--validate-codes", type=int, default=60)
    args = ap.parse_args(argv)
    if not args.dsn:
        print("KR_QUANT_DB 가 없다.", file=sys.stderr)
        return 2

    raw_dir = os.path.join(args.out, "raw")
    print("원본 읽는 중…", file=sys.stderr)
    bars = load_raw(raw_dir)
    bars["synthetic"] = False
    codes = sorted(bars["code"].unique())
    lo = bars["ts"].min().strftime("%Y-%m-%d")
    hi = bars["ts"].max().strftime("%Y-%m-%d")
    print(f"  {len(bars):,}행 · {len(codes):,}종목 · {lo} ~ {hi}", file=sys.stderr)

    print("장 모양 점검 중…", file=sys.stderr)
    prof = session_profile(bars)
    odd = prof[prof["irregular"]]
    print(f"  이상한 날 {len(odd)}/{len(prof)}일", file=sys.stderr)

    print("일봉으로 마감 구간 복원 중…", file=sys.stderr)
    daily = read_sql(
        args.dsn,
        "select code, date, open, high, low, close, volume from daily_bars_adjusted "
        "where date between %s and %s and code = any(%s)",
        (lo, hi, codes),
    )
    daily["code"] = daily["code"].astype(str)

    print("수정주가 불연속 점검 중(전 종목)…", file=sys.stderr)
    breaks = adjustment_breaks(bars, daily[["code", "date", "open"]])
    bcount = breaks["verdict"].value_counts().to_dict()
    print(f"  {bcount}", file=sys.stderr)
    recovered = recover_close_session(bars, daily)
    print(f"  복원 {len(recovered):,}봉", file=sys.stderr)

    print("복원 검증(겹치는 구간의 진짜 분봉과 대조) 중…", file=sys.stderr)
    report = validate_recovery(args.dsn, recovered, args.validate_codes)
    print("  " + json.dumps(report, ensure_ascii=False), file=sys.stderr)

    fallbacks, failures, suspects = collect_fetch_log(raw_dir)
    suspects = classify_suspects(suspects, bars, daily[["code", "date", "close"]])
    if suspects:
        verdicts = pd.Series([s["verdict"] for s in suspects]).value_counts().to_dict()
        print(f"  의심 {len(suspects)}종목 분류: {verdicts}", file=sys.stderr)

    real = bars.drop(columns=["date"], errors="ignore").copy()
    real["recovery_ok"] = True  # Yahoo 원본은 복원한 게 아니라 받은 그대로다
    real["volume"] = real["volume"].astype("Int64")
    full = pd.concat([real, recovered], ignore_index=True)
    full = full.sort_values(["code", "ts"]).reset_index(drop=True)
    full["year"] = full["ts"].dt.year
    full["irregular_session"] = (
        full["ts"]
        .dt.normalize()
        .map(prof.set_index("date")["irregular"])
        .fillna(False)
        .astype(bool)
    )
    # 같은 종목임을 일봉으로 확인하지 못한 코드 — 쓰는 쪽이 골라낼 수 있게 표시한다
    step_codes = set(breaks.loc[breaks["verdict"] == "adjustment_step", "code"])
    bad_codes = set(breaks.loc[breaks["verdict"] == "unresolved", "code"])
    unknown = set(breaks.loc[breaks["verdict"] == "unverifiable", "code"])
    unverified = (
        bad_codes
        | unknown
        | {s["code"] for s in suspects if s["verdict"] != "scale_mismatch"} - step_codes
    )
    full["identity_verified"] = ~full["code"].isin(unverified)
    # 수정주가 계단이 있는 종목 — kr_quant 일봉과 섞을 때만 문제가 된다
    full["adjustment_step"] = full["code"].isin(step_codes)

    seam = seam_check(full)
    print(f"  이음매 점검: {json.dumps(seam, ensure_ascii=False)}", file=sys.stderr)
    n_bad = int((full["synthetic"] & ~full["recovery_ok"]).sum())
    print(
        f"  복원 실패(종가 비움) {n_bad:,}봉 · "
        f"거래량 비움 {int(full['synthetic'].sum() - full.loc[full['synthetic'], 'volume'].notna().sum()):,}봉",
        file=sys.stderr,
    )

    manifest = {
        "dataset": "yahoo_hourly",
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "Yahoo Finance chart API, interval=1h, range=730d",
        "resolution": "60분",
        "rows": int(len(full)),
        "codes": int(full["code"].nunique()),
        "ts_min": str(full["ts"].min()),
        "ts_max": str(full["ts"].max()),
        "synthetic_rows": int(full["synthetic"].sum()),
        "trading_days": int(len(prof)),
        "irregular_sessions": [
            {
                "date": r["date"].strftime("%Y-%m-%d"),
                "first_hour": int(r["first_hour"]),
                "last_hour": int(r["last_hour"]),
                "bars": int(r["bars"]),
            }
            for _, r in odd.iterrows()
        ],
        "normal_session": {
            "first_hour": NORMAL_SESSION[0],
            "last_hour": NORMAL_SESSION[1],
            "bars": NORMAL_SESSION[2],
        },
        "seam_check": seam,
        "recovery_failed_bars": n_bad,
        "adjustment_break_summary": bcount,
        "adjustment_breaks": breaks[
            breaks["verdict"].isin(["adjustment_step", "unresolved"])
        ].to_dict("records"),
        "recovery_validation": report,
        "suffix_fallback_codes": fallbacks,
        "suspect_codes": suspects,
        "no_data_codes": failures,
        "defects": DEFECTS,
    }

    bars_dir = os.path.join(args.out, "bars")
    table = pa.Table.from_pandas(full, preserve_index=False)
    table = table.replace_schema_metadata(
        {
            **(table.schema.metadata or {}),
            b"daytrade_it_manifest": json.dumps(manifest, ensure_ascii=False).encode(),
        }
    )
    pq.write_to_dataset(
        table, root_path=bars_dir, partition_cols=["year"], existing_data_behavior="delete_matching"
    )

    with open(os.path.join(args.out, "_MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    write_readme(args.out, manifest)
    print(f"완료 · {len(full):,}행 → {bars_dir}", file=sys.stderr)
    return 0


def write_readme(out: str, m: dict) -> None:
    lines = [
        "# yahoo_hourly — KRX 60분봉 (2023-09~)",
        "",
        f"생성 {m['built_at']} · {m['rows']:,}행 · {m['codes']:,}종목 · "
        f"{m['ts_min']} ~ {m['ts_max']} · 복원분 {m['synthetic_rows']:,}행",
        "",
        "키움 `ka10080` 은 2025-09-01 이전을 안 준다(봉 크기를 키워도 같다 — 서버 보존",
        "기간이다). 그 이전 장중 가격은 이 데이터셋이 유일하다.",
        "",
        "## 쓰는 법",
        "",
        "```python",
        "import pandas as pd",
        f"df = pd.read_parquet('{os.path.join(out, 'bars')}')",
        "```",
        "",
        "`year` 로 파티션돼 있다. `source='yahoo_1h'` 는 Yahoo 원본(09:00~14:00),",
        "`source='daily_recovered'` 는 일봉에서 복원한 15:00 봉(`synthetic=True`)이다.",
        "`irregular_session=True` 는 개장시간이 평소와 달랐던 날이다 — 보통 걸러라.",
        "`identity_verified=False` 는 일봉 대조로 같은 종목임을 확인하지 못한 코드다.",
        "`adjustment_step=True` 는 우리 일봉과 수정 기준이 다른 종목이다 — 아래를 읽어라.",
        "`recovery_ok=False` 인 15:00 봉은 close·volume 이 NaN 이다.",
        "",
        "```python",
        "# 권장 기본 필터",
        "ok = df.identity_verified & ~df.irregular_session & df.recovery_ok",
        "```",
        "",
        "## 이음매 점검 (14:00 → 15:00)",
        "",
        "복원봉이 앞 봉과 같은 가격 기준 위에 있는지 **전 종목·전 구간**으로 잰다.",
        "표본 검증은 이걸 못 잡았다 — 최근 13개월은 두 기준이 대체로 같아서다.",
        "",
        "```json",
        json.dumps(m["seam_check"], ensure_ascii=False, indent=1),
        "```",
        "",
        "## 결함 — 인용하기 전에 읽어라",
        "",
    ]
    for k, text in m["defects"].items():
        lines += [f"### {k}", "", text, ""]
    lines += [
        "## 복원 검증 (실측)",
        "",
        "겹치는 구간(2025-08-01~)의 진짜 `minute_bars` 15:00 이후와 대조한 결과:",
        "",
        "```json",
        json.dumps(m["recovery_validation"], ensure_ascii=False, indent=2),
        "```",
        "",
        "`volume_vs_post15` 가 1 보다 크면 복원 거래량이 실제보다 크다는 뜻이다",
        "(일봉 거래량이 분봉 합보다 평균 5~6% 크고, 장외·시간외분이 섞인다).",
        "거래량을 임계값으로 쓰는 로직에 이 봉을 넣지 마라.",
        "",
        "## 수정주가 불연속 (전 종목 점검)",
        "",
        f"판정 분포: `{json.dumps(m['adjustment_break_summary'], ensure_ascii=False)}`",
        "",
        "Yahoo 시가와 우리 `daily_bars_adjusted` 시가의 비를 날짜별로 봤다 — 같은",
        "종목이고 수정 기준이 같으면 계속 1이다. **수집기의 동일종목 검사는 최근",
        "54거래일만 보기 때문에, 2024년에 분할한 종목은 검사를 통과하면서도 그 앞",
        "구간이 배수만큼 틀어져 있다.** 그래서 전 종목·전 구간을 따로 훑었다.",
        "",
        "계단 925건의 달 분포가 1월 340 · 6월 186 · 9월 188 로 **한국 배당락일에**",
        "**정확히 몰린다.** 계단 크기도 중앙값 2.4%, 58%가 3% 미만이다. 즉 대부분은",
        "종목이 바뀐 게 아니라 **Yahoo 가 배당을 소급 반영하고 우리는 안 하는 차이**다.",
        "장중 분석에는 영향이 없고(하루 안에서는 상수), 배당락일을 가로지르는 수익률만",
        "틀어진다.",
        "",
        "- `identical`: 비가 계속 1. 안전하다.",
        "- `adjustment_step`: 계단이 있다. 대개 배당, 큰 것(p95 = 90%)은 액면분할.",
        "  `adjustment_step=True` 로 표시했고 사건일은 아래 `breaks` 에 있다.",
        "- `unresolved`: 계단으로도 설명이 안 된다. `identity_verified=False`.",
        "- `unverifiable`: 대조할 일봉이 20일 미만. `identity_verified=False`.",
        "",
        "```json",
        json.dumps(m["adjustment_breaks"][:80], ensure_ascii=False, indent=1),
        "```",
        "",
        f"## 개장시간이 달랐던 날 ({len(m['irregular_sessions'])}/{m['trading_days']}일)",
        "",
        f"평소 장 = 첫 봉 {m['normal_session']['first_hour']}시 · "
        f"마지막 봉 {m['normal_session']['last_hour']}시 · "
        f"{m['normal_session']['bars']}봉. 여기서 벗어난 날을 데이터에서 찾아 "
        "`irregular_session=True` 로 표시했다(목록을 사람이 관리하지 않는다).",
        "",
        "```json",
        json.dumps(m["irregular_sessions"], ensure_ascii=False),
        "```",
        "",
    ]
    if m["suffix_fallback_codes"]:
        lines += [
            "## `stocks.market` 이 틀린 종목",
            "",
            "아래 종목은 `stocks.market` 이 가리키는 시장의 Yahoo 접미사로는 일봉 대조를",
            "통과하지 못하고 반대쪽으로 통과했다. **`stocks` 테이블의 데이터 결함**이라",
            "다른 데서도 문제를 낸다.",
            "",
            "| code | stocks.market | 실제 | 일치율 |",
            "|---|---|---|---|",
        ]
        for f in m["suffix_fallback_codes"]:
            lines.append(
                f"| {f['code']} | {f['stocks_market']} | .{f['actual_suffix']} "
                f"| {f['match_score']} |"
            )
        lines.append("")
    if m["suspect_codes"]:
        lines += [
            f"## 의심 종목 ({len(m['suspect_codes'])}개)",
            "",
            "어느 접미사로도 일봉 고저 대조를 통과하지 못한 코드다. 다만 통과 못 한 이유가",
            "둘이라 일별 가격비로 갈랐다 — 같은 종목이면 비가 거의 일정하다.",
            "",
            "- `scale_mismatch`: 같은 종목인데 수정주가 기준이 다르다(비가 일정). 쓸 수 있다.",
            "- `likely_wrong_instrument`: 비가 제멋대로다. **다른 종목일 수 있다.**",
            "- `unverifiable`: 대조할 일봉이 없다(최근 일봉 20일 미만).",
            "",
            "뒤의 둘은 `identity_verified=False` 로 표시돼 있다. 걸러 쓰라.",
            "",
            "```json",
            json.dumps(m["suspect_codes"], ensure_ascii=False, indent=1),
            "```",
            "",
        ]
    if m["no_data_codes"]:
        lines += [
            f"## Yahoo 에 없는 종목 ({len(m['no_data_codes'])}개)",
            "",
            "```json",
            json.dumps(m["no_data_codes"], ensure_ascii=False),
            "```",
            "",
        ]
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
