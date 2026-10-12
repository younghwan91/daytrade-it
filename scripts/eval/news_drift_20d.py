"""뉴스 종목의 5·20거래일 초과수익과 진입 시점 지표 — 사전등록 측정.

    docs/research/frozen/2026-10-09-news-drift-20d-prereg.json

    ~/git/daytrade-it/.venv/bin/python scripts/eval/news_drift_20d.py --period explore
    ~/git/daytrade-it/.venv/bin/python scripts/eval/news_drift_20d.py --period confirm --features a,b

탐색(explore)은 지표 전체를, 확인(confirm)은 --features 로 넘긴 후보만 잰다. 확인 구간 이벤트는
--period confirm 일 때만 계산된다. LLM 판정은 쓰지 않는다. DB 는 읽기 전용.

오염 경고: 가격은 ``daily_bars_adjusted`` 다. 2026-09-14 부터 그 종가·고가·저가·거래대금에 KRX 애프터마켓
(16:00~20:00) 체결이 섞였다. 확인 구간 끝(2026-08-31) 근처 진입의 20일 지평은 그 날을 넘는다 —
:func:`warn_aftermarket_closes` 가 실행 때 행 수를 알린다. 2026-09-14 이후 날짜로 구간을 늘리지 않는다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from krx_quant_core.stats.matched_null import matched_control
from krx_quant_core.stats.metrics import newey_west_t
from krx_session import AFTERMARKET_FROM
from rr_entry_filter import db_url, log

REPO = Path(__file__).resolve().parents[2]
PREREG = REPO / "docs/research/frozen/2026-10-09-news-drift-20d-prereg.json"
ARCHIVE = REPO.parent / "daytrade-it/data/toss_universe_42m"
PERIODS = {
    "explore": (dt.date(2023, 3, 1), dt.date(2024, 12, 31)),
    "confirm": (dt.date(2025, 1, 1), dt.date(2026, 8, 31)),
}
UNIVERSE_TOP = 1000
N_PER = 5
SEED = 0
HORIZONS = {"r5": 5, "r20": 20}
PRICE_MOVE_RE = re.compile(r"상한가|급등|급락|신고가|\d+(\.\d+)?% ?(상승|하락|↑|↓)|주가 [+-]?\d")
FEATURES = [
    "n_today",
    "n_prev5",
    "first_in_20",
    "n_sources",
    "tag_special",
    "tag_exclusive",
    "price_move_title",
    "ret_prev1",
    "ret_prev20",
    "tv_ratio_prev1",
    "range_pos_prev1",
    "gap",
    "size_rank",
]
BINARY = {"first_in_20", "tag_special", "tag_exclusive", "price_move_title"}


# ---------------------------------------------------------------- 가격


def load_prices(con, first: dt.date, last: dt.date) -> pd.DataFrame:
    px = pd.read_sql(
        "select a.code, a.date, a.open, a.high, a.low, a.close, a.trade_value "
        "from daily_bars_adjusted a join stocks s on s.code = a.code and s.kind = 'A' "
        "where a.date >= %s and a.date <= %s",
        con,
        params=(first - dt.timedelta(days=60), last + dt.timedelta(days=45)),
    )
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    g = px.groupby("code")
    c1, c2 = g["close"].shift(1), g["close"].shift(2)
    px["ret_prev1"] = c1 / c2 - 1
    px["ret_prev20"] = c1 / g["close"].shift(21) - 1
    tv20 = g["trade_value"].transform(lambda s: s.shift(1).rolling(20, min_periods=15).mean())
    px["tv20"] = tv20
    px["tv_ratio_prev1"] = g["trade_value"].shift(1) / tv20
    h1, l1 = g["high"].shift(1), g["low"].shift(1)
    px["range_pos_prev1"] = ((c1 - l1) / (h1 - l1)).where(h1 > l1)
    px["gap"] = px["open"] / c1 - 1
    for name, h in HORIZONS.items():
        px[name] = g["close"].shift(-(h - 1)) / px["open"] - 1
    warn_aftermarket_closes(px, g, first, last)
    px["size_rank"] = px.groupby("date")["tv20"].rank(ascending=False)
    return px


def warn_aftermarket_closes(px: pd.DataFrame, g, first: dt.date, last: dt.date) -> int:  # noqa: ANN001
    """오염 경고: 지평 종가가 :data:`AFTERMARKET_FROM` 이후인 진입 행 수를 알린다.

    2026-09-14 부터 ``daily_bars_adjusted`` 종가에 KRX 애프터마켓(16:00~20:00) 체결이 섞였다. 이 측정기는
    수정주가 수익률이 필요해 ``krx_session.regular_daily``(원 ``daily_bars``)로 바꾸지 않았고, 확인 구간
    (~2026-08-31) 은 이미 이 코드로 채점됐다(2026-10-09). 지평이 그 날을 넘는 행은 정규장 종가가 아닌 값으로
    수익률을 잰다 — 그 날짜 이후로 구간을 늘려 다시 돌리지 않는다.
    """
    date = pd.to_datetime(px["date"])
    in_window = (date >= pd.Timestamp(first)) & (date <= pd.Timestamp(last))
    n = 0
    for name, h in HORIZONS.items():
        exit_date = pd.to_datetime(g["date"].shift(-(h - 1)))
        bad = in_window & px[name].notna() & (exit_date >= pd.Timestamp(AFTERMARKET_FROM))
        n = max(n, int(bad.sum()))
    if n:
        log(
            f"경고: 지평 종가가 {AFTERMARKET_FROM} 이후(애프터마켓 포함 종가)인 진입 행 최대 {n:,} 개 "
            "— 그 수익률은 정규장 기준이 아니다"
        )
    return n


# ---------------------------------------------------------------- 뉴스


def load_articles() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(ARCHIVE / "*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                rows.append(
                    (d["code"], d["createdAt"], d.get("title") or "", d.get("source") or "")
                )
    a = pd.DataFrame(rows, columns=["code", "ts", "title", "source"]).drop_duplicates()
    a["ts"] = pd.to_datetime(a["ts"])
    return a


def entry_day(ts: pd.Series, days: np.ndarray) -> pd.Series:
    """기사 이후 첫 정규장 시가가 서는 거래일. 장전(09:00 전) 기사는 그날, 이후는 다음 거래일."""
    d = ts.dt.normalize().to_numpy(dtype="datetime64[D]")
    before_open = (ts.dt.hour < 9).to_numpy()
    pos = np.searchsorted(days, d, side="left")  # d 이상 첫 거래일
    is_td = (pos < len(days)) & (days[np.minimum(pos, len(days) - 1)] == d)
    pos = np.where(is_td & ~before_open, pos + 1, pos)
    out = np.full(len(d), np.datetime64("NaT", "D"), dtype="datetime64[D]")
    ok = pos < len(days)
    out[ok] = days[pos[ok]]
    return pd.Series(pd.to_datetime(out).date, index=ts.index)


def build_events(a: pd.DataFrame, days: np.ndarray) -> pd.DataFrame:
    a = a.copy()
    a["e"] = entry_day(a["ts"], days)
    a = a.dropna(subset=["e"])
    a["special"] = a["title"].str.contains(r"\[특징주\]", regex=True)
    a["exclusive"] = a["title"].str.contains(r"\[단독\]", regex=True)
    a["pmove"] = a["title"].str.contains(PRICE_MOVE_RE)
    ev = a.groupby(["code", "e"]).agg(
        n_today=("title", "size"),
        n_sources=("source", "nunique"),
        tag_special=("special", "max"),
        tag_exclusive=("exclusive", "max"),
        price_move_title=("pmove", "max"),
    )
    ev = ev.reset_index()
    for c in ("tag_special", "tag_exclusive", "price_move_title"):
        ev[c] = ev[c].astype(int)
    # 거래일 인덱스로 직전 5·20일 기사 수, 보유창 (e, e+19] 후속 기사 수
    idx = {d: i for i, d in enumerate(pd.to_datetime(days).date)}
    ev["ei"] = ev["e"].map(idx)
    n_by = {}
    for code, g in ev.groupby("code"):
        n_by[code] = dict(zip(g["ei"], g["n_today"], strict=True))
    prev5, prev20, follow = [], [], []
    for code, ei in zip(ev["code"], ev["ei"], strict=True):
        m = n_by[code]
        prev5.append(sum(m.get(ei - k, 0) for k in range(1, 6)))
        prev20.append(sum(m.get(ei - k, 0) for k in range(1, 21)))
        follow.append(sum(m.get(ei + k, 0) for k in range(1, 20)))
    ev["n_prev5"] = prev5
    ev["n_prev20"] = prev20
    ev["first_in_20"] = (ev["n_prev20"] == 0).astype(int)
    ev["followups"] = follow
    return ev, n_by, idx


# ---------------------------------------------------------------- 통계


def daily_nw(df: pd.DataFrame, col: str, lag: int) -> dict:
    s = df.groupby("e")[col].mean().sort_index()
    mu, t = newey_west_t(s.to_numpy(), lag)
    return {"mean_bp": mu * 1e4, "nw_t": t, "days": int(len(s)), "n": int(df[col].notna().sum())}


def spread_nw(df: pd.DataFrame, col: str, hi_mask, lo_mask, lag: int) -> dict:
    hi = df[hi_mask].groupby("e")[col].mean()
    lo = df[lo_mask].groupby("e")[col].mean()
    s = (hi - lo).dropna().sort_index()
    mu, t = newey_west_t(s.to_numpy(), lag)
    return {"spread_bp": mu * 1e4, "nw_t": t, "days": int(len(s))}


def feature_table(ev: pd.DataFrame, f: str, target: str, lag: int) -> dict:
    d = ev.dropna(subset=[f, target]).copy()
    if f in BINARY:
        g = d.groupby(f)[target].agg(["mean", "size"])
        res = {
            "bins": {str(int(k)): [float(r["mean"] * 1e4), int(r["size"])] for k, r in g.iterrows()}
        }
        res.update(spread_nw(d, target, d[f] == 1, d[f] == 0, lag))
        res["monotone"] = None
        return res
    pct = d.groupby("e")[f].rank(pct=True, method="average")
    d["q"] = np.minimum((pct * 5).apply(np.ceil).astype(int), 5)
    g = d.groupby("q")[target].agg(["mean", "size"])
    means = [g.loc[q, "mean"] if q in g.index else np.nan for q in range(1, 6)]
    diffs = np.sign(np.diff(means))
    mono = int(max((diffs > 0).sum(), (diffs < 0).sum()))
    res = {
        "bins": {str(q): [float(g.loc[q, "mean"] * 1e4), int(g.loc[q, "size"])] for q in g.index}
    }
    res.update(spread_nw(d, target, d["q"] == 5, d["q"] == 1, lag))
    res["monotone"] = mono
    return res


# ---------------------------------------------------------------- 실행


def run(args) -> dict:
    import psycopg2

    lo, hi = PERIODS[args.period]
    con = psycopg2.connect(db_url(Path(args.db_env).expanduser()))
    px = load_prices(con, lo, hi)
    days = np.array(sorted(px["date"].unique()), dtype="datetime64[D]")
    log(f"가격 {len(px):,} 행, 거래일 {len(days)}")
    a = load_articles()
    log(f"기사 {len(a):,}")
    ev, n_by, idx = build_events(a, days)
    ev = ev[(ev["e"] >= lo) & (ev["e"] <= hi)]

    keep = [
        "code",
        "date",
        "open",
        *HORIZONS,
        "ret_prev1",
        "ret_prev20",
        "tv_ratio_prev1",
        "range_pos_prev1",
        "gap",
        "size_rank",
        "tv20",
    ]
    p = px[keep][(px["date"] >= lo) & (px["date"] <= hi)].copy()
    p = p[p["size_rank"] <= UNIVERSE_TOP]
    ev = ev.merge(p, left_on=["code", "e"], right_on=["code", "date"], how="inner").drop(
        columns="date"
    )
    log(f"이벤트(universe 안) {len(ev):,}, {ev['e'].nunique()}일")

    # 매치드 널: 그날·직전 20일 기사가 없는 universe 종목
    p["ei"] = p["date"].map(idx)
    has_news = np.array(
        [
            any(n_by.get(c, {}).get(i - k, 0) for k in range(0, 21))
            for c, i in zip(p["code"], p["ei"], strict=True)
        ]
    )
    for col, q in (("tv20", "q_tv"), ("ret_prev20", "q_r20"), ("ret_prev1", "q_r1")):
        p[q] = p.groupby("date")[col].transform(
            lambda s: pd.qcut(s.rank(method="first"), 4, labels=False)
        )
    pool = p[~has_news].rename(columns={"date": "e"}).reset_index(drop=True)
    pool.index = pool.index + 50_000_000
    ev = ev.merge(
        p[["code", "date", "q_tv", "q_r20", "q_r1"]],
        left_on=["code", "e"],
        right_on=["code", "date"],
        how="left",
    ).drop(columns="date")
    ctl = matched_control(
        ev, pool, strata=["e", "q_tv", "q_r20", "q_r1"], n_per=N_PER, seed=SEED, exclude_self=False
    )
    for name in HORIZONS:
        cm = (
            pd.Series(pool.loc[ctl["control_idx"], name].to_numpy(), index=ctl["trade_idx"])
            .groupby(level=0)
            .mean()
        )
        ev[f"x_{name}"] = ev[name] - cm.reindex(ev.index)
    log(f"널 풀 {len(pool):,}, 대조 {len(ctl):,}, 미매치 {len(ctl.attrs['unmatched']):,}")

    res = {
        "period": args.period,
        "range": [str(lo), str(hi)],
        "n_events": int(len(ev)),
        "days": int(ev["e"].nunique()),
    }
    res["base"] = {
        name: {"raw": daily_nw(ev, name, h), "excess": daily_nw(ev, f"x_{name}", h)}
        for name, h in HORIZONS.items()
    }
    feats = FEATURES if args.period == "explore" else [f for f in args.features.split(",") if f]
    res["features"] = {f: feature_table(ev, f, "x_r20", 20) for f in feats}
    if args.period == "explore":
        res["features_r5"] = {f: feature_table(ev, f, "x_r5", 5) for f in feats}
        cands = []
        for f, r in res["features"].items():
            ok_t = abs(r["nw_t"]) >= 3 if np.isfinite(r["nw_t"]) else False
            ok_m = (r["monotone"] is None) or (r["monotone"] >= 3)
            if ok_t and ok_m:
                cands.append(
                    {
                        "feature": f,
                        "direction": int(np.sign(r["spread_bp"])),
                        "spread_bp": r["spread_bp"],
                        "nw_t": r["nw_t"],
                    }
                )
        res["candidates"] = cands
    fb = pd.cut(ev["followups"], [-1, 0, 3, 10**9], labels=["0", "1-3", "4+"])
    res["followups"] = {
        str(k): {"n": int(len(g)), "x_r20_bp": float(g["x_r20"].mean() * 1e4)}
        for k, g in ev.groupby(fb, observed=True)
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"result_{args.period}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=str)
    )
    ev.to_parquet(out / f"events_{args.period}.parquet") if args.save_events else None
    return res


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--period", choices=tuple(PERIODS), required=True)
    ap.add_argument("--features", default="", help="confirm 에서만: 탐색이 고른 후보, 쉼표 구분")
    ap.add_argument("--out", default=str(REPO / "data/eval/news_drift_20d"))
    ap.add_argument("--db-env", default="~/git/quant-airflow/.env")
    ap.add_argument("--save-events", action="store_true")
    args = ap.parse_args()
    assert PREREG.exists(), PREREG
    if args.period == "confirm" and not args.features:
        sys.exit("confirm 은 --features 가 필요하다 (탐색이 고른 후보만)")
    res = run(args)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
