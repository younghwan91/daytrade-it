"""뉴스 양(+) 판정 × 거래대금 5배 양봉 진입, −2% / +4% 고정 — 사전등록 재생.

규칙·합격 바·대조는 돌리기 전에 고정했다:
    docs/research/frozen/2026-10-09-volume-entry-prereg.json
후보·통계·대조는 rr_entry_filter 와 같은 함수를 쓰고, 진입·청산만 다르다.

    ~/git/daytrade-it/.venv/bin/python scripts/eval/volume_entry.py --stage 1 --out data/eval/volume_entry

  트리거  09:01 이후, decide 가 속한 분 이후의 양봉, close×volume ≥ 5 × (직전 20거래일 평균 거래대금 / 380).
  진입    트리거 다음 봉 open (≤ 14:30, 상한가면 다음 트리거).
  청산    −2% (갭이면 open, 아니면 S−1틱) / +4% (엄격히 넘을 때) / 같은 봉이면 손절 / 15:00.

전일 종가(상한가 기준가·이음매)·거래대금은 `krx_session.regular_daily` 로 읽는다 — 2026-09-14 부터 daily_bars 에
KRX 애프터마켓(16:00~20:00)이 섞였기 때문이다(정정 docs/research/frozen/2026-10-09-volume-entry-amend2.json).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from costs import round_trip
from krx_quant_core.market.limits import limit_up_price
from krx_quant_core.market.ticks import tick_size_int
from krx_quant_core.stats.matched_null import matched_control
from krx_session import regular_daily
from rr_entry_filter import (
    FLAT,
    LAST_SCAN,
    N_PER,
    OPEN_WINDOW,
    REPO,
    SEAM,
    SEED,
    candidates,
    db_url,
    diff_ci,
    load_stage1,
    load_stage2,
    log,
    max_concurrent,
    summarize,
)

PREREG = REPO / "docs/research/frozen/2026-10-09-volume-entry-amend1.json"
SPIKE = 5.0
BASE_DAYS = 20
SESSION_MIN = 380
#: daily_bars.trade_value 는 백만원 단위다(분봉 close×volume 합과 비율 ~1e6, amend1). 원으로 바꿔 쓴다.
DAILY_TV_UNIT = 1_000_000
STOP = 0.02
TARGET = 0.04


def minute_base(daily_tv_million):
    """직전 20거래일 평균 일 거래대금(백만원) → 평균 1분 거래대금(원)."""
    return daily_tv_million * DAILY_TV_UNIT / SESSION_MIN


def load_daily(con, first: dt.date, last: dt.date) -> pd.DataFrame:
    """전일 정규장 종가(= 가격제한 기준가)·거래대금. 정규장 값이 없는 전일은 NaN → no_data."""
    d = regular_daily(con, first - dt.timedelta(days=45), last)
    d = d.sort_values(["code", "date"])
    g = d.groupby("code")
    d["prev_close"] = g["close"].shift(1)
    d["prev_tv"] = g["trade_value"].shift(1)
    d["prev_ret"] = d["prev_close"] / g["close"].shift(2) - 1
    d["base"] = minute_base(
        g["trade_value"].transform(lambda s: s.shift(1).rolling(BASE_DAYS, min_periods=1).mean())
    )
    return d[(d["date"] >= first) & (d["date"] <= last)].reset_index(drop=True)


def load_minutes(con, day: dt.date, codes: list[str]) -> dict[str, np.ndarray]:
    lo = dt.datetime.combine(day, dt.time(9, 0))
    hi = dt.datetime.combine(day, dt.time(15, 0))
    cur = con.cursor()
    cur.execute(
        "select code, ts, open, high, low, close, volume from minute_bars "
        "where code = any(%s) and ts >= %s and ts <= %s order by code, ts",
        (codes, lo, hi),
    )
    out: dict[str, list] = defaultdict(list)
    for code, ts, o, h, low, c, v in cur.fetchall():
        out[code].append((ts.hour * 60 + ts.minute - 540, o, h, low, c, v or 0))
    return {k: np.asarray(v, float) for k, v in out.items()}


def buy_share(con, day: dt.date, code: str, minute: int) -> float | None:
    """트리거 분의 매수 주도 체결 비중. ticks 가 없으면 None (참고 지표)."""
    t0 = dt.datetime.combine(day, dt.time(9, 0)) + dt.timedelta(minutes=minute)
    cur = con.cursor()
    cur.execute(
        "select sum(volume) filter (where side = 1), sum(volume) from ticks where code = %s and ts >= %s and ts < %s",
        (code, t0, t0 + dt.timedelta(minutes=1)),
    )
    buy, tot = cur.fetchone()
    return float(buy or 0) / float(tot) if tot else None


def simulate(
    bars: np.ndarray, start: int, base: float, prev_close: float, c_open: float, c_late: float
) -> dict | None:
    """사전등록 규칙 그대로. 자리가 없으면 None, 이음매로 제외면 {"excluded": True}."""
    if len(bars) < 2 or not (prev_close > 0) or not (base > 0):
        return None
    m, o, h, low, c, v = bars.T
    if abs(o[0] / prev_close - 1) > SEAM:
        return {"excluded": True}
    lu = limit_up_price(prev_close)
    trig = (m >= 1) & (m >= start - 1) & (c > o) & (c * v >= SPIKE * base)
    j_entry = None
    for b in np.flatnonzero(trig):
        j = b + 1
        if j >= len(m) or m[j] > LAST_SCAN:
            break
        if o[j] >= lu:
            continue
        j_entry, b_trig = j, b
        break
    if j_entry is None:
        return None
    E = float(o[j_entry])
    S, T = E * (1 - STOP), E * (1 + TARGET)
    cost = c_open if m[j_entry] < OPEN_WINDOW else c_late
    exit_px, reason, k_exit = None, None, None
    for k in range(j_entry, len(m)):
        if m[k] >= FLAT:
            exit_px, reason, k_exit = o[k], "time", k
            break
        if k > j_entry and o[k] <= S:
            exit_px, reason, k_exit = o[k], "gap_stop", k
            break
        if low[k] <= S:
            exit_px, reason, k_exit = S - tick_size_int(S - 1), "stop", k
            break
        if h[k] > T:
            exit_px, reason, k_exit = T, "target", k
            break
    if exit_px is None:
        k_exit = len(m) - 1
        exit_px, reason = c[k_exit], "time"
    gross = exit_px / E - 1
    return {
        "trigger_min": int(m[b_trig]),
        "spike": float(c[b_trig] * v[b_trig] / base),
        "entry_min": int(m[j_entry]),
        "exit_min": int(m[k_exit]),
        "E": E,
        "cost": cost,
        "reason": reason,
        "gross_bp": gross * 1e4,
        "net_bp": (gross - cost) * 1e4,
    }


def verdict(res: dict, stage: int) -> dict:
    p = res.get("pos", {})
    if not p.get("n"):
        return {"result": "no_trades"}
    if stage == 1:
        pos_m = sum(1 for v in p["monthly"].values() if v[0] > 0)
        p1 = p["mean"] > 0 and p["ci"][0] > 0
        p2 = pos_m >= 4
        p3 = (p["ex_top1_mean"] or -1) > 0
        return {
            "P1_mean_ci_gt0": p1,
            "P2_pos_months_ge4": p2,
            "P3_ex_top1_gt0": p3,
            "pos_months": f"{pos_m}/{len(p['monthly'])}",
            "result": "PASS" if (p1 and p2 and p3) else "REJECT",
        }
    if p["n"] < 100:
        return {"result": "DEFER", "n": p["n"]}
    return {"result": "PASS" if p["mean"] > 0 else "REJECT", "n": p["n"]}


def run(args) -> dict:
    import psycopg2

    con = psycopg2.connect(db_url(Path(args.db_env).expanduser()))
    if args.stage == 1:
        judg = load_stage1(REPO.parent / "daytrade-it/data/eval/prompt_v3/master.jsonl")
    else:
        judg = load_stage2(
            [
                REPO.parent / "daytrade-it/data/news_judgments_paper",
                REPO.parent / "daytrade-it-judge/data/news_judgments_paper",
            ]
        )
    log(f"판정 {len(judg):,} 행, {judg['day'].min()}..{judg['day'].max()}")
    stocks = pd.read_sql("select code, market from stocks where market in ('거래소','코스닥')", con)
    market = dict(zip(stocks["code"], stocks["market"], strict=True))

    groups = {d: candidates(judg, d) for d in (1, 0, -1)}
    days_all = sorted(set(groups[1]["day"]))
    daily = load_daily(con, days_all[0], days_all[-1])
    daily = daily[daily["code"].isin(market)].copy()
    for col, q in (("prev_tv", "tv_q"), ("prev_ret", "ret_q")):
        daily[q] = daily.groupby("date")[col].transform(
            lambda s: pd.qcut(s.rank(method="first"), 4, labels=False)
        )
    dkey = daily.set_index(["code", "date"])

    news_codes = judg.groupby("day")["code"].apply(set).to_dict()
    pos = groups[1].merge(
        daily[["code", "date", "tv_q", "ret_q"]],
        left_on=["code", "day"],
        right_on=["code", "date"],
        how="left",
    )
    pool = daily[["code", "date", "tv_q", "ret_q"]].copy()
    pool = pool[
        [c not in news_codes.get(d, ()) for c, d in zip(pool["code"], pool["date"], strict=True)]
    ]
    pool = pool.rename(columns={"date": "day"}).reset_index(drop=True)
    pool.index = pool.index + 10_000_000
    ctl = matched_control(
        pos, pool, strata=["day", "tv_q", "ret_q"], n_per=N_PER, seed=SEED, exclude_self=False
    )
    ctl["code"] = pool.loc[ctl["control_idx"], "code"].to_numpy()
    ctl["start"] = pos.loc[ctl["trade_idx"], "start"].to_numpy()
    log(f"후보 +{len(groups[1]):,} 0:{len(groups[0]):,} -:{len(groups[-1]):,}, 대조 {len(ctl):,}")

    by_day: dict[dt.date, list] = defaultdict(list)
    for gname, g in (("pos", groups[1]), ("neu", groups[0]), ("neg", groups[-1])):
        for r in g.itertuples(index=False):
            by_day[r.day].append((gname, r.code, r.start))
    for r in ctl.itertuples(index=False):
        by_day[r.day].append(("null", r.code, r.start))

    cost_cache: dict = {}
    rows: list[dict] = []
    counts = defaultdict(lambda: defaultdict(int))
    for i, day in enumerate(sorted(by_day)):
        js = by_day[day]
        bars = load_minutes(con, day, sorted({jb[1] for jb in js}))
        for gname, code, start in js:
            counts[gname]["candidates"] += 1
            mk = market.get(code)
            if mk is None or (code, day) not in dkey.index or code not in bars:
                counts[gname]["no_data"] += 1
                continue
            drow = dkey.loc[(code, day)]
            if not drow["prev_close"] > 0:  # 전일 정규장 종가 없음(애프터마켓 이후 분봉 공백)
                counts[gname]["no_data"] += 1
                continue
            if (day, mk) not in cost_cache:
                cost_cache[(day, mk)] = (
                    round_trip(day, mk, at_open=True),
                    round_trip(day, mk, at_open=False),
                )
            res = simulate(
                bars[code], start, drow["base"], drow["prev_close"], *cost_cache[(day, mk)]
            )
            if res is None:
                counts[gname]["no_trigger"] += 1
            elif res.get("excluded"):
                counts[gname]["seam"] += 1
            else:
                counts[gname]["trades"] += 1
                bs = (
                    buy_share(con, day, code, res["trigger_min"])
                    if gname == "pos" and day >= dt.date(2026, 8, 24)
                    else None
                )
                rows.append(
                    {
                        "group": gname,
                        "day": day,
                        "code": code,
                        "market": mk,
                        "start": start,
                        "buy_share": bs,
                        **res,
                    }
                )
        if i % 10 == 0:
            log(f"  {day} ({i + 1}/{len(by_day)}) 거래 {len(rows):,}")

    tr = pd.DataFrame(rows)
    res = {
        "stage": args.stage,
        "range": [str(days_all[0]), str(days_all[-1])],
        "counts": {k: dict(v) for k, v in counts.items()},
    }
    for gname in ("pos", "neu", "neg", "null"):
        t = tr[tr["group"] == gname] if not tr.empty else tr
        res[gname] = summarize(t)
        if not t.empty:
            res[gname]["max_concurrent"] = max_concurrent(t)
    if not tr.empty:
        p = tr[tr["group"] == "pos"]
        res["news_alpha_vs_null"] = diff_ci(p, tr[tr["group"] == "null"])
        res["by_market"] = {k: summarize(g) for k, g in p.groupby("market")}
        res["by_entry_hour"] = {
            int(k): summarize(g) for k, g in p.groupby(p["entry_min"] // 60 + 9)
        }
        bs = p.dropna(subset=["buy_share"])
        if len(bs):
            res["by_buy_share"] = {
                str(k): summarize(g) for k, g in bs.groupby(bs["buy_share"] >= 0.6)
            }
    res["verdict"] = verdict(res, args.stage)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tr.to_csv(out / f"trades_stage{args.stage}.csv", index=False)
    (out / f"result_stage{args.stage}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=str)
    )
    return res


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--out", default=str(REPO / "data/eval/volume_entry"))
    ap.add_argument("--db-env", default="~/git/quant-airflow/.env")
    args = ap.parse_args()
    assert PREREG.exists(), PREREG
    res = run(args)
    print(
        json.dumps(
            {
                k: res[k]
                for k in ("counts", "pos", "null", "news_alpha_vs_null", "verdict")
                if k in res
            },
            ensure_ascii=False,
            indent=1,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
