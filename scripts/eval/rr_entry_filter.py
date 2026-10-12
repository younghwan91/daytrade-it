"""뉴스 양(+) 판정 × 진입 전 차트 손익비 2:1 필터 — 사전등록 재생.

규칙·합격 바·대조는 전부 돌리기 전에 고정했다:
    docs/research/frozen/2026-10-08-rr-entry-prereg.json
이 스크립트는 그 문서를 그대로 옮긴 것이다. 숫자를 바꾸려면 새 사전등록이 필요하다.

    ~/git/daytrade-it/.venv/bin/python scripts/eval/rr_entry_filter.py \
        --stage 1 --out data/eval/rr_entry

입력은 읽기만 한다. DB 는 `~/git/quant-airflow/.env` 의 KR_QUANT_DB (읽기 전용 SELECT).

  후보   (code, day) 중 v2.sentiment_direction == 1 판정이 있는 종목-일, 시작 = 가장 이른 decide_at.
  스캔   t = max(ceil_minute(decide_at), 09:01) .. 14:30, ts < t 봉만 안다. 진입가 E = 봉 t open.
  손절   S = 당일 저가 so far.  목표 T = {전일 고가, 당일 고가 so far} 중 E 위의 가장 가까운 값.
  조건   (T − E − cE) / (E − S + cE) ≥ 2,  c = costs.round_trip(day, market, at_open=t<09:05).
  청산   갭 손절(open ≤ S) → open, 손절(low ≤ S) → S−1틱, 목표(high > T) → T, 같은 봉이면 손절.
         없으면 15:00 봉 open.

전일 종가(상한가 기준가·이음매)·전일 고가·거래대금은 `krx_session.regular_daily` 로 읽는다 — 2026-09-14 부터
daily_bars 에 KRX 애프터마켓(16:00~20:00)이 섞였기 때문이다(정정 docs/research/frozen/2026-10-08-rr-entry-prereg-amend1.json).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from costs import round_trip
from krx_quant_core.market.limits import limit_up_price
from krx_quant_core.market.ticks import tick_size_int
from krx_quant_core.stats.matched_null import cluster_bootstrap_ci, matched_control
from krx_session import regular_daily

REPO = Path(__file__).resolve().parents[2]
PREREG = REPO / "docs/research/frozen/2026-10-08-rr-entry-prereg.json"

RR_MIN = 2.0
FIRST_SCAN = 1  # 09:01 — 완성 봉이 하나는 있어야 저가가 정의된다
LAST_SCAN = 330  # 14:30
FLAT = 360  # 15:00
OPEN_WINDOW = 5  # t < 09:05 면 장초 스프레드
SEAM = 0.30
N_PER = 3
SEED = 0
N_BOOT = 2000

STAGE1 = (dt.date(2026, 3, 17), dt.date(2026, 9, 15))
STAGE2_FIRST = dt.date(2026, 9, 16)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def db_url(env_path: Path) -> str:
    url = os.environ.get("KR_QUANT_DB")
    if url:
        return url
    for line in env_path.read_text().splitlines():
        if line.startswith("KR_QUANT_DB="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("KR_QUANT_DB 가 없다 (환경 또는 --db-env)")


# ---------------------------------------------------------------- 후보


def _start_minute(decide_at: pd.Timestamp) -> int:
    m = decide_at.hour * 60 + decide_at.minute - 540
    if decide_at.second or decide_at.microsecond:
        m += 1  # ceil: 진행 중인 분은 아직 완성 봉이 아니다
    return max(m, FIRST_SCAN)


def load_stage1(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            v2 = d.get("v2") or {}
            if not d.get("decide_at") or v2.get("sentiment_direction") is None:
                continue
            rows.append(
                {
                    "code": d["code"],
                    "day": d["day"],
                    "market": d.get("market"),
                    "decide_at": d["decide_at"],
                    "direction": int(v2["sentiment_direction"]),
                    "persistence": v2.get("persistence"),
                    "reports_price_move": bool(v2.get("reports_price_move")),
                }
            )
    df = pd.DataFrame(rows)
    df["day"] = pd.to_datetime(df["day"]).dt.date
    lo, hi = STAGE1
    return df[(df["day"] >= lo) & (df["day"] <= hi)].reset_index(drop=True)


def load_stage2(dirs: list[Path]) -> pd.DataFrame:
    best: dict[str, dict] = {}
    for d in dirs:
        for fn in sorted(d.glob("*.jsonl")):
            for line in fn.open():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                v2 = r.get("v2") or {}
                if v2.get("sentiment_direction") is None or not r.get("judged_at"):
                    continue
                k = r["key"]
                if k not in best or r["judged_at"] < best[k]["judged_at"]:
                    best[k] = r
    rows = []
    for r in best.values():
        ja = pd.Timestamp(r["judged_at"]).tz_convert("Asia/Seoul")
        v2 = r["v2"]
        rows.append(
            {
                "code": r["ticker"],
                "day": ja.date(),
                "market": None,
                "decide_at": ja.isoformat(),
                "direction": int(v2["sentiment_direction"]),
                "persistence": v2.get("persistence"),
                "reports_price_move": bool(v2.get("reports_price_move")),
            }
        )
    df = pd.DataFrame(rows)
    return df[df["day"] >= STAGE2_FIRST].reset_index(drop=True)


def candidates(judg: pd.DataFrame, direction: int) -> pd.DataFrame:
    """(code, day) 당 한 행. 시작 = 그 방향 판정 중 가장 이른 decide_at."""
    j = judg[judg["direction"] == direction].copy()
    j["ts"] = pd.to_datetime(j["decide_at"], utc=True).dt.tz_convert("Asia/Seoul")
    j = j.sort_values("ts")
    g = j.groupby(["code", "day"], as_index=False).agg(
        ts=("ts", "first"),
        persistence=("persistence", "first"),
        reports_price_move=("reports_price_move", "first"),
        n_news=("ts", "size"),
    )
    g["start"] = g["ts"].map(_start_minute)
    return g[g["start"] <= LAST_SCAN].reset_index(drop=True)


# ---------------------------------------------------------------- DB


def load_daily(con, first: dt.date, last: dt.date) -> pd.DataFrame:
    """전일 정규장 고가·종가(= 가격제한 기준가)·거래대금. 정규장 값이 없는 전일은 NaN → no_data."""
    d = regular_daily(con, first - dt.timedelta(days=14), last)
    d = d.sort_values(["code", "date"])
    g = d.groupby("code")
    d["prev_high"] = g["high"].shift(1)
    d["prev_close"] = g["close"].shift(1)
    d["prev_tv"] = g["trade_value"].shift(1)
    d["prev_ret"] = d["prev_close"] / g["close"].shift(2) - 1
    return d[(d["date"] >= first) & (d["date"] <= last)].reset_index(drop=True)


def load_minutes(con, day: dt.date, codes: list[str]) -> dict[str, np.ndarray]:
    lo = dt.datetime.combine(day, dt.time(9, 0))
    hi = dt.datetime.combine(day, dt.time(15, 0))
    cur = con.cursor()
    cur.execute(
        "select code, ts, open, high, low, close from minute_bars "
        "where code = any(%s) and ts >= %s and ts <= %s order by code, ts",
        (codes, lo, hi),
    )
    out: dict[str, list] = defaultdict(list)
    for code, ts, o, h, low, c in cur.fetchall():
        out[code].append((ts.hour * 60 + ts.minute - 540, o, h, low, c))
    return {k: np.asarray(v, float) for k, v in out.items()}


# ---------------------------------------------------------------- 규칙


def simulate(
    bars: np.ndarray,
    start: int,
    prev_high: float,
    prev_close: float,
    c_open: float,
    c_late: float,
    *,
    scan_once: bool = False,
) -> dict | None:
    """사전등록 규칙 그대로. 자리가 없으면 None, 이음매로 제외면 {"excluded": True}."""
    if len(bars) < 2 or not (prev_close > 0):
        return None
    m, o, h, low, c = bars.T
    if abs(o[0] / prev_close - 1) > SEAM:
        return {"excluded": True}
    cum_low = np.minimum.accumulate(low)
    cum_high = np.maximum.accumulate(h)
    lu = limit_up_price(prev_close)
    idx = np.arange(1, len(m))
    mi = m[idx]
    valid = (mi >= start) & (mi <= LAST_SCAN)
    if not valid.any():
        return None
    E = o[idx]
    S = cum_low[idx - 1]
    hd = cum_high[idx - 1]
    cost = np.where(mi < OPEN_WINDOW, c_open, c_late)
    ph = prev_high if prev_high is not None and not math.isnan(prev_high) else -np.inf
    T = np.minimum(np.where(ph > E, ph, np.inf), np.where(hd > E, hd, np.inf))
    with np.errstate(invalid="ignore", divide="ignore"):
        rr = (T - E - cost * E) / (E - S + cost * E)
    ok = valid & (S < E) & (lu > E) & np.isfinite(T) & (rr >= RR_MIN)
    if scan_once:
        first_valid = int(np.argmax(valid))
        if not ok[first_valid]:
            return None
        pick = first_valid
    else:
        if not ok.any():
            return None
        pick = int(np.argmax(ok))
    j_entry = pick + 1
    E, S, T, c_, rr_ = (
        float(E[pick]),
        float(S[pick]),
        float(T[pick]),
        float(cost[pick]),
        float(rr[pick]),
    )
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
        "entry_min": int(m[j_entry]),
        "exit_min": int(m[k_exit]),
        "E": E,
        "S": S,
        "T": T,
        "rr": rr_,
        "cost": c_,
        "reason": reason,
        "gross_bp": gross * 1e4,
        "net_bp": (gross - c_) * 1e4,
    }


# ---------------------------------------------------------------- 통계


def summarize(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {"n": 0}
    v = tr["net_bp"].to_numpy()
    days = tr["day"].astype(str).to_numpy()
    lo, hi, _ = cluster_bootstrap_ci(v, days, n_boot=N_BOOT, seed=SEED)
    win = tr["net_bp"] > 0
    avg_w = tr.loc[win, "net_bp"].mean()
    avg_l = -tr.loc[~win, "net_bp"].mean()
    cut = np.quantile(v, 0.99)
    monthly = (
        tr.assign(month=tr["day"].astype(str).str[:7])
        .groupby("month")["net_bp"]
        .agg(["mean", "size"])
    )
    return {
        "n": int(len(tr)),
        "days": int(tr["day"].nunique()),
        "mean": float(v.mean()),
        "ci": [lo, hi],
        "gross_mean": float(tr["gross_bp"].mean()),
        "cost_mean": float(tr["cost"].mean() * 1e4),
        "win_rate": float(win.mean()),
        "avg_win": float(avg_w) if win.any() else None,
        "avg_loss": float(avg_l) if (~win).any() else None,
        "payoff": float(avg_w / avg_l) if win.any() and (~win).any() and avg_l > 0 else None,
        "ex_top1_mean": float(v[v < cut].mean()) if (v < cut).any() else None,
        "reasons": tr["reason"].value_counts().to_dict(),
        "monthly": {k: [float(r["mean"]), int(r["size"])] for k, r in monthly.iterrows()},
        "per_day": float(len(tr) / tr["day"].nunique()),
    }


def max_concurrent(tr: pd.DataFrame) -> int:
    best = 0
    for _, g in tr.groupby("day"):
        ev = sorted(
            [(a, 1) for a in g["entry_min"]] + [(b, -1) for b in g["exit_min"]],
            key=lambda x: (x[0], x[1]),
        )
        cur = 0
        for _, s in ev:
            cur += s
            best = max(best, cur)
    return best


def diff_ci(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """a 평균 − b 평균, 날짜를 같이 재추출."""
    if a.empty or b.empty:
        return {}
    days = sorted(set(a["day"].astype(str)) | set(b["day"].astype(str)))
    idx = {d: i for i, d in enumerate(days)}
    k = len(days)

    def sums(t):
        i = t["day"].astype(str).map(idx).to_numpy()
        return (
            np.bincount(i, weights=t["net_bp"].to_numpy(), minlength=k),
            np.bincount(i, minlength=k).astype(float),
        )

    sa, na = sums(a)
    sb, nb = sums(b)
    rng = np.random.default_rng(SEED)
    draw = rng.integers(0, k, size=(N_BOOT, k))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = sa[draw].sum(1) / na[draw].sum(1) - sb[draw].sum(1) / nb[draw].sum(1)
    point = a["net_bp"].mean() - b["net_bp"].mean()
    return {
        "alpha": float(point),
        "ci": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))],
        "p_gt0": float(np.nanmean(boot > 0)),
    }


# ---------------------------------------------------------------- 실행


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
    first, last = days_all[0], days_all[-1]
    daily = load_daily(con, first, last)
    daily = daily[daily["code"].isin(market)]
    daily["tv_q"] = daily.groupby("date")["prev_tv"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=False)
    )
    daily["ret_q"] = daily.groupby("date")["prev_ret"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=False)
    )
    dkey = daily.set_index(["code", "date"])

    # 매치드 널: 후보(양 판정) 마다 그날 아무 판정도 없는 종목 N_PER 개, 시작 시각 상속.
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
    trades_df = pos.rename(columns={"date": "_d"})
    ctl = matched_control(
        trades_df, pool, strata=["day", "tv_q", "ret_q"], n_per=N_PER, seed=SEED, exclude_self=False
    )
    ctl["code"] = pool.loc[ctl["control_idx"], "code"].to_numpy()
    ctl["start"] = trades_df.loc[ctl["trade_idx"], "start"].to_numpy()
    log(f"후보 +{len(groups[1]):,} 0:{len(groups[0]):,} -:{len(groups[-1]):,}, 대조 {len(ctl):,}")

    jobs = []  # (group, day, code, start, extra)
    for gname, g in (("pos", groups[1]), ("neu", groups[0]), ("neg", groups[-1])):
        for r in g.itertuples(index=False):
            jobs.append(
                (
                    gname,
                    r.day,
                    r.code,
                    r.start,
                    {"persistence": r.persistence, "rpm": r.reports_price_move},
                )
            )
    for r in ctl.itertuples(index=False):
        jobs.append(("null", r.day, r.code, r.start, {}))
    by_day: dict[dt.date, list] = defaultdict(list)
    for jb in jobs:
        by_day[jb[1]].append(jb)

    cost_cache: dict = {}
    out_rows: list[dict] = []
    once_rows: list[dict] = []
    counts = defaultdict(lambda: defaultdict(int))
    for i, day in enumerate(sorted(by_day)):
        js = by_day[day]
        bars = load_minutes(con, day, sorted({jb[2] for jb in js}))
        for gname, _, code, start, extra in js:
            counts[gname]["candidates"] += 1
            mk = market.get(code)
            if mk is None or (code, day) not in dkey.index or code not in bars:
                counts[gname]["no_data"] += 1
                continue
            drow = dkey.loc[(code, day)]
            if not drow["prev_close"] > 0:  # 전일 정규장 종가 없음(애프터마켓 이후 분봉 공백)
                counts[gname]["no_data"] += 1
                continue
            key = (day, mk)
            if key not in cost_cache:
                cost_cache[key] = (
                    round_trip(day, mk, at_open=True),
                    round_trip(day, mk, at_open=False),
                )
            c_open, c_late = cost_cache[key]
            res = simulate(bars[code], start, drow["prev_high"], drow["prev_close"], c_open, c_late)
            if res is None:
                counts[gname]["no_setup"] += 1
            elif res.get("excluded"):
                counts[gname]["seam"] += 1
            else:
                counts[gname]["trades"] += 1
                out_rows.append(
                    {
                        "group": gname,
                        "day": day,
                        "code": code,
                        "market": mk,
                        "start": start,
                        **extra,
                        **res,
                    }
                )
            if gname == "pos":
                r1 = simulate(
                    bars[code],
                    start,
                    drow["prev_high"],
                    drow["prev_close"],
                    c_open,
                    c_late,
                    scan_once=True,
                )
                if r1 and not r1.get("excluded"):
                    once_rows.append({"group": "pos_once", "day": day, "code": code, **r1})
        if i % 10 == 0:
            log(f"  {day} ({i + 1}/{len(by_day)}) 거래 {len(out_rows):,}")

    tr = pd.DataFrame(out_rows)
    once = pd.DataFrame(once_rows)
    res = {
        "stage": args.stage,
        "range": [str(first), str(last)],
        "counts": {k: dict(v) for k, v in counts.items()},
    }
    for gname in ("pos", "neu", "neg", "null"):
        t = tr[tr["group"] == gname] if not tr.empty else tr
        res[gname] = summarize(t)
        if not t.empty:
            res[gname]["max_concurrent"] = max_concurrent(t)
    res["pos_once"] = summarize(once)
    if not tr.empty:
        pos_t = tr[tr["group"] == "pos"]
        res["news_alpha_vs_null"] = diff_ci(pos_t, tr[tr["group"] == "null"])
        res["by_market"] = {k: summarize(g) for k, g in pos_t.groupby("market")}
        res["by_persistence"] = {
            str(k): summarize(g) for k, g in pos_t.groupby("persistence", dropna=False)
        }
        res["by_rpm"] = {str(k): summarize(g) for k, g in pos_t.groupby("rpm")}
    res["verdict"] = verdict(res, args.stage)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tr.to_csv(out / f"trades_stage{args.stage}.csv", index=False)
    (out / f"result_stage{args.stage}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=str)
    )
    return res


def verdict(res: dict, stage: int) -> dict:
    p = res.get("pos", {})
    if not p.get("n"):
        return {"result": "no_trades"}
    if stage == 1:
        months = p["monthly"]
        p1 = p["mean"] > 0 and p["ci"][0] > 0
        p2 = sum(1 for v in months.values() if v[0] > 0) >= 4
        p3 = (p["ex_top1_mean"] or -1) > 0
        return {
            "P1_mean_ci_gt0": p1,
            "P2_pos_months_ge4": p2,
            "P3_ex_top1_gt0": p3,
            "pos_months": f"{sum(1 for v in months.values() if v[0] > 0)}/{len(months)}",
            "result": "PASS" if (p1 and p2 and p3) else "REJECT",
        }
    if p["n"] < 100:
        return {"result": "DEFER", "n": p["n"]}
    return {"result": "PASS" if p["mean"] > 0 else "REJECT", "n": p["n"]}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--out", default=str(REPO / "data/eval/rr_entry"))
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
