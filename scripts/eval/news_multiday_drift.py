"""뉴스 1단계 순위의 다일(15:00 이후) 지평 드리프트 — 기술(description)이지 판정이 아니다.

1단계 선정(tfidf+수치, 타깃 mfe_n)은 확인 구간 IC 0.132 로 견고하나 경제 곡선은 일중(15:00
청산) 라벨에서 전 구간 음수였다. 이 스크립트는 **같은 순위**를 15:00 이후 지평(야간·1~20
거래일)에서 잰다. 이 지평의 라벨은 1단계 선택에 한 번도 쓰이지 않았다.

    .venv/bin/python scripts/eval/news_multiday_drift.py \
        --scores ~/git/daytrade-it-rlnews/data/eval/pipeline/stage1_oof.jsonl \
        --out data/eval/news_multiday

입력은 읽기만 한다. DB 는 `~/git/quant-airflow/.env` 의 KR_QUANT_DB (읽기 전용 SELECT).

단위·순위·지평·대조·통계는 전부 사전에 고정했다 (docs/research/2026-10-05-news-multiday-drift.md):
  단위   종목-일(code, day). 같은 날 여러 뉴스는 최대 score_mfe_n 하나로 합친다.
  순위   날짜 내 백분위, 상위 1% / 5% / 10% (ceil(f·n), stage1_select.econ_curve 와 같은 규칙), 10분위.
  지평   ① 종가(t)→익일 시가 ② 종가→종가(t+1) ③ t+2 ④ t+5 ⑤ t+10 ⑥ t+20
         ⑦ 익일 시가→t+5 종가 ⑧ 익일 시가→t+20 종가.   전부 bp, daily_bars_adjusted.
  대조   (a) 같은 날 뉴스 종목 전체 평균 (b) 매치드 널: 같은 날 × 전일 거래대금 4분위 × 전일 수익률
         4분위에서 뽑은 무작위 종목-일 (상위 300 유니버스, 뉴스 유무 무관), n_per=5, seed=0.
  통계   날짜 클러스터 부트스트랩 95% CI, 일평균 Newey-West t(lag = 지평 거래일), 음수월 비율,
         split 별(selection 은 참고, confirm+post_train 이 깨끗함), 월별 부호.
  취약성 상위 1% 거래 제거 후 평균, 상한가 마감일(종가 매수 불가) 제외.
  비용   비차감 표 + 41.5bp(scalp-it 왕복) 차감, 야간은 23bp(경매→경매) 병기. 이 레포 36.7bp 는 각주.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from krx_quant_core.stats.matched_null import cluster_bootstrap_ci, matched_alpha, matched_control
from krx_quant_core.stats.metrics import newey_west_t

REPO = Path(__file__).resolve().parents[2]

#: 지평 이름 → (진입, 청산). 진입은 decide_at 이후의 가격만: 종가(t) 또는 익일 시가.
HORIZONS: dict[str, tuple[str, str]] = {
    "c0_o1": ("close", "open+1"),
    "c0_c1": ("close", "close+1"),
    "c0_c2": ("close", "close+2"),
    "c0_c5": ("close", "close+5"),
    "c0_c10": ("close", "close+10"),
    "c0_c20": ("close", "close+20"),
    "o1_c5": ("open+1", "close+5"),
    "o1_c20": ("open+1", "close+20"),
}
HORIZON_LAG = {
    "c0_o1": 1,
    "c0_c1": 1,
    "c0_c2": 2,
    "c0_c5": 5,
    "c0_c10": 10,
    "c0_c20": 20,
    "o1_c5": 5,
    "o1_c20": 20,
}
HORIZON_LABEL = {
    "c0_o1": "종가→익일시가(야간)",
    "c0_c1": "종가→t+1",
    "c0_c2": "종가→t+2",
    "c0_c5": "종가→t+5",
    "c0_c10": "종가→t+10",
    "c0_c20": "종가→t+20",
    "o1_c5": "익일시가→t+5",
    "o1_c20": "익일시가→t+20",
}
CUTS = {"top1": 0.01, "top5": 0.05, "top10": 0.10}
SPLITS = {
    "all_oof": ("selection", "confirm", "post_train"),
    "clean": ("confirm", "post_train"),
    "selection": ("selection",),
    "confirm": ("confirm",),
    "post_train": ("post_train",),
}
COST_RT_BP = 41.5  # scalp-it 왕복 (문서 명시)
COST_OVN_BP = 23.0  # 야간 경매→경매 가정
COST_REPO_BP = 36.7  # 이 레포 scripts/eval/costs.py::round_trip (각주)
UNIVERSE = 300
MARKET_ETF = "069500"
LIMIT_UP = 0.295  # KRX ±30% 가격제한, 호가 단위 반올림 여유
SCORE_MAIN = "score_mfe_n"
SCORE_APPX = ("score_ret_n", "score_rr", "score_lrr", "score_mfe", "score_ret_close")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- data
def load_scores(path: Path) -> pd.DataFrame:
    rows = pd.read_json(path, lines=True, dtype={"code": str})
    rows = rows[rows[SCORE_MAIN].notna()].copy()
    rows["code"] = rows["code"].astype(str).str.zfill(6)
    rows["decide_t"] = pd.to_datetime(rows["decide_at"].astype(str).str[:19])
    agg = {
        SCORE_MAIN: "max",
        "split": "first",
        "month": "first",
        "key": "size",
        "decide_t": "max",
        "trained_through": "max",
    }
    for s in SCORE_APPX:
        agg[s] = "max"
    g = rows.groupby(["code", "day"], as_index=False).agg(agg).rename(columns={"key": "n_news"})
    g["day"] = pd.to_datetime(g["day"]).dt.date
    return g


def db_url(env_path: Path) -> str:
    url = os.environ.get("KR_QUANT_DB")
    if url:
        return url
    for line in env_path.read_text().splitlines():
        if line.startswith("KR_QUANT_DB="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("KR_QUANT_DB 가 없다 (환경 또는 --db-env)")


def load_bars(
    url: str, first: dt.date, last: dt.date
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import psycopg2

    con = psycopg2.connect(url)
    adj = pd.read_sql(
        "select code, date, open, close, trade_value from daily_bars_adjusted "
        "where date >= %s and date <= %s",
        con,
        params=(first, last),
    )
    raw = pd.read_sql(
        "select code, date, close from daily_bars where date >= %s and date <= %s",
        con,
        params=(first, last),
    )
    delisted = pd.read_sql("select code, name, market, last_trade_date from delisted_stocks", con)
    con.close()
    return adj, raw, delisted


# ------------------------------------------------------------------- construction
def build_universe(adj: pd.DataFrame, raw: pd.DataFrame, days: list[dt.date]) -> pd.DataFrame:
    """날짜 d 의 유니버스 = 전 거래일 거래대금 상위 300 (select_universe 규칙, 069500 제외).
    버킷은 전일 정보만: 전일 거래대금 4분위, 전일 수익률(종가 d-1 / 종가 d-2) 4분위."""
    tdays = sorted(adj["date"].unique())
    pos = {d: i for i, d in enumerate(tdays)}
    piv_c = adj.pivot(index="date", columns="code", values="close").sort_index()
    piv_o = adj.pivot(index="date", columns="code", values="open").sort_index()
    piv_tv = adj.pivot(index="date", columns="code", values="trade_value").sort_index()
    raw_c = raw.pivot(index="date", columns="code", values="close").sort_index()
    ok_code = np.array([len(c) == 6 and c.isdigit() and c != MARKET_ETF for c in piv_c.columns])
    out = []
    for d in days:
        i = pos.get(d)
        if i is None or i < 2:
            continue
        p1, p2 = tdays[i - 1], tdays[i - 2]
        tv = piv_tv.loc[p1].to_numpy(dtype=float)
        tv = np.where(ok_code, tv, np.nan)
        order = np.lexsort((piv_tv.columns.to_numpy(), -np.nan_to_num(tv, nan=-1.0)))
        order = [j for j in order if tv[j] > 0][:UNIVERSE]
        codes = piv_tv.columns.to_numpy()[order]
        prev_ret = (piv_c.loc[p1, codes] / piv_c.loc[p2, codes] - 1.0).to_numpy(dtype=float)
        rc = raw_c.reindex(columns=codes)
        prev_raw = (
            rc.loc[p1].to_numpy(dtype=float) if p1 in rc.index else np.full(len(codes), np.nan)
        )
        d_raw = rc.loc[d].to_numpy(dtype=float) if d in rc.index else np.full(len(codes), np.nan)
        frame = pd.DataFrame(
            {
                "code": codes,
                "day": d,
                "prev_tv": tv[order],
                "prev_ret": prev_ret,
                "limit_up": (d_raw / prev_raw - 1.0) >= LIMIT_UP,
            }
        )

        def px(kind: str, k: int, *, i: int = i, codes: np.ndarray = codes) -> np.ndarray:
            j = i + k
            if j >= len(tdays):
                return np.full(len(codes), np.nan)
            src = piv_c if kind == "close" else piv_o
            v = src.loc[tdays[j], codes].to_numpy(dtype=float)
            return np.where(v > 0, v, np.nan)

        c0 = px("close", 0)
        for h, (a, b) in HORIZONS.items():
            pa = c0 if a == "close" else px("open", 1)
            kind, k = b.split("+")
            pb = px(kind, int(k))
            frame[h] = (pb / pa - 1.0) * 1e4
        frame["has_t20"] = np.isfinite(px("close", 20))
        out.append(frame)
    uni = pd.concat(out, ignore_index=True)
    # 전일 정보로만 만든 날짜 내 4분위 (0 = 상위).
    for col, name in (("prev_tv", "tv_q"), ("prev_ret", "ret_q")):
        r = uni.groupby("day")[col].rank(method="first", ascending=False)
        n = uni.groupby("day")[col].transform("count")
        uni[name] = np.floor((r - 1) * 4 / n).clip(0, 3)
    uni["stratum_day"] = uni["day"].astype(str)
    return uni


def rank_news(news: pd.DataFrame) -> pd.DataFrame:
    news = news.copy()
    n = news.groupby("day")[SCORE_MAIN].transform("size")
    r = news.groupby("day")[SCORE_MAIN].rank(method="first", ascending=False)
    news["day_n"] = n
    news["rank"] = r
    news["pct"] = (r - 0.5) / n
    news["decile"] = np.floor((r - 1) * 10 / n).clip(0, 9).astype(int)
    for cut, f in CUTS.items():
        news[cut] = r <= np.ceil(f * n)
    return news


# ------------------------------------------------------------------------- stats
def cell_stats(v: pd.Series, days: pd.Series, lag: int, n_boot: int, seed: int) -> dict:
    ok = v.notna()
    v, days = v[ok].astype(float), days[ok]
    if len(v) == 0:
        return {"n": 0}
    lo, hi, _ = cluster_bootstrap_ci(
        v.to_numpy(), days.astype(str).to_numpy(), n_boot=n_boot, seed=seed
    )
    dd = pd.to_datetime(days.astype(str))
    daily = v.groupby(dd.to_numpy()).mean().sort_index()
    monthly = v.groupby(dd.dt.to_period("M").to_numpy()).mean().sort_index()
    t = np.nan
    if len(daily) >= 3:
        _, t = newey_west_t(daily.to_numpy(), lag=min(lag, len(daily) - 2))
    k = max(1, int(math.ceil(0.01 * len(v))))
    trimmed = float(v.sort_values().iloc[:-k].mean()) if len(v) > k else np.nan
    return {
        "n": int(len(v)),
        "n_days": int(len(daily)),
        "mean_bp": float(v.mean()),
        "median_bp": float(v.median()),
        "ci_low": lo,
        "ci_high": hi,
        "t_nw": float(t),
        "day_pos_share": float((daily > 0).mean()),
        "neg_month_share": float((monthly < 0).mean()),
        "n_months": int(len(monthly)),
        "month_signs": "".join("+" if x > 0 else "-" for x in monthly),
        "trim_top1_mean_bp": trimmed,
    }


def excess_vs_all_news(
    sub: pd.DataFrame, allnews: pd.DataFrame, h: str, n_boot: int, seed: int
) -> dict:
    """대조 (a): 상위 집합 평균 − 같은 날 뉴스 종목 전체 평균, 날짜 i.i.d. 부트스트랩."""
    a = sub.groupby("day")[h].mean()
    b = allnews.groupby("day")[h].mean()
    m = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(m) < 3:
        return {"excess_bp": np.nan}
    diff = (m["a"] - m["b"]).to_numpy()
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return {
        "excess_bp": float(diff.mean()),
        "excess_ci_low": float(np.percentile(means, 2.5)),
        "excess_ci_high": float(np.percentile(means, 97.5)),
        "excess_days": int(len(diff)),
    }


def null_alpha(
    trades: pd.DataFrame, pool: pd.DataFrame, controls: pd.DataFrame, h: str, n_boot: int, seed: int
) -> dict:
    ctrl = controls[controls["trade_idx"].isin(trades.index)]
    ctrl = ctrl.copy()
    ctrl.attrs["unmatched"] = [i for i in controls.attrs.get("unmatched", []) if i in trades.index]
    # 지평 값이 없는 트레이드·대조는 뺀다 (t+20 미도래 등).
    tr = trades[trades[h].notna()]
    ctrl = ctrl[
        ctrl["trade_idx"].isin(tr.index) & pool.loc[ctrl["control_idx"], h].notna().to_numpy()
    ]
    if len(ctrl) == 0:
        return {"alpha_bp": np.nan}
    ma = matched_alpha(tr, pool, ctrl, value=h, cluster="stratum_day", n_boot=n_boot, seed=seed)
    return {
        "alpha_bp": ma.alpha,
        "alpha_ci_low": ma.ci_low,
        "alpha_ci_high": ma.ci_high,
        "alpha_p_greater": ma.p_greater,
        "null_mean_bp": ma.control_mean,
        "alpha_n_trades": ma.n_trades,
        "alpha_n_controls": ma.n_controls,
        "alpha_n_days": ma.n_clusters,
    }


# -------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--db-env", type=Path, default=Path.home() / "git/quant-airflow/.env")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-per", type=int, default=5, help="매치드 널: 트레이드당 대조 수")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    news = load_scores(args.scores)
    log(
        f"scored code-days {len(news)}  days {news['day'].nunique()}  split {news['split'].value_counts().to_dict()}"
    )
    first = min(news["day"]) - dt.timedelta(days=10)
    last = dt.date.today()
    adj, raw, delisted = load_bars(db_url(args.db_env), first, last)
    adj["code"] = adj["code"].astype(str)
    raw["code"] = raw["code"].astype(str)
    log(f"bars {len(adj)} rows, {adj['date'].nunique()} days, last {adj['date'].max()}")

    days = sorted(news["day"].unique())
    uni = build_universe(adj, raw, days)
    uni.index = pd.RangeIndex(len(uni))
    log(f"universe code-days {len(uni)} over {uni['day'].nunique()} days")

    # ---- 뉴스 종목-일을 유니버스에 붙인다 (look-ahead 점검: 유니버스는 전일 기준) ----
    m = news.merge(uni, on=["code", "day"], how="left", indicator=True)
    in_uni = (m["_merge"] == "both").sum()
    checks = {
        "news_code_days": int(len(news)),
        "news_in_prev_day_universe": int(in_uni),
        "news_in_prev_day_universe_share": float(in_uni / len(news)),
        "decide_at_max_time": str(news["decide_t"].dt.time.max()),
        "entry_after_decide_all": bool((news["decide_t"].dt.time <= dt.time(15, 30)).all()),
        "score_trained_before_day_all": bool(
            (pd.to_datetime(news["trained_through"]).dt.date < news["day"]).all()
        ),
    }
    news = m[m["_merge"] == "both"].drop(columns="_merge")
    news = rank_news(news)
    # 트레이드와 풀이 같은 인덱스 공간이어야 exclude_self 가 동작한다.
    key_to_idx = pd.Series(uni.index, index=pd.MultiIndex.from_frame(uni[["code", "day"]]))
    news.index = key_to_idx.loc[list(zip(news["code"], news["day"], strict=True))].to_numpy()

    # 상장폐지·바 누락 점검 (t+20 미도래 제외).
    t20_possible = uni.groupby("day")["has_t20"].transform("max")
    missing = news[(~news["has_t20"]) & t20_possible.loc[news.index].to_numpy()]
    dl = set(delisted["code"].astype(str))
    checks["news_missing_t20_bar_while_day_has_t20"] = int(len(missing))
    checks["of_which_in_delisted_stocks"] = int(missing["code"].isin(dl).sum())
    checks["news_days_without_t20_yet"] = int((~t20_possible.loc[news.index]).sum())

    # ---- 매치드 널 대조군: 트레이드 = 뉴스 종목-일 전체 (한 번 뽑아 모든 지평에 재사용) ----
    strata = ["stratum_day", "tv_q", "ret_q"]
    controls = matched_control(
        news, uni, strata=strata, n_per=args.n_per, seed=args.seed, exclude_self=True
    )
    checks["null_unmatched"] = len(controls.attrs["unmatched"])
    checks["null_controls"] = int(len(controls))
    log(f"controls {len(controls)} unmatched {len(controls.attrs['unmatched'])}")

    # ---- 셀 통계 ----
    results: dict = {"checks": checks, "cells": [], "deciles": [], "appendix_scores": []}
    for split, members in SPLITS.items():
        base = news[news["split"].isin(members)]
        if len(base) == 0:
            continue
        variants = {"base": base, "no_limit_up": base[~base["limit_up"].fillna(False)]}
        for h in HORIZONS:
            lag = HORIZON_LAG[h]
            for vname, nb in variants.items():
                for cut in ("all_news", *CUTS):
                    sub = nb if cut == "all_news" else nb[nb[cut]]
                    st = cell_stats(sub[h], sub["day"], lag, args.n_boot, args.seed)
                    if st["n"] == 0:
                        continue
                    row = {"split": split, "horizon": h, "variant": vname, "cut": cut, **st}
                    if cut != "all_news":
                        row.update(excess_vs_all_news(sub, nb, h, args.n_boot, args.seed))
                    row.update(null_alpha(sub, uni, controls, h, args.n_boot, args.seed))
                    results["cells"].append(row)
            for dec in range(10):
                sub = base[base["decile"] == dec]
                st = cell_stats(sub[h], sub["day"], lag, args.n_boot, args.seed)
                row = {"split": split, "horizon": h, "decile": dec, **st}
                row.update(null_alpha(sub, uni, controls, h, args.n_boot, args.seed))
                results["deciles"].append(row)
        log(f"split {split} done ({len(base)} code-days)")

    # ---- 부록: 다른 점수 상위 10% (clean 구간, 종가→t+5 / t+20 / 야간) ----
    clean = news[news["split"].isin(SPLITS["clean"])].copy()
    for s in (SCORE_MAIN, *SCORE_APPX):
        n = clean.groupby("day")[s].transform("size")
        r = clean.groupby("day")[s].rank(method="first", ascending=False)
        top = clean[r <= np.ceil(0.10 * n)]
        for h in ("c0_o1", "c0_c5", "c0_c20"):
            st = cell_stats(top[h], top["day"], HORIZON_LAG[h], args.n_boot, args.seed)
            row = {"score": s, "horizon": h, "split": "clean", "cut": "top10", **st}
            row.update(null_alpha(top, uni, controls, h, args.n_boot, args.seed))
            results["appendix_scores"].append(row)

    # ---- 유니버스 자체 드리프트 (뉴스 무관 기준선) ----
    results["universe_drift"] = []
    for split, members in SPLITS.items():
        ds = set(news.loc[news["split"].isin(members), "day"])
        u = uni[uni["day"].isin(ds)]
        for h in HORIZONS:
            st = cell_stats(u[h], u["day"], HORIZON_LAG[h], args.n_boot, args.seed)
            results["universe_drift"].append({"split": split, "horizon": h, **st})

    results["config"] = {
        "scores": str(args.scores),
        "n_boot": args.n_boot,
        "n_per": args.n_per,
        "seed": args.seed,
        "strata": strata,
        "cuts": CUTS,
        "horizons": HORIZONS,
        "universe": UNIVERSE,
        "cost_bp": {
            "round_trip": COST_RT_BP,
            "overnight_auction": COST_OVN_BP,
            "repo_round_trip": COST_REPO_BP,
        },
        "bars_last_date": str(adj["date"].max()),
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    cells = pd.DataFrame(results["cells"])
    cells.to_csv(args.out / "cells.csv", index=False)
    pd.DataFrame(results["deciles"]).to_csv(args.out / "deciles.csv", index=False)
    (args.out / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=float)
    )
    news.drop(columns=["decide_t"]).to_csv(args.out / "news_code_days.csv", index=False)
    (args.out / "summary.md").write_text(render(results))
    log(f"wrote {args.out}")
    print(render(results))


# ------------------------------------------------------------------------ report
def _f(x, d=1):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{d}f}"


def render(res: dict) -> str:
    cells = pd.DataFrame(res["cells"])
    out = ["# news_multiday_drift — 요약 (bp, 비용 미차감; 괄호 = 날짜 클러스터 95% CI)", ""]
    out.append("## 점검")
    for k, v in res["checks"].items():
        out.append(f"- {k}: {v}")
    out.append("")
    for split in SPLITS:
        c = cells[(cells["split"] == split) & (cells["variant"] == "base")]
        if c.empty:
            continue
        out.append(
            f"## split = {split}  (n code-days all_news: {int(c[c['cut'] == 'all_news']['n'].max())})"
        )
        out.append(
            "| 지평 | 유니버스 | 뉴스 전체 | 상위10% | 널 대비 alpha(10%) | 상위5% | 상위1% | 널 대비 alpha(1%) | NW t(10%) | 음수월(10%) |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        ud = {r["horizon"]: r for r in res["universe_drift"] if r["split"] == split}
        for h in HORIZONS:
            r = {row["cut"]: row for _, row in c[c["horizon"] == h].iterrows()}
            if "all_news" not in r:
                continue
            a, t10, t5, t1 = r["all_news"], r.get("top10", {}), r.get("top5", {}), r.get("top1", {})
            out.append(
                f"| {HORIZON_LABEL[h]} | {_f(ud.get(h, {}).get('mean_bp'))} | {_f(a['mean_bp'])} [{_f(a['ci_low'], 0)},{_f(a['ci_high'], 0)}] "
                f"| {_f(t10.get('mean_bp'))} [{_f(t10.get('ci_low'), 0)},{_f(t10.get('ci_high'), 0)}] "
                f"| {_f(t10.get('alpha_bp'))} [{_f(t10.get('alpha_ci_low'), 0)},{_f(t10.get('alpha_ci_high'), 0)}] "
                f"| {_f(t5.get('mean_bp'))} | {_f(t1.get('mean_bp'))} [{_f(t1.get('ci_low'), 0)},{_f(t1.get('ci_high'), 0)}] "
                f"| {_f(t1.get('alpha_bp'))} [{_f(t1.get('alpha_ci_low'), 0)},{_f(t1.get('alpha_ci_high'), 0)}] "
                f"| {_f(t10.get('t_nw'), 2)} | {t10.get('neg_month_share', float('nan')):.2f} ({t10.get('month_signs', '')}) |"
            )
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    main()
