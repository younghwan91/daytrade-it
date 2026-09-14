"""보유 중 매도 판단 — 분봉 규칙(이평선·VWAP·트레일링) vs 머신러닝, 거래일 LODO. simnode 전용.

    uv run --with scikit-learn python scripts/eval/exit_ml_news.py

보유 표본: 운영 대시보드 기사 BUY(v2 > 0.15), (종목, 날짜)당 1회, 운영 진입 시각
(장 전·후 09:01 / 장중 발행 + 18분, 14:30까지). 평가는 두 묶음:
- open: 09:01에 들어가는 장 전·후 기사 종목(운영 슬롯을 실제로 채우는 것)
- all: 전체 BUY
매도는 판단 분의 다음 분 시가. 판단하지 않으면 15:00 시가. 시장 ETF 초과, ±10% 절삭.
머신러닝 타깃: '지금(다음 분 시가) 팔 때 대비 15:00까지 들고 갈 때의 초과수익'. 매도 비용은 어느 쪽이든 같아서 뺀다.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg2
from sklearn.ensemble import HistGradientBoostingRegressor

from daytrade_it.application.trading.exit_rules import FEE, SELL_TAX
from daytrade_it.shared.runtime_host import require_backtest_host

PAIRS = Path("data/eval/news_pairs.jsonl")
PROXY = {"KOSPI": "069500", "KOSDAQ": "229200"}
FLAT = dt.time(15, 0)
START_AFTER = 5  # 진입 후 최소 보유 분
SAMPLE_EVERY = 5  # 학습 행은 5분마다(자기상관 줄이기)
FEATS = [
    "el",
    "tl",
    "r_entry",
    "dd",
    "c_ma5",
    "ma5_ma20",
    "ma20_ma60",
    "c_vwap",
    "r5",
    "r15",
    "r30",
    "vol10",
    "m_entry",
    "m15",
    "pre",
    "after",
]


def clip(x: float) -> float:
    return max(-0.1, min(0.1, x))


def entry_time(r: dict) -> dt.datetime | None:
    day = dt.date.fromisoformat(r["trade_day"])
    if r["session"] != "intraday":
        t = dt.datetime.combine(day, dt.time(9, 1))
    else:
        t = dt.datetime.fromisoformat(r["published_kst"]) + dt.timedelta(minutes=18)
        t = t.replace(second=0) + dt.timedelta(minutes=1 if t.second else 0)
        t = max(t, dt.datetime.combine(day, dt.time(9, 1)))
    return t if t.time() <= dt.time(14, 30) else None


def load(cur) -> list[dict]:
    seen, out = set(), []
    rows = sorted((json.loads(line) for line in PAIRS.open()), key=lambda r: r["published_kst"])
    for r in rows:
        if (r["score"] or 0) <= 0.15 or (r["ticker"], r["trade_day"]) in seen:
            continue
        et = entry_time(r)
        if et is None:
            continue
        day = r["trade_day"]
        cur.execute(
            "select ts, open, high, low, close, volume from minute_bars where code = %s "
            "and ts >= %s::date + time '09:00' and ts <= %s::date + time '15:00' order by ts",
            (r["ticker"], day, day),
        )
        b = (
            np.array([[float(x) for x in row[1:]] for row in cur.fetchall()])
            if cur.rowcount
            else None
        )
        cur.execute(
            "select ts from minute_bars where code = %s and ts >= %s::date + time '09:00' "
            "and ts <= %s::date + time '15:00' order by ts",
            (r["ticker"], day, day),
        )
        ts = [row[0] for row in cur.fetchall()]
        mk = "KOSDAQ" if r.get("market") == "코스닥" else "KOSPI"
        cur.execute(
            "select ts, open from minute_bars where code = %s and ts >= %s::date + time '09:00' "
            "and ts <= %s::date + time '15:00' order by ts",
            (PROXY[mk], day, day),
        )
        px = dict(cur.fetchall())
        if b is None or len(ts) < 300 or ts[-1].time() != FLAT:
            continue
        i0 = next((i for i, t in enumerate(ts) if t >= et), None)
        if i0 is None or ts[i0] - et > dt.timedelta(minutes=5) or ts[i0] not in px:
            continue
        seen.add((r["ticker"], day))
        out.append(
            {"r": r, "ts": ts, "b": b, "px": [float(px.get(t, np.nan)) for t in ts], "i0": i0}
        )
    return out


def features(h: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """분 i(i0+START_AFTER..마지막-1)마다 특징, 타깃(보유-매도 초과), 매도 시 초과수익."""
    b, px, i0, r = h["b"], np.array(h["px"]), h["i0"], h["r"]
    o, hi, c, v = b[:, 0], b[:, 1], b[:, 3], b[:, 4]
    n = len(c)
    px = np.where(np.isnan(px), np.nan, px)
    px = np.array(
        [
            px[i]
            if not np.isnan(px[i])
            else next((px[j] for j in range(i, -1, -1) if not np.isnan(px[j])), np.nan)
            for i in range(n)
        ]
    )
    e, pe = o[i0], px[i0]
    final = o[-1] / e - px[-1] / pe  # 15:00까지 보유 시 (비용 전) 초과
    cum_pv = np.cumsum(c * v)
    cum_v = np.cumsum(v) + 1e-9
    rows, ys, sells = [], [], []
    for i in range(i0 + START_AFTER, n - 1):

        def ma(k: int, i: int = i) -> float:
            return float(c[max(0, i - k + 1) : i + 1].mean())

        vol_recent = v[max(i0, i - 9) : i + 1].mean()
        vol_base = v[i0 : i + 1].mean() + 1e-9
        f = {
            "el": i - i0,
            "tl": n - 1 - i,
            "r_entry": c[i] / e - 1,
            "dd": c[i] / hi[i0 : i + 1].max() - 1,
            "c_ma5": c[i] / ma(5) - 1,
            "ma5_ma20": ma(5) / ma(20) - 1,
            "ma20_ma60": ma(20) / ma(60) - 1,
            "c_vwap": c[i] / (cum_pv[i] / cum_v[i]) - 1,
            "r5": c[i] / c[max(0, i - 5)] - 1,
            "r15": c[i] / c[max(0, i - 15)] - 1,
            "r30": c[i] / c[max(0, i - 30)] - 1,
            "vol10": vol_recent / vol_base,
            "m_entry": px[i] / pe - 1,
            "m15": px[i] / px[max(0, i - 15)] - 1,
            "pre": 1.0 if r["session"] == "pre_market" else 0.0,
            "after": 1.0 if r["session"] == "after_close" else 0.0,
        }
        sell_ex = o[i + 1] / e - px[i + 1] / pe
        rows.append([f[k] for k in FEATS])
        ys.append(final - sell_ex)
        sells.append(sell_ex)
    return np.array(rows), np.array(ys), np.array(sells), final


def net_excess(gross_excess: float) -> float:
    return gross_excess - SELL_TAX - 2 * FEE


def policy_rule(h: dict, name: str, p: tuple) -> float:
    """규칙 매도 시 초과수익(비용 포함)."""
    b, i0 = h["b"], h["i0"]
    x, _, sells, final = h["feat"]
    col = {k: j for j, k in enumerate(FEATS)}
    for j in range(len(x)):
        row = x[j]
        if name == "ma_cross":
            fast_slow, min_gain = p
            key = {"5/20": "ma5_ma20", "20/60": "ma20_ma60"}[fast_slow]
            prev = x[j - 1][col[key]] if j else 1.0
            if prev >= 0 > row[col[key]] and row[col["r_entry"]] >= min_gain:
                return net_excess(sells[j])
        elif name == "vwap":
            gap, min_gain = p
            if row[col["c_vwap"]] < -gap and row[col["r_entry"]] >= min_gain:
                return net_excess(sells[j])
        elif name == "trail":
            if row[col["dd"]] <= -p[0]:
                return net_excess(sells[j])
    del b, i0
    return net_excess(final)


RULE_GRID = {
    "ma_cross": [(fs, g) for fs in ("5/20", "20/60") for g in (-1.0, 0.0, 0.01, 0.03)],
    "vwap": [(gap, g) for gap in (0.0, 0.005, 0.01, 0.02) for g in (-1.0, 0.0, 0.02)],
    "trail": [(x,) for x in (0.02, 0.03, 0.05, 0.08)],
}


def fit_gbm(hs: list[dict]) -> HistGradientBoostingRegressor:
    x = np.vstack([h["feat"][0][::SAMPLE_EVERY] for h in hs])
    y = np.concatenate([np.clip(h["feat"][1][::SAMPLE_EVERY], -0.1, 0.1) for h in hs])
    m = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40, random_state=0
    )
    return m.fit(x, y)


def policy_ml(h: dict, pred: np.ndarray, thr: float) -> float:
    _, _, sells, final = h["feat"]
    hit = np.nonzero(pred < thr)[0]
    return net_excess(sells[hit[0]]) if len(hit) else net_excess(final)


THRS = (-0.03, -0.02, -0.01, -0.005, 0.0)


def main() -> None:
    require_backtest_host()
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5)
    conn.set_session(readonly=True)
    cur = conn.cursor()
    cur.execute("SET max_parallel_workers_per_gather = 0")
    hs = load(cur)
    for h in hs:
        h["feat"] = features(h)
    hs = [h for h in hs if len(h["feat"][0])]
    days = sorted({h["r"]["trade_day"] for h in hs})
    print(
        f"보유 표본 {len(hs)} (09:01 진입 {sum(h['r']['session'] != 'intraday' for h in hs)}), 거래일 {days}"
    )

    res: dict[str, dict[str, list]] = defaultdict(
        lambda: defaultdict(list)
    )  # method -> day -> [(open?, value)]
    chosen: dict[str, list] = defaultdict(list)
    for d in days:
        train = [h for h in hs if h["r"]["trade_day"] != d]
        test = [h for h in hs if h["r"]["trade_day"] == d]
        for h in test:
            res["hold"][d].append((h, net_excess(h["feat"][3])))
        for name, grid in RULE_GRID.items():
            best = max(grid, key=lambda p: sum(clip(policy_rule(h, name, p)) for h in train))
            chosen[name].append(best)
            for h in test:
                res[name][d].append((h, policy_rule(h, name, best)))
        # ML: 임계값은 학습일 안에서 다시 LODO로 고른다
        inner = defaultdict(float)
        for d2 in sorted({h["r"]["trade_day"] for h in train}):
            tr2 = [h for h in train if h["r"]["trade_day"] != d2]
            te2 = [h for h in train if h["r"]["trade_day"] == d2]
            m2 = fit_gbm(tr2)
            for h in te2:
                p2 = m2.predict(h["feat"][0])
                for t in THRS:
                    inner[t] += clip(policy_ml(h, p2, t))
        thr = max(THRS, key=lambda t: inner[t])
        chosen["ml"].append(thr)
        m = fit_gbm(train)
        for h in test:
            res["ml"][d].append((h, policy_ml(h, m.predict(h["feat"][0]), thr)))

    for group in ("open", "all"):
        print(f"\n[{group}] 방법 | 거래당 절삭 초과(비용 포함) | hold 대비 이긴 날 | 일별")
        base = {
            d: statistics.fmean(
                clip(v)
                for h, v in res["hold"][d]
                if group == "all" or h["r"]["session"] != "intraday"
            )
            for d in days
            if any(group == "all" or h["r"]["session"] != "intraday" for h, _ in res["hold"][d])
        }
        for meth in ("hold", "trail", "ma_cross", "vwap", "ml"):
            vals, daily = [], {}
            for d in base:
                vs = [
                    clip(v)
                    for h, v in res[meth][d]
                    if group == "all" or h["r"]["session"] != "intraday"
                ]
                vals += vs
                daily[d] = statistics.fmean(vs)
            wins = sum(daily[d] > base[d] + 1e-12 for d in base)
            print(
                f"{meth:>9} | {statistics.fmean(vals) * 100:+.2f}%p (n={len(vals)}) | {wins}/{len(base)} | "
                + " ".join(f"{daily[d] * 100:+.1f}" for d in base)
            )
    print("\n빠진 날마다 학습일에서 고른 파라미터:", {k: v for k, v in chosen.items()})
    m = fit_gbm(hs)
    try:
        from sklearn.inspection import permutation_importance  # noqa: PLC0415

        x = np.vstack([h["feat"][0][::SAMPLE_EVERY] for h in hs])
        y = np.concatenate([np.clip(h["feat"][1][::SAMPLE_EVERY], -0.1, 0.1) for h in hs])
        imp = permutation_importance(m, x, y, n_repeats=5, random_state=0)
        print(
            "전체 적합 특징 중요도(참고):",
            sorted(zip(FEATS, np.round(imp.importances_mean * 1e4, 2)), key=lambda kv: -kv[1])[:8],
        )
    except Exception as exc:  # noqa: BLE001
        print("중요도 계산 실패:", exc)


if __name__ == "__main__":
    main()
