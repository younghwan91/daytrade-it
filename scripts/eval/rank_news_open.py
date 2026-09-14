"""09:01 진입 후보(장 전·장 후 기사) 랭킹 — 상위 K개만 산다면 무엇을 고를까. simnode 전용.

    uv run python scripts/eval/rank_news_open.py [--k 5] [--cap 0.02|none]

운영은 09:01에 동시 보유 5개가 차고 15:00까지 보유한다(정책 최적화에서 가장 안정적인 부분).
그래서 조절할 수 있는 건 '어떤 5개'다. 특징은 09:01 판단 시점에 알 수 있는 것만 쓴다:
v2 판정 필드와 점수, 시가 갭(시가/전일 종가), 09:00 첫 분봉 수익·거래대금, 전일 수익률, 전일 거래대금, 시장.
검증은 거래일 하나씩 빼고(LODO) 학습 → 빠진 날의 후보를 점수순으로 상위 K개 매수(09:01 시가 → 15:00 시가).
비교: 무작위 K개(200회 평균), 전부, 사후 최선 K개. 수익은 비용 차감 후 시장 ETF 초과, ±10% 절삭.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import statistics
from pathlib import Path

import psycopg2

from daytrade_it.application.trading.exit_rules import FEE, SELL_TAX
from daytrade_it.shared.runtime_host import require_backtest_host

PAIRS = Path("data/eval/news_pairs.jsonl")
PROXY = {"KOSPI": "069500", "KOSDAQ": "229200"}
ORD = {
    "persistence": {"one_off": 0, "multi_quarter": 1, "structural": 2},
    "scale_vs_size": {"unknown": 0, "small": 1, "medium": 2, "large": 3},
    "freshness": {"repeat": 0, "follow_up": 1, "new": 2},
    "subject": {"mention": 0, "partial": 1, "main": 2},
}
FEATURES = [
    "score",
    "subject",
    "persistence",
    "scale_vs_size",
    "specificity",
    "surprise",
    "freshness",
    "sentiment_strength",
    "gap",
    "m1_ret",
    "m1_value_log",
    "prev_ret",
    "prev_value_log",
    "kosdaq",
]


def clip(x: float) -> float:
    return max(-0.1, min(0.1, x))


def build(cur, cap: float | None) -> list[dict]:
    rows = [json.loads(line) for line in PAIRS.open()]
    by_key: dict[tuple[str, str], dict] = {}
    for r in rows:
        if r["session"] == "intraday" or (r["score"] or 0) <= 0.15:
            continue
        k = (r["ticker"], r["trade_day"])
        if k not in by_key or (r["score"] or 0) > (by_key[k]["score"] or 0):
            by_key[k] = r  # 같은 종목·날짜에 기사가 여럿이면 점수 높은 판정 하나
    out = []
    for (code, day), r in by_key.items():
        cur.execute(
            "select ts, open, close, volume from minute_bars where code = %s and ts >= %s::date + time '09:00' "
            "and ts <= %s::date + time '15:00' order by ts",
            (code, day, day),
        )
        b = cur.fetchall()
        mk = "KOSDAQ" if r.get("market") == "코스닥" else "KOSPI"
        cur.execute(
            "select ts, open from minute_bars where code = %s and ts >= %s::date + time '09:01' "
            "and ts <= %s::date + time '15:00' order by ts",
            (PROXY[mk], day, day),
        )
        px = cur.fetchall()
        cur.execute(
            "select close, trade_value from daily_bars_adjusted where code = %s and date < %s order by date desc limit 2",
            (code, day),
        )
        d = cur.fetchall()
        if len(b) < 300 or len(px) < 300 or len(d) < 2 or b[0][0].time() != dt.time(9, 0):
            continue
        entry = next(((ts, float(o)) for ts, o, _c, _v in b if ts.time() >= dt.time(9, 1)), None)
        exit_ = next(((ts, float(o)) for ts, o, _c, _v in b if ts.time() >= dt.time(15, 0)), None)
        if entry is None or exit_ is None or entry[0].time() > dt.time(9, 5):
            continue
        prev_close, prev_value = float(d[0][0]), float(d[0][1] or 0)
        gap = float(b[0][1]) / prev_close - 1
        if cap is not None and entry[1] / prev_close - 1 > cap:
            continue
        net = exit_[1] * (1 - SELL_TAX - FEE) / (entry[1] * (1 + FEE)) - 1
        mkt = float(px[-1][1]) / float(px[0][1]) - 1
        f = {
            "score": r["score"],
            "specificity": int(r.get("specificity") or 0),
            "surprise": int(r.get("surprise") or 0),
            "sentiment_strength": int(r.get("sentiment_strength") or 0),
            "gap": gap,
            "m1_ret": float(b[0][2]) / float(b[0][1]) - 1,
            "m1_value_log": math.log1p(float(b[0][2]) * float(b[0][3])),
            "prev_ret": prev_close / float(d[1][0]) - 1,
            "prev_value_log": math.log1p(prev_value),
            "kosdaq": 1 if mk == "KOSDAQ" else 0,
        }
        for name, table in ORD.items():
            f[name] = table.get(r.get(name), 0)
        out.append(
            {
                "code": code,
                "day": day,
                "name": r.get("name"),
                "title": r["title"],
                "f": f,
                "excess": net - mkt,
                "net": net,
            }
        )
    return out


def standardize(train: list[dict], feats: list[str]) -> dict[str, tuple[float, float]]:
    st = {}
    for k in feats:
        v = [c["f"][k] for c in train]
        sd = statistics.pstdev(v) or 1.0
        st[k] = (statistics.fmean(v), sd)
    return st


def ridge(train: list[dict], feats: list[str], lam: float) -> dict[str, float]:
    """표준화 특징의 릿지 회귀(닫힌 해, numpy 없이 작은 차원이라 가우스 소거)."""
    st = standardize(train, feats)
    x = [[(c["f"][k] - st[k][0]) / st[k][1] for k in feats] for c in train]
    y = [clip(c["excess"]) for c in train]
    n = len(feats)
    a = [
        [sum(r[i] * r[j] for r in x) + (lam if i == j else 0.0) for j in range(n)] for i in range(n)
    ]
    bv = [sum(r[i] * yy for r, yy in zip(x, y, strict=True)) for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(a[r][i]))
        a[i], a[p], bv[i], bv[p] = a[p], a[i], bv[p], bv[i]
        for r in range(n):
            if r != i and a[i][i]:
                m = a[r][i] / a[i][i]
                a[r] = [a[r][j] - m * a[i][j] for j in range(n)]
                bv[r] -= m * bv[i]
    w = {k: bv[i] / a[i][i] for i, k in enumerate(feats)}
    return {"_st": st, **w}


def predict(model: dict, c: dict, feats: list[str]) -> float:
    st = model["_st"]
    return sum(model[k] * (c["f"][k] - st[k][0]) / st[k][1] for k in feats)


def single_feature(train: list[dict], k: str) -> float:
    """학습일에서 특징과 수익의 부호(스피어만 근사: 순위 상관)."""
    xs = sorted(range(len(train)), key=lambda i: train[i]["f"][k])
    ys = sorted(range(len(train)), key=lambda i: train[i]["excess"])
    rx = {i: r for r, i in enumerate(xs)}
    ry = {i: r for r, i in enumerate(ys)}
    n = len(train)
    return statistics.fmean((rx[i] - n / 2) * (ry[i] - n / 2) for i in range(n))


def main() -> None:
    require_backtest_host()
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cap", default="0.02", help="시가 진입가/전일 종가 상한, none이면 없음")
    a = ap.parse_args()
    cap = None if a.cap == "none" else float(a.cap)
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5)
    conn.set_session(readonly=True)
    cur = conn.cursor()
    cur.execute("SET max_parallel_workers_per_gather = 0")
    cands = build(cur, cap)
    days = sorted({c["day"] for c in cands})
    per_day = {d: sum(c["day"] == d for c in cands) for d in days}
    print(f"후보 {len(cands)}개 · 일별 {per_day} · K={a.k} cap={a.cap}")

    def pick_mean(chosen: list[dict]) -> float:
        return statistics.fmean(clip(c["excess"]) for c in chosen) if chosen else 0.0

    methods: dict[str, list[float]] = {"전부": [], "무작위K": [], "사후최선K": []}
    methods.update({f"릿지(λ={lam:g})": [] for lam in (1, 10, 50)})
    methods.update({f"단일:{k}": [] for k in FEATURES})
    per_day_n = []
    rng = random.Random(0)
    for d in days:
        test = [c for c in cands if c["day"] == d]
        train = [c for c in cands if c["day"] != d]
        k = min(a.k, len(test))
        per_day_n.append(len(test))
        methods["전부"].append(pick_mean(test))
        methods["무작위K"].append(
            statistics.fmean(pick_mean(rng.sample(test, k)) for _ in range(200))
        )
        methods["사후최선K"].append(pick_mean(sorted(test, key=lambda c: -c["excess"])[:k]))
        for lam in (1, 10, 50):
            m = ridge(train, FEATURES, lam)
            methods[f"릿지(λ={lam:g})"].append(
                pick_mean(sorted(test, key=lambda c: -predict(m, c, FEATURES))[:k])
            )
        for f in FEATURES:
            sign = 1 if single_feature(train, f) >= 0 else -1
            methods[f"단일:{f}"].append(
                pick_mean(sorted(test, key=lambda c: -sign * c["f"][f])[:k])
            )

    print(
        "\n빠진 날 기준 거래당 절삭 초과수익 %p (일별 평균, 괄호는 일별 값) — 후보 ≤ K인 날은 모든 방법이 같다"
    )
    print("일자:", days, "후보수:", per_day_n)
    base = methods["무작위K"]
    for name, v in sorted(methods.items(), key=lambda kv: -statistics.fmean(kv[1])):
        wins = sum(x > y + 1e-12 for x, y in zip(v, base, strict=True))
        print(
            f"{name:>22} {statistics.fmean(v) * 100:+6.2f}  vs 무작위 이긴 날 {wins}/{len(days)}  ({' '.join(f'{x * 100:+.1f}' for x in v)})"
        )

    m = ridge(cands, FEATURES, 10)
    print(
        "\n전체 데이터 릿지(λ=10) 계수(표준화, 참고):", {k: round(m[k] * 100, 2) for k in FEATURES}
    )


if __name__ == "__main__":
    main()
