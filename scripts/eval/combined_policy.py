"""묶은 정책 — 09:01 종목 랭킹 × 매도 방법 × 빈 슬롯 재진입, 운영 한도로 시뮬레이션. simnode 전용.

    uv run --with scikit-learn python scripts/eval/combined_policy.py

보유 표본·특징·매도 규칙은 `exit_ml_news.py`를 그대로 쓴다. 추가한 것:
- 랭킹: 09:01 묶음(장 전·후 기사)을 무작위 / surprise 순 / persistence 순으로 슬롯에 넣는다.
- 매도 `ml_prob`: 분마다 "15:00까지 들고 가는 게 지금 파는 것보다 나을 확률"을 분류기로 예측,
  확률이 p 아래로 떨어지면 판다. p는 학습일 안에서 한 번 더 LODO로 고른다.
- 재진입: live = 슬롯이 비면 이후 신호(장중 포함) 진입 / open_only = 09:01 묶음만, 이후 진입 없음.
한도: 동시 5, 하루 10, (종목, 날짜)당 1회. 매도 규칙 파라미터와 ML 임계값은 빠진 날을 보지 않고 고른다.
지표: 하루 포트폴리오 합(거래별 ±10% 절삭 초과, 비용 포함), 6일 합과 이긴 날 수.
"""

from __future__ import annotations

import importlib.util
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg2
from sklearn.ensemble import HistGradientBoostingClassifier

from daytrade_it.shared.runtime_host import require_backtest_host

_p = Path(__file__).with_name("exit_ml_news.py")
_s = importlib.util.spec_from_file_location("exit_ml_news", _p)
em = importlib.util.module_from_spec(_s)
sys.modules[_s.name] = em
_s.loader.exec_module(em)

MAX_OPEN, MAX_PER_DAY = 5, 10
PERSIST = {"one_off": 0, "multi_quarter": 1, "structural": 2}
PROBS = (0.3, 0.35, 0.4, 0.45, 0.5)
TIE_SEEDS = range(1, 6)


def rule_exit(h: dict, name: str, p: tuple) -> tuple[int, float]:
    """(매도 체결 분 인덱스, 비용 포함 초과). 발동 안 하면 마지막 분(15:00)."""
    x, _, sells, final = h["feat"]
    col = {k: j for j, k in enumerate(em.FEATS)}
    base = h["i0"] + em.START_AFTER
    for j in range(len(x)):
        row = x[j]
        hit = False
        if name == "ma_cross":
            key = {"5/20": "ma5_ma20", "20/60": "ma20_ma60"}[p[0]]
            prev = x[j - 1][col[key]] if j else 1.0
            hit = prev >= 0 > row[col[key]] and row[col["r_entry"]] >= p[1]
        elif name == "vwap":
            hit = row[col["c_vwap"]] < -p[0] and row[col["r_entry"]] >= p[1]
        if hit:
            return base + j + 1, em.net_excess(sells[j])
    return len(h["ts"]) - 1, em.net_excess(final)


def pred_exit(h: dict, pred: np.ndarray, below: float) -> tuple[int, float]:
    _, _, sells, final = h["feat"]
    hit = np.nonzero(pred < below)[0]
    if len(hit):
        j = int(hit[0])
        return h["i0"] + em.START_AFTER + j + 1, em.net_excess(sells[j])
    return len(h["ts"]) - 1, em.net_excess(final)


def fit_clf(hs: list[dict]) -> HistGradientBoostingClassifier:
    x = np.vstack([h["feat"][0][:: em.SAMPLE_EVERY] for h in hs])
    y = np.concatenate([(h["feat"][1][:: em.SAMPLE_EVERY] > 0).astype(int) for h in hs])
    m = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40, random_state=0
    )
    return m.fit(x, y)


def choose_exits(train: list[dict], test: list[dict]) -> dict[str, dict[int, tuple[int, float]]]:
    """매도 방법별로 test 보유의 (체결 분, 값). 파라미터·임계값은 train에서만 고른다."""
    out: dict[str, dict[int, tuple[int, float]]] = {}
    out["hold"] = {id(h): (len(h["ts"]) - 1, em.net_excess(h["feat"][3])) for h in test}
    for name in ("ma_cross", "vwap"):
        grid = em.RULE_GRID[name]
        best = max(grid, key=lambda p: sum(em.clip(rule_exit(h, name, p)[1]) for h in train))
        out[name] = {id(h): rule_exit(h, name, best) for h in test}
    inner: dict[float, float] = defaultdict(float)
    for d2 in sorted({h["r"]["trade_day"] for h in train}):
        tr2 = [h for h in train if h["r"]["trade_day"] != d2]
        te2 = [h for h in train if h["r"]["trade_day"] == d2]
        m2 = fit_clf(tr2)
        for h in te2:
            p2 = m2.predict_proba(h["feat"][0])[:, 1]
            for p in PROBS:
                inner[p] += em.clip(pred_exit(h, p2, p)[1])
    p_best = max(PROBS, key=lambda p: inner[p])
    m = fit_clf(train)
    out["ml_prob"] = {
        id(h): pred_exit(h, m.predict_proba(h["feat"][0])[:, 1], p_best) for h in test
    }
    out["_p"] = p_best  # type: ignore[assignment]
    return out


def simulate_day(hs: list[dict], exits: dict, rank: str, reentry: str, seed: int) -> list[float]:
    rng = random.Random(seed)
    tie = {id(h): rng.random() for h in hs}

    def key(h: dict) -> tuple:
        r = h["r"]
        opening = r["session"] != "intraday"
        if opening and rank == "surprise":
            k = -(int(r.get("surprise") or 0))
        elif opening and rank == "persistence":
            k = -PERSIST.get(r.get("persistence"), 0)
        else:
            k = 0
        return (h["ts"][h["i0"]], k, tie[id(h)])

    open_pos: list[object] = []  # 청산 시각
    vals: list[float] = []
    first_ts = None
    for h in sorted(hs, key=key):
        t = h["ts"][h["i0"]]
        first_ts = first_ts or t
        if reentry == "open_only" and t != first_ts:
            continue
        if len(vals) >= MAX_PER_DAY:
            break
        open_pos = [x for x in open_pos if x > t]
        if len(open_pos) >= MAX_OPEN:
            continue
        idx, v = exits[id(h)]
        open_pos.append(h["ts"][idx])
        vals.append(em.clip(v))
    return vals


def main() -> None:
    require_backtest_host()
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5)
    conn.set_session(readonly=True)
    cur = conn.cursor()
    cur.execute("SET max_parallel_workers_per_gather = 0")
    hs = [h for h in em.load(cur) if True]
    for h in hs:
        h["feat"] = em.features(h)
    hs = [h for h in hs if len(h["feat"][0])]
    days = sorted({h["r"]["trade_day"] for h in hs})
    print(f"보유 후보 {len(hs)}, 거래일 {days}")

    results: dict[tuple, dict[str, list[float]]] = defaultdict(dict)
    p_used = []
    for d in days:
        train = [h for h in hs if h["r"]["trade_day"] != d]
        test = [h for h in hs if h["r"]["trade_day"] == d]
        exits_by = choose_exits(train, test)
        p_used.append(exits_by.pop("_p"))
        for ex_name, exits in exits_by.items():
            for rank in ("random", "surprise", "persistence"):
                for reentry in ("live", "open_only"):
                    sums, trades = [], []
                    for seed in TIE_SEEDS:
                        v = simulate_day(test, exits, rank, reentry, seed)
                        sums.append(sum(v))
                        trades.append(len(v))
                    results[(rank, ex_name, reentry)][d] = [
                        statistics.fmean(sums),
                        statistics.fmean(trades),
                    ]

    base = results[("random", "hold", "live")]
    print(f"\nml_prob가 빠진 날마다 고른 확률 기준: {p_used}")
    print("\n랭킹 | 매도 | 재진입 | 6일 합(%p) | 하루 평균 거래 | 현재 운영 대비 이긴 날 | 일별 합")
    rows = []
    for (rank, ex_name, reentry), per_day in results.items():
        total = sum(v[0] for v in per_day.values())
        n = statistics.fmean(v[1] for v in per_day.values())
        wins = sum(per_day[d][0] > base[d][0] + 1e-9 for d in days)
        rows.append((total, rank, ex_name, reentry, n, wins, [per_day[d][0] for d in days]))
    for total, rank, ex_name, reentry, n, wins, daily in sorted(rows, reverse=True):
        mark = "  ← 현재 운영" if (rank, ex_name, reentry) == ("random", "hold", "live") else ""
        print(
            f"{rank:>11} | {ex_name:>8} | {reentry:>9} | {total * 100:+7.2f} | {n:4.1f} | {wins}/{len(days)} | "
            + " ".join(f"{x * 100:+.1f}" for x in daily)
            + mark
        )


if __name__ == "__main__":
    main()
