"""불타기(오를 때 1주 추가) — surprise 랭킹 + VWAP 매도 + live 재진입 위에서 LODO. simnode 전용.

    uv run --with scikit-learn python scripts/eval/pyramid_policy.py

추가 조건(보유 중 1회, 추가 체결은 판단 다음 분 시가, 청산은 원래 포지션과 같은 분):
- trend: 진입 후 +g% 이상 & 가격 > VWAP & 5분선 > 20분선
- breakout: 진입 후 +g% 이상 & 진입 후 최고가 갱신 & 최근 10분 거래량 ≥ 진입 후 평균 × v
g, v는 학습일에서만 고른다. 지표: 하루 합(주식 1주당 ±10% 절삭 초과, 비용 포함)과 투입 주식 수당 수익.
추가 매수는 동시 보유 한도(종목 수)를 늘리지 않지만 현금은 더 쓴다 — 결과에 주식 수를 같이 적는다.
"""

from __future__ import annotations

import importlib.util
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import psycopg2

from daytrade_it.shared.runtime_host import require_backtest_host

_p = Path(__file__).with_name("combined_policy.py")
_s = importlib.util.spec_from_file_location("combined_policy", _p)
cp = importlib.util.module_from_spec(_s)
sys.modules[_s.name] = cp
_s.loader.exec_module(cp)
em = cp.em

GRID = {
    "trend": [(g,) for g in (0.01, 0.02, 0.03, 0.05)],
    "breakout": [(g, v) for g in (0.01, 0.02, 0.03) for v in (1.0, 1.5, 2.0)],
}


def add_on(h: dict, exit_idx: int, name: str, p: tuple) -> float | None:
    """추가 1주의 비용 포함 초과수익, 조건 불충족이면 None."""
    x, _, sells, _ = h["feat"]
    col = {k: j for j, k in enumerate(em.FEATS)}
    base = h["i0"] + em.START_AFTER
    b, px = h["b"], h["px"]
    hi_so_far = b[h["i0"], 1]
    for j in range(len(x)):
        i = base + j
        if i + 1 >= exit_idx:
            return None
        row = x[j]
        new_high = b[i, 1] > hi_so_far
        hi_so_far = max(hi_so_far, b[i, 1])
        if name == "trend":
            hit = (
                row[col["r_entry"]] >= p[0]
                and row[col["c_vwap"]] > 0
                and row[col["c_ma5"]] > -1
                and row[col["ma5_ma20"]] > 0
            )
        else:
            hit = row[col["r_entry"]] >= p[0] and new_high and row[col["vol10"]] >= p[1]
        if hit:
            a, pa = b[i + 1, 0], px[i + 1]
            xo, pxo = b[exit_idx, 0], px[exit_idx]
            return em.net_excess(xo / a - pxo / pa)
    return None


def day_values(test: list[dict], exits: dict, adds: dict | None, seed: int) -> tuple[float, int]:
    rng = random.Random(seed)
    tie = {id(h): rng.random() for h in test}

    def key(h: dict) -> tuple:
        r = h["r"]
        k = -(int(r.get("surprise") or 0)) if r["session"] != "intraday" else 0
        return (h["ts"][h["i0"]], k, tie[id(h)])

    open_pos, total, shares, n = [], 0.0, 0, 0
    for h in sorted(test, key=key):
        t = h["ts"][h["i0"]]
        if n >= cp.MAX_PER_DAY:
            break
        open_pos = [x for x in open_pos if x > t]
        if len(open_pos) >= cp.MAX_OPEN:
            continue
        idx, v = exits[id(h)]
        open_pos.append(h["ts"][idx])
        total += em.clip(v)
        shares += 1
        n += 1
        if adds is not None and adds.get(id(h)) is not None:
            total += em.clip(adds[id(h)])
            shares += 1
    return total, shares


def main() -> None:
    require_backtest_host()
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5)
    conn.set_session(readonly=True)
    cur = conn.cursor()
    cur.execute("SET max_parallel_workers_per_gather = 0")
    hs = em.load(cur)
    for h in hs:
        h["feat"] = em.features(h)
    hs = [h for h in hs if len(h["feat"][0])]
    cp.pre_moves(cur, hs)
    cap = os.environ.get("PRE_MOVE_CAP")
    if cap:
        hs = [h for h in hs if h["pre_move"] is not None and h["pre_move"] <= float(cap)]
    days = sorted({h["r"]["trade_day"] for h in hs})
    print(f"추격 금지 상한 {cap or '없음'}, 보유 후보 {len(hs)}")

    res: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    chosen = defaultdict(list)
    for d in days:
        train = [h for h in hs if h["r"]["trade_day"] != d]
        test = [h for h in hs if h["r"]["trade_day"] == d]
        ex_train = cp.choose_exits([h for h in train if h["r"]["trade_day"] != d], train)["vwap"]
        ex_test = cp.choose_exits(train, test)["vwap"]
        variants = {"no_add": None}
        for name, grid in GRID.items():
            best = max(
                grid,
                key=lambda p, name=name: sum(
                    em.clip(a)
                    for h in train
                    if (a := add_on(h, ex_train[id(h)][0], name, p)) is not None
                ),
            )
            chosen[name].append(best)
            variants[name] = {id(h): add_on(h, ex_test[id(h)][0], name, best) for h in test}
        for vname, adds in variants.items():
            out = [day_values(test, ex_test, adds, s) for s in cp.TIE_SEEDS]
            res[vname][d] = (
                statistics.fmean(o[0] for o in out),
                statistics.fmean(o[1] for o in out),
            )

    print(f"거래일 {days}")
    print("학습일에서 고른 추가 조건:", dict(chosen))
    base = res["no_add"]
    print(
        "\n추가 매수 | 6일 합(%p) | 하루 평균 주식 수 | 주식당(%p) | 추가 없음 대비 이긴 날 | 일별 합"
    )
    for vname, per_day in res.items():
        total = sum(v[0] for v in per_day.values())
        shares = sum(v[1] for v in per_day.values())
        wins = sum(per_day[d][0] > base[d][0] + 1e-9 for d in days)
        print(
            f"{vname:>9} | {total * 100:+7.2f} | {shares / len(days):4.1f} | {total / shares * 100:+.2f} | {wins}/{len(days)} | "
            + " ".join(f"{per_day[d][0] * 100:+.1f}" for d in days)
        )


if __name__ == "__main__":
    main()
