"""뉴스 쌍의 진입 후 수익 경로를 특성별로 쪼개 '보유 시간이 갈리는 차원'을 찾는다 — simnode 전용.

    uv run python scripts/eval/news_path_dims.py [--universe buy|pos|all]

쌍마다 진입 분 시가에 사서 h분 뒤(또는 15:00) 시가에 판다고 치고, 같은 구간 시장
ETF(KOSPI=069500, KOSDAQ=229200) 수익을 뺀 초과수익을 계산한다. 비용은 왕복으로 뺀다.
특성 값마다 호라이즌별 평균·중앙값과, 거래일 하나씩 빼고 고른 최적 호라이즌이
얼마나 일치하는지(LODO)를 출력한다. 탐색용이다 — 여기서 찾은 차원은 전진 데이터로 확인한다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import psycopg2

from daytrade_it.application.trading.exit_rules import FEE, SELL_TAX
from daytrade_it.shared.runtime_host import require_backtest_host

PAIRS = Path("data/eval/news_pairs.jsonl")
OUT = Path("data/eval/news_paths.jsonl")
HORIZONS = (5, 15, 30, 60, 120, 240, "close")
FLATTEN = dt.time(15, 0)
PROXY = {"KOSPI": "069500", "KOSDAQ": "229200"}
COST = SELL_TAX + 2 * FEE


def load_bars(cur, code: str, day: str) -> list[tuple]:
    cur.execute(
        "select ts, open, close from minute_bars where code = %s and ts >= %s::date + time '09:00' "
        "and ts < %s::date + time '15:31' order by ts",
        (code, day, day),
    )
    return cur.fetchall()


def open_at(bars: list[tuple], t: dt.datetime) -> tuple[dt.datetime, float] | None:
    for ts, o, _c in bars:
        if ts >= t:
            return ts, float(o)
    return None


def pair_path(bars, proxy, entry_ts: dt.datetime) -> dict | None:
    e = open_at(bars, entry_ts)
    pe = open_at(proxy, entry_ts)
    if (
        e is None
        or pe is None
        or e[0] - entry_ts > dt.timedelta(minutes=5)
        or e[0].time() >= FLATTEN
    ):
        return None
    out = {}
    close_t = dt.datetime.combine(e[0].date(), FLATTEN)
    for h in HORIZONS:
        t = close_t if h == "close" else min(e[0] + dt.timedelta(minutes=h), close_t)
        x, px = open_at(bars, t), open_at(proxy, t)
        if x is None or px is None or x[0] - t > dt.timedelta(minutes=10):
            return None
        out[str(h)] = (x[1] / e[1] - 1 - COST) - (px[1] / pe[1] - 1)
    return {"entry_price": e[1], "paths": out}


def main() -> None:
    require_backtest_host()
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["buy", "pos", "all"], default="buy")
    ap.add_argument("--min-n", type=int, default=15)
    a = ap.parse_args()

    rows = [json.loads(line) for line in PAIRS.open()]
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5)
    conn.set_session(readonly=True)
    cur = conn.cursor()
    cur.execute("SET max_parallel_workers_per_gather = 0")
    bars_cache: dict[tuple[str, str], list] = {}

    def bars(code: str, day: str) -> list:
        if (code, day) not in bars_cache:
            bars_cache[(code, day)] = load_bars(cur, code, day)
        return bars_cache[(code, day)]

    kept, drop = [], Counter()
    for r in rows:
        mk = (
            "KOSDAQ"
            if (r.get("market") or "").upper().startswith("KOSDAQ") or r.get("market") == "코스닥"
            else "KOSPI"
        )
        b = bars(r["ticker"], r["trade_day"])
        if len(b) < 300:
            drop["no_bars"] += 1
            continue
        cur.execute(
            "select close from daily_bars_adjusted where code = %s and date < %s order by date desc limit 1",
            (r["ticker"], r["trade_day"]),
        )
        prev = cur.fetchone()
        p = pair_path(b, bars(PROXY[mk], r["trade_day"]), dt.datetime.fromisoformat(r["entry_ts"]))
        if p is None or not prev:
            drop["no_entry_or_prev"] += 1
            continue
        r["pre_move"] = p["entry_price"] / float(prev[0]) - 1
        r["paths"] = p["paths"]
        kept.append(r)
    with OUT.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"pairs={len(rows)} kept={len(kept)} dropped={dict(drop)}")

    if a.universe == "buy":
        uni = [r for r in kept if (r["score"] or 0) > 0.15 and r["pre_move"] <= 0.02]
    elif a.universe == "pos":
        uni = [r for r in kept if (r["score"] or 0) > 0]
    else:
        uni = kept
    print(f"universe={a.universe} n={len(uni)} days={len({r['trade_day'] for r in uni})}")

    def bucket(r: dict, dim: str):
        v = r.get(dim)
        if dim == "pre_move":
            return (
                "<-3%"
                if v < -0.03
                else "-3~-1%"
                if v < -0.01
                else "-1~+1%"
                if v <= 0.01
                else ">+1%"
            )
        if dim == "score":
            return (
                None if v is None else "<=0.15" if v <= 0.15 else "0.15~0.4" if v <= 0.4 else ">0.4"
            )
        if dim == "entry_hour":
            h = int(r["entry_ts"][11:13])
            return "09" if h == 9 else "10-11" if h < 12 else "12+"
        return v

    dims = [
        "session",
        "entry_hour",
        "subject",
        "material_type",
        "persistence",
        "freshness",
        "surprise",
        "scale_vs_size",
        "specificity",
        "sentiment_strength",
        "reports_price_move",
        "score",
        "pre_move",
        "market",
        "category",
        "source",
    ]
    hs = [str(h) for h in HORIZONS]

    def best_h(rs: list[dict]) -> str:
        return max(
            hs, key=lambda h: statistics.fmean(max(-0.1, min(0.1, r["paths"][h])) for r in rs)
        )

    print(
        "\n각 칸: 절삭(±10%) 평균 초과수익 %. best=전체 최적 / LODO=거래일 하나씩 뺐을 때 같은 best가 나온 비율"
    )
    print(
        f"{'dim':>18} {'value':>14} {'n':>4} " + " ".join(f"{h:>6}" for h in hs) + "   best  LODO"
    )
    for dim in dims:
        groups: dict = defaultdict(list)
        for r in uni:
            k = bucket(r, dim)
            if k is not None:
                groups[k].append(r)
        for k, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if len(rs) < a.min_n:
                continue
            means = [statistics.fmean(max(-0.1, min(0.1, r["paths"][h])) for r in rs) for h in hs]
            b = best_h(rs)
            days = sorted({r["trade_day"] for r in rs})
            agree = [
                best_h([r for r in rs if r["trade_day"] != d]) == b
                for d in days
                if any(r["trade_day"] != d for r in rs)
            ]
            print(
                f"{dim:>18} {str(k)[:14]:>14} {len(rs):4d} "
                + " ".join(f"{m * 100:+6.2f}" for m in means)
                + f"   {b:>5}  {sum(agree) / len(agree):.0%}"
            )


if __name__ == "__main__":
    main()
