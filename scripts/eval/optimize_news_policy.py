"""뉴스 매매 정책 최적화 — 거래일 하나씩 빼고(LODO) 베이즈 최적화, 빠진 날로 평가. simnode 전용.

    uv run --with optuna python scripts/eval/optimize_news_policy.py --trials 300

표본은 운영 대시보드 기사(`data/eval/news_pairs.jsonl`). 운영과 같게 맞춘 것:
- 진입 시각: 장중 기사는 발행 + `--delay`분(09-14 데몬 실측 중앙값 18), 장 전·후 기사는 09:01.
  14:30 이후 진입 없음(`TradingLimits.entry_end`).
- 판정: v2(Haiku) 점수 > 0.15, 1주, (종목, 거래일)당 첫 진입만.
- 한도: 하루 진입 10건, 동시 보유 5개(`TradingLimits`), 신호 시각 순서대로.
- 청산: 15:00 강제. 손절은 그 봉 안에서 min(시가, 손절가), 시간 청산은 해당 분 시가.
- 비용: 매도세 0.20% + 수수료 편도 0.015%.
남는 차이: 체결을 분봉 시가로 본다(운영은 매도호가 지정가·5분 TTL, 매수호가 매도).

목적함수는 학습일들의 거래당 절삭(±10%) 초과수익 합(= 1주씩 산 초과손익의 합과 비례).
시장 ETF(KOSPI 069500 / KOSDAQ 229200)의 같은 구간 수익을 뺀다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pickle
import statistics
from dataclasses import dataclass
from pathlib import Path

import optuna
import psycopg2

from daytrade_it.application.trading.exit_rules import FEE, SELL_TAX
from daytrade_it.shared.runtime_host import require_backtest_host

PAIRS = Path("data/eval/news_pairs.jsonl")
CACHE = Path("data/eval/news_policy_bars.pkl")
PROXY = {"KOSPI": "069500", "KOSDAQ": "229200"}
ENTRY_OPEN = dt.time(9, 1)
ENTRY_END = dt.time(14, 30)
FLATTEN = dt.time(15, 0)
MAX_PER_DAY = 10
MAX_OPEN = 5
CLOSE = 999  # hold 분 값이 이 이상이면 15:00까지
SESSIONS = ("pre_market", "intraday", "after_close")


@dataclass
class Pair:
    key: str
    ticker: str
    day: str
    session: str
    entry: dt.datetime
    score: float
    persistence: str | None
    prev_close: float
    bars: list[tuple]  # (ts, open, high, low)
    proxy: list[tuple]  # (ts, open)


def live_entry(r: dict, delay: int) -> dt.datetime | None:
    day = dt.date.fromisoformat(r["trade_day"])
    if r["session"] != "intraday":
        t = dt.datetime.combine(day, ENTRY_OPEN)
    else:
        t = dt.datetime.fromisoformat(r["published_kst"]) + dt.timedelta(minutes=delay)
        if t.second:
            t = t.replace(second=0) + dt.timedelta(minutes=1)
        t = max(t, dt.datetime.combine(day, ENTRY_OPEN))
    return t if t.time() <= ENTRY_END else None


def load_pairs(delay: int) -> list[Pair]:
    raw = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5)
    conn.set_session(readonly=True)
    cur = conn.cursor()
    cur.execute("SET max_parallel_workers_per_gather = 0")

    def bars(code: str, day: str) -> list[tuple]:
        k = (code, day)
        if k not in raw:
            cur.execute(
                "select ts, open, high, low from minute_bars where code = %s "
                "and ts >= %s::date + time '09:00' and ts < %s::date + time '15:31' order by ts",
                (code, day, day),
            )
            raw[k] = [(ts, float(o), float(h), float(lo)) for ts, o, h, lo in cur.fetchall()]
        return raw[k]

    out = []
    for line in PAIRS.open():
        r = json.loads(line)
        if (r["score"] or 0) <= 0.15:
            continue
        entry = live_entry(r, delay)
        b = bars(r["ticker"], r["trade_day"])
        if entry is None or len(b) < 300:
            continue
        mk = "KOSDAQ" if r.get("market") == "코스닥" else "KOSPI"
        k = ("prev", r["ticker"], r["trade_day"])
        if k not in raw:
            cur.execute(
                "select close from daily_bars_adjusted where code = %s and date < %s order by date desc limit 1",
                (r["ticker"], r["trade_day"]),
            )
            row = cur.fetchone()
            raw[k] = float(row[0]) if row else None
        if raw[k] is None:
            continue
        px = [(ts, o) for ts, o, _h, _l in bars(PROXY[mk], r["trade_day"])]
        out.append(
            Pair(
                r["key"],
                r["ticker"],
                r["trade_day"],
                r["session"],
                entry,
                r["score"],
                r.get("persistence"),
                raw[k],
                b,
                px,
            )
        )
    CACHE.write_bytes(pickle.dumps(raw))
    return out


def first_at(seq: list[tuple], t: dt.datetime) -> int | None:
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        if seq[mid][0] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(seq) else None


def trade(p: Pair, hold: int, stop: float) -> tuple[dt.datetime, float, float] | None:
    """(청산 시각, 순수익률, 초과수익률) 또는 진입 불가 시 None."""
    i = first_at(p.bars, p.entry)
    pi = first_at(p.proxy, p.entry)
    if i is None or pi is None or p.bars[i][0] - p.entry > dt.timedelta(minutes=5):
        return None
    t_entry, e = p.bars[i][0], p.bars[i][1]
    close_t = dt.datetime.combine(t_entry.date(), FLATTEN)
    if t_entry >= close_t:
        return None
    t_exit = close_t if hold >= CLOSE else min(t_entry + dt.timedelta(minutes=hold), close_t)
    stop_px = e * (1 - stop)
    x_ts, x_px = None, None
    for ts, o, _h, lo in p.bars[i:]:
        if ts >= t_exit:
            x_ts, x_px = ts, o
            break
        if lo <= stop_px:
            x_ts, x_px = ts, min(o, stop_px)
            break
    if x_ts is None:
        x_ts, x_px = p.bars[-1][0], p.bars[-1][1]
    pj = first_at(p.proxy, x_ts)
    proxy_ret = (p.proxy[pj][1] / p.proxy[pi][1] - 1) if pj is not None else 0.0
    net = x_px * (1 - SELL_TAX - FEE) / (e * (1 + FEE)) - 1
    return x_ts, net, net - proxy_ret


def pre_move(p: Pair) -> float | None:
    i = first_at(p.bars, p.entry)
    return None if i is None else p.bars[i][1] / p.prev_close - 1


def run_policy(pairs: list[Pair], params: dict) -> list[dict]:
    """운영 한도를 날짜별 시간순으로 적용해 실제로 들어간 거래만 돌려준다."""
    done = []
    by_day: dict[str, list[Pair]] = {}
    for p in pairs:
        by_day.setdefault(p.day, []).append(p)
    for day, ps in by_day.items():
        ps = sorted(ps, key=lambda p: p.entry)
        open_until: list[dt.datetime] = []
        tickers: set[str] = set()
        n = 0
        for p in ps:
            s = p.session
            if not params[f"enter_{s}"] or p.ticker in tickers or n >= MAX_PER_DAY:
                continue
            if params["drop_one_off"] and p.persistence == "one_off":
                continue
            if p.score <= params["min_score"]:
                continue
            m = pre_move(p)
            if m is None or m > params["pre_move_cap"]:
                continue
            open_until = [t for t in open_until if t > p.entry]
            if len(open_until) >= MAX_OPEN:
                continue
            t = trade(p, params[f"hold_{s}"], params["stop"])
            if t is None:
                continue
            tickers.add(p.ticker)
            n += 1
            open_until.append(t[0])
            done.append({"day": day, "key": p.key, "session": s, "net": t[1], "excess": t[2]})
    return done


BASELINE = {
    "enter_pre_market": True,
    "enter_intraday": True,
    "enter_after_close": True,
    "hold_pre_market": CLOSE,
    "hold_intraday": CLOSE,
    "hold_after_close": CLOSE,
    "pre_move_cap": 0.02,
    "stop": 1.0,
    "drop_one_off": False,
    "min_score": 0.15,
}


def suggest(trial: optuna.Trial) -> dict:
    p = {}
    for s in SESSIONS:
        p[f"enter_{s}"] = trial.suggest_categorical(f"enter_{s}", [True, False])
        p[f"hold_{s}"] = trial.suggest_categorical(f"hold_{s}", [15, 30, 60, 120, 240, CLOSE])
    p["pre_move_cap"] = trial.suggest_float("pre_move_cap", 0.0, 0.10)
    p["stop"] = trial.suggest_categorical("stop", [0.01, 0.02, 0.03, 0.05, 1.0])
    p["drop_one_off"] = trial.suggest_categorical("drop_one_off", [True, False])
    p["min_score"] = trial.suggest_float("min_score", 0.15, 0.5)
    return p


def score(trades: list[dict]) -> float:
    return sum(max(-0.1, min(0.1, t["excess"])) for t in trades)


def fit(pairs: list[Pair], trials: int, seed: int) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(BASELINE)
    study.optimize(lambda tr: score(run_policy(pairs, suggest(tr))), n_trials=trials)
    return {**BASELINE, **study.best_params}


def summary(trades: list[dict]) -> str:
    if not trades:
        return "n=0"
    ex = [t["excess"] for t in trades]
    net = [t["net"] for t in trades]
    return (
        f"n={len(trades):3d} 초과합={sum(ex) * 100:+7.2f}%p 초과평균={statistics.fmean(ex) * 100:+.2f}%p "
        f"순수익평균={statistics.fmean(net) * 100:+.2f}% 적중={sum(x > 0 for x in net) / len(net):.0%}"
    )


def main() -> None:
    require_backtest_host()
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--delay", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pairs = load_pairs(a.delay)
    days = sorted({p.day for p in pairs})
    print(f"BUY 쌍 {len(pairs)}개, 거래일 {days}")
    oof_base, oof_opt = [], []
    for d in days:
        train = [p for p in pairs if p.day != d]
        test = [p for p in pairs if p.day == d]
        best = fit(train, a.trials, a.seed)
        b, o = run_policy(test, BASELINE), run_policy(test, best)
        oof_base += b
        oof_opt += o
        shown = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items()}
        print(f"\n[빠진 날 {d}] 학습 최적 {shown}")
        print(f"  기준(현재 운영) {summary(b)}")
        print(f"  최적 정책      {summary(o)}")
    print("\n=== 빠진 날들 합산 (out-of-fold) ===")
    print(f"기준(현재 운영) {summary(oof_base)}")
    print(f"최적 정책      {summary(oof_opt)}")
    full = fit(pairs, a.trials, a.seed)
    shown = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in full.items()}
    print(f"\n전체 6일로 맞춘 정책 (참고, 검증 아님): {shown}")
    print(f"  전체 기준 {summary(run_policy(pairs, BASELINE))}")
    print(f"  전체 최적 {summary(run_policy(pairs, full))}")


if __name__ == "__main__":
    main()
