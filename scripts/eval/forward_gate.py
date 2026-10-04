#!/usr/bin/env python
"""전진 승격 게이트를 장부에 돌린다 -- 매일 15:50 크론, 결과는 data/eval/forward_gate/.

    uv run python scripts/eval/forward_gate.py            # 실매매+종이 장부 전체
    uv run python scripts/eval/forward_gate.py --since 2026-09-21 --n-trials 200

입력: data/auto_trader*/YYYY-MM-DD.jsonl (거래), data/news_judgments*/ (무작위 대조용
매수 풀), KR_QUANT_DB 의 minute_bars (청산가 = 청산 제출 시각 봉 종가, 대조 = 판정 직후
봉 시가 진입·15:00 봉 종가 청산). 비용은 왕복 36.7bp 고정 차감.

판정은 숫자만 적는다 -- 7개 전부 통과(`promote: true`)가 아니면 실매매 재개 논의도 하지 않는다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import random
import statistics as st
import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from daytrade_it.application.research.forward_gate import (  # noqa: E402
    ROUND_TRIP_TOLL_BP,
    drop_days_without_market,
    gate_report,
    ledger_trades,
)
from daytrade_it.infrastructure.external.anthropic_client import SIGNAL_THRESHOLD  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "eval" / "forward_gate"


def _dsn() -> str:
    dsn = os.environ.get("KR_QUANT_DB")
    if not dsn:
        env = ROOT.parent / "quant-airflow" / ".env"
        for line in env.read_text().splitlines():
            if line.startswith("KR_QUANT_DB="):
                dsn = line.split("=", 1)[1].strip().strip("\"'")
    if not dsn:
        sys.exit("KR_QUANT_DB 가 없다")
    # 읽기만 하므로 trader 의 로컬 복제본을 우선 쓴다.
    return (
        dsn.replace("192.168.45.9:5433", "localhost:5432")
        if os.environ.get("KR_QUANT_REPLICA", "1") == "1"
        else dsn
    )


class Bars:
    def __init__(self, cur):
        self.cur = cur
        self.cache: dict = {}

    def day(self, code: str, day: dt.date):
        k = (code, day)
        if k not in self.cache:
            self.cur.execute(
                "select ts, open, close from minute_bars where code=%s and ts>=%s and ts<%s order by ts",
                (code, day, day + dt.timedelta(days=1)),
            )
            self.cache[k] = [(r[0], float(r[1]), float(r[2])) for r in self.cur.fetchall()]
        return self.cache[k]

    def close_at_or_after(self, code: str, ts: dt.datetime, fallback: float) -> float:
        for row in self.day(code, ts.date()):
            if row[0] >= ts:
                return row[2]
        return fallback

    def open_after(self, code: str, ts: dt.datetime):
        for row in self.day(code, ts.date()):
            if row[0] >= ts:
                return row[1]
        return None

    def close_by(self, code: str, day: dt.date, t=dt.time(15, 0)):
        tgt = dt.datetime.combine(day, t)
        last = None
        for row in self.day(code, day):
            if row[0] <= tgt:
                last = row[2]
        return last


def random_control(bars: Bars, days: set[dt.date], rng: random.Random, draws: int = 300) -> dict:
    """그날 매수 풀(score > 문턱, 14:30 이전 판정)에서 무작위 5종목, 판정 직후 봉 시가 → 15:00 종가."""
    pool: dict[dt.date, list[float]] = {}
    for f in sorted(glob.glob(str(ROOT / "data" / "news_judgments*" / "*.jsonl"))):
        day = dt.date.fromisoformat(Path(f).stem)
        if day not in days:
            continue
        with open(f) as fh:
            rows = [json.loads(line) for line in fh]
        for j in rows:
            v = j.get("v2") or {}
            if (v.get("score") or 0) <= SIGNAL_THRESHOLD:
                continue
            jt = dt.datetime.fromisoformat(j["judged_at"]).replace(tzinfo=None)
            if jt.time() > dt.time(14, 30):
                continue
            ep = bars.open_after(j["ticker"], jt)
            xp = bars.close_by(j["ticker"], day)
            if ep and xp:
                pool.setdefault(day, []).append((xp / ep - 1) * 1e4 - ROUND_TRIP_TOLL_BP)
    return {
        d: st.mean(st.mean(rng.sample(lst, min(5, len(lst)))) for _ in range(draws))
        for d, lst in pool.items()
        if lst
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=dt.date.fromisoformat, default=dt.date(2026, 9, 14))
    ap.add_argument(
        "--n-trials", type=int, default=200, help="이 정책까지 시도한 후보 수(Deflated Sharpe 용)"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    conn = psycopg2.connect(_dsn())
    bars = Bars(conn.cursor())
    trades = []
    # 종이 장부의 주문번호(P000001...)는 날마다 다시 시작하므로 장부 파일(=하루)별로 짝을 맞춘다.
    for f in sorted(glob.glob(str(ROOT / "data" / "auto_trader*" / "*.jsonl"))):
        if dt.date.fromisoformat(Path(f).stem) >= args.since:
            with open(f) as fh:
                events = [json.loads(line) for line in fh]
            trades += ledger_trades(
                events, exit_price=bars.close_at_or_after, toll_bp=ROUND_TRIP_TOLL_BP
            )
    trades, holidays = drop_days_without_market(trades, lambda code, day: bool(bars.day(code, day)))
    days = {t.day for t in trades}
    rng = random.Random(args.seed)
    control = random_control(bars, days, rng)
    rep = gate_report(trades, control, n_trials=args.n_trials, seed=args.seed)
    rep["control_by_day_bp"] = {d.isoformat(): round(v, 1) for d, v in sorted(control.items())}
    rep["generated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    rep["toll_bp"] = ROUND_TRIP_TOLL_BP
    rep["dropped_no_market_days"] = [d.isoformat() for d in holidays]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    lines = [
        f"# forward gate {rep['generated_at']} -- days {len(rep['days'])} trades {rep['n_trades']} "
        f"median {rep['median_bp']:.1f}bp  promote={rep['promote']}",
        "",
        "| gate | value | ok |",
        "|---|---|---|",
    ]
    for name, g in rep["gates"].items():
        val = g.get("value", g.get("trades"))
        lines.append(
            f"| {name} | {val if isinstance(val, int) else f'{val:.3f}'} | {'PASS' if g['ok'] else 'FAIL'} |"
        )
    lines += [
        "",
        "per-day sum bp: " + ", ".join(f"{d} {v:+.0f}" for d, v in rep["per_day_sum_bp"].items()),
    ]
    md = "\n".join(lines)
    (OUT / "latest.md").write_text(md + "\n")
    (OUT / f"{dt.date.today().isoformat()}.md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
