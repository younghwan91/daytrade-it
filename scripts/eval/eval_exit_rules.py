"""청산 규칙 1단계 평가 — simnode 전용.

    uv run python scripts/eval/eval_exit_rules.py --phase dev
    uv run python scripts/eval/eval_exit_rules.py --phase test --frozen "T0.03,SL0.03/TP-,H120,M0.03/30"

dev에서 규칙별 파라미터를 고르고, 그 이름을 문서에 적은 뒤 test를 한 번만 본다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path

import psycopg2

from daytrade_it.application.trading.exit_rules import (
    Bar,
    Baseline,
    MomentumFade,
    StopTake,
    TimeExit,
    Trailing,
    simulate_exit,
)
from daytrade_it.shared.runtime_host import require_backtest_host

SAMPLES = Path("data/eval/exit_samples.jsonl")
RESULTS = Path("data/eval/exit_results.jsonl")
MIN_BARS = 300
MAX_GAP_RATIO = 0.10

GRID = {
    "T": [Trailing(x) for x in (0.01, 0.02, 0.03, 0.05)],
    "SL/TP": [StopTake(y, z) for y in (0.02, 0.03, 0.05) for z in (0.03, 0.05, 0.08, None)],
    "H": [TimeExit(n) for n in (30, 60, 120, 240)],
    "M": [MomentumFade(a, k) for a in (0.01, 0.03) for k in (15, 30, 60)],
}


def load_bars(cur, code: str, day: str) -> list[Bar]:
    cur.execute(
        "select ts, open, high, low, close, volume from minute_bars "
        "where code = %s and ts >= %s::date + time '09:00' and ts < %s::date + time '15:31' order by ts",
        (code, day, day),
    )
    return [
        Bar(ts, float(o), float(h), float(lo), float(c), int(v))
        for ts, o, h, lo, c, v in cur.fetchall()
    ]


def entry_index(bars: list[Bar], entry_ts: str) -> int | None:
    t = dt.datetime.fromisoformat(entry_ts)
    for i, b in enumerate(bars):
        if b.ts.replace(tzinfo=None) >= t:
            return i
    return None


def usable(bars: list[Bar], i: int) -> bool:
    if len(bars) < MIN_BARS:
        return False
    start = bars[i].ts.replace(tzinfo=None)
    end = start.replace(hour=15, minute=0)
    expected = max(int((end - start).total_seconds() // 60), 1)
    have = sum(1 for b in bars[i:] if b.ts.replace(tzinfo=None) < end)
    return have >= expected * (1 - MAX_GAP_RATIO)


def boot_ci(vals: list[float], n: int = 2000) -> tuple[float, float]:
    rng = random.Random(0)
    meds = sorted(statistics.median(rng.choices(vals, k=len(vals))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n)]


def winsor_mean(vals: list[float], cap: float = 0.10) -> float:
    return statistics.fmean(max(-cap, min(cap, v)) for v in vals)


def run(phase: str, frozen: list[str] | None) -> None:
    require_backtest_host()
    samples = [json.loads(line) for line in SAMPLES.open()]
    samples = [s for s in samples if s["split"] == phase and s["wide"]]
    rules = [Baseline()] + [r for rs in GRID.values() for r in rs]
    if frozen is not None:
        rules = [Baseline()] + [r for r in rules if r.name in frozen]
    rows = []
    skipped = 0
    with psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5) as c, c.cursor() as cur:
        for s in samples:
            bars = load_bars(cur, s["ticker"], s["trade_day"])
            i = entry_index(bars, s["entry_ts"]) if bars else None
            if i is None or not usable(bars, i):
                skipped += 1
                continue
            base = simulate_exit(bars, i, bars[i].open, Baseline())
            for rule in rules:
                r = simulate_exit(bars, i, bars[i].open, rule)
                rows.append(
                    {
                        **s,
                        "rule": rule.name,
                        "net": r.net_return,
                        "base_net": base.net_return,
                        "reason": r.reason,
                    }
                )
    with RESULTS.open("a") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        k: r[k]
                        for k in (
                            "key",
                            "rule",
                            "split",
                            "ops",
                            "wide",
                            "net",
                            "base_net",
                            "reason",
                        )
                    }
                )
                + "\n"
            )
    print(f"phase={phase} samples={len(samples)} skipped(no/gappy bars)={skipped}")
    for subset in ("ops", "wide"):
        by_rule: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r[subset]:
                by_rule[r["rule"]].append(r)
        print(f"\n[{subset}]  rule | n | mean | median | hit | winsor | Δmedian vs B | Δ 95% CI")
        for name, rs in sorted(
            by_rule.items(),
            key=lambda kv: -statistics.median(x["net"] - x["base_net"] for x in kv[1]),
        ):
            nets = [x["net"] for x in rs]
            diffs = [x["net"] - x["base_net"] for x in rs]
            lo, hi = boot_ci(diffs)
            print(
                f"{name:>14} | {len(rs):3d} | {statistics.fmean(nets):+.2%} | {statistics.median(nets):+.2%} | "
                f"{sum(n > 0 for n in nets) / len(nets):.0%} | {winsor_mean(nets):+.2%} | "
                f"{statistics.median(diffs):+.2%} | [{lo:+.2%}, {hi:+.2%}]"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "test", "forward"], required=True)
    ap.add_argument("--frozen", default=None, help="test/forward: 쉼표 구분 규칙 이름")
    a = ap.parse_args()
    frozen = a.frozen.split(",") if a.frozen else None
    if a.phase != "dev" and frozen is None:
        ap.error("test/forward는 --frozen 필수 (dev에서 고른 규칙만 본다)")
    run(a.phase, frozen)


if __name__ == "__main__":
    main()
