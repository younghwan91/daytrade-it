"""Learn the two new live exit rules' thresholds: stall take-profit and break-even stop.

Run from a worktree checked out at main, so `stall_exit_hit` / `breakeven_stop_hit` /
`vwap_exit_hit` are imported from the code live will actually run rather than
reimplemented. The evaluation order mirrors `_early_exits` exactly:

    VWAP break  ->  stall take-profit  ->  break-even stop

and if none fires the position is flattened at the first print at or after
`flatten_at` (live retries every minute with a widening discount until it fills,
so a ticker with no 15:00 bar leaves on its next print).

Objective, per the user's asymmetric rule ("잃는 게 안 버느니만 못하다"): capital
preservation outranks expectancy. Results are reported under BOTH a
downside-weighted score (losses weighted by `lam`) and plain expected KRW, so the
trade-off is visible instead of buried in the objective.

Everything here is the sweep pool -- the uncapped one-share counterfactual. Per-trade
bp transfer; KRW totals do not. Anything that wins must still be validated on
dashboard days through the real engine before it is called adoptable, because three
risk-cutting changes have already reversed between the two pools.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import multiprocessing
from collections import defaultdict
from decimal import Decimal
from pathlib import Path as FsPath

import numpy as np

from daytrade_it.application.trading.auto_trader import (
    TradingLimits,
    breakeven_stop_hit,
    stall_exit_hit,
    vwap_exit_hit,
)
from daytrade_it.infrastructure.brokers.base import MinuteBar

SELL_COST = Decimal("0.00215")
BUY_COST = Decimal("0.00015")
FLATTEN_MINUTE = 15 * 60
OPEN_MINUTE = 9 * 60
NOTIONAL = 100_000.0

#: 0 is deliberately excluded: at min_gain = 0 a flat post-entry window satisfies
#: both the gain and the stall conditions, so the rule degenerates into "sell when
#: it stops moving, profit or not". Selecting it would be a signal the rule is not
#: earning its place rather than a learned value.
STALL_GAIN = (None, Decimal("0.005"), Decimal("0.010"), Decimal("0.020"))
STALL_MINUTES = (5, 10)
BREAKEVEN = (None, Decimal("0.010"), Decimal("0.020"), Decimal("0.030"))
BUFFER = (Decimal("0.003"),)


_ROWS: list[dict] = []
_BARS: list = []


def _score_config(lim: TradingLimits) -> np.ndarray:
    return np.array([simulate(r, b, lim)[0] for r, b in zip(_ROWS, _BARS, strict=True)])


def load(path: FsPath) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def bars_of(row: dict) -> list[MinuteBar]:
    day = dt.date.fromisoformat(row["day"])
    return [
        MinuteBar(
            ts=dt.datetime.combine(day, dt.time(m // 60, m % 60)),
            open=Decimal(str(o)),
            high=Decimal(str(h)),
            low=Decimal(str(lo)),
            close=Decimal(str(c)),
            volume=int(v),
        )
        for m, o, h, lo, c, v in row["bars"]
    ]


def simulate(row: dict, bars: list[MinuteBar], lim: TradingLimits) -> tuple[float, str, int]:
    """(reward, exit reason, exit minute) under `lim`, using the live predicates."""
    day = dt.date.fromisoformat(row["day"])
    buy_minute = int(row["buy_minute"])
    entered_at = dt.datetime.combine(day, dt.time(buy_minute // 60, buy_minute % 60))
    entry = Decimal(str(row["buy_price"]))
    by_min = {b.ts.hour * 60 + b.ts.minute: b for b in bars}
    after = sorted(m for m in by_min if m > buy_minute)
    if not after or entry <= 0:
        return 0.0, "no_bars", buy_minute
    at_or_after = [m for m in after if m >= FLATTEN_MINUTE]
    close_at = at_or_after[0] if at_or_after else after[-1]
    real_flatten = bool(at_or_after)

    for m in after:
        if m > close_at:
            break
        bar = by_min[m]
        if m == close_at:
            price = bar.open if real_flatten else bar.close
            return _net(entry, price), ("flatten" if real_flatten else "last_bar"), m
        now = dt.datetime.combine(day, dt.time(m // 60, m % 60))
        # _early_exits' order, exactly.
        if lim.vwap_exit_gap is not None:
            hit, _c, _v = vwap_exit_hit(bars, entry, now, lim.vwap_exit_gap, lim.vwap_exit_min_gain)
            if hit:
                return _net(entry, bar.open), "vwap_break", m
        if lim.stall_exit_min_gain is not None:
            hit, _g = stall_exit_hit(
                bars, entry, now, lim.stall_exit_min_gain, lim.stall_exit_minutes, entered_at
            )
            if hit:
                return _net(entry, bar.open), "stall", m
        if lim.breakeven_after_gain is not None:
            hit, _p, _c = breakeven_stop_hit(
                bars, entry, now, lim.breakeven_after_gain, lim.breakeven_buffer, entered_at
            )
            if hit:
                return _net(entry, bar.open), "breakeven", m
    return 0.0, "no_bars", buy_minute


def _net(entry: Decimal, proceeds: Decimal) -> float:
    return float((proceeds * (1 - SELL_COST) - entry * (1 + BUY_COST)) / entry)


def grid() -> list[TradingLimits]:
    out = []
    for sg, sm, be, buf in itertools.product(STALL_GAIN, STALL_MINUTES, BREAKEVEN, BUFFER):
        if sg is None and sm != STALL_MINUTES[0]:
            continue  # stall off: stall_minutes is inert, keep one copy
        if be is None and buf != BUFFER[0]:
            continue  # breakeven off: buffer is inert
        out.append(
            TradingLimits(
                stall_exit_min_gain=sg,
                stall_exit_minutes=sm,
                breakeven_after_gain=be,
                breakeven_buffer=buf,
            )
        )
    return out


def label(lim: TradingLimits) -> str:
    sg = "off" if lim.stall_exit_min_gain is None else f"{float(lim.stall_exit_min_gain):.3f}"
    be = "off" if lim.breakeven_after_gain is None else f"{float(lim.breakeven_after_gain):.3f}"
    return f"stall={sg}/{lim.stall_exit_minutes}m be={be}/{float(lim.breakeven_buffer):.3f}"


def day_score(days: np.ndarray, rewards: np.ndarray, lam: float) -> float:
    """Downside-weighted daily total: losing days count `lam` times.

    LATENT COMPOSITION -- read before routing a candidate selected by this into
    `bayes_judge`. This penalises bad DAYS, which is the same shape as the gate's
    `worst_day` criterion. Selecting on it and then gating on `worst_day` would be
    a select-on-X / gate-on-X composition: the gate cannot catch a distortion the
    selector introduced in the statistic they share. Today this feeds descriptive
    tables only, so the composition is latent rather than active. If you change
    that, weight the day term by the day's trade count, or judge on a criterion
    the selector does not touch (`loser_ratio` is trade-weighted on both sides and
    no selector uses it).
    """
    per = defaultdict(float)
    for d, r in zip(days.tolist(), rewards.tolist(), strict=True):
        per[d] += r
    return float(sum(v if v > 0 else lam * v for v in per.values()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=FsPath, required=True)
    ap.add_argument("--out", type=FsPath, required=True)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    from stall_repro import main as gate  # noqa: PLC0415

    print("regression gate: rules must not read pre-entry bars ...", end=" ")
    gate()
    print("passed\n")

    rows = load(args.paths)
    bars = [bars_of(r) for r in rows]
    days = np.array([r["day"] for r in rows])
    months = sorted({d[:7] for d in days.tolist()})
    configs = grid()
    print(
        f"entries={len(rows)} days={len(set(days.tolist()))} months={months} configs={len(configs)}"
    )

    global _ROWS, _BARS  # noqa: PLW0603
    _ROWS, _BARS = rows, bars
    with multiprocessing.Pool(args.jobs) as pool:
        scored = pool.map(_score_config, configs, chunksize=1)
    scores: dict[int, np.ndarray] = dict(enumerate(scored))
    base_idx = next(
        i
        for i, c in enumerate(configs)
        if c.stall_exit_min_gain is None and c.breakeven_after_gain is None
    )
    base = scores[base_idx]
    print(
        f"live default (both rules off): {base.mean() * 1e4:+.0f} bp,"
        f" win {float((base > 0).mean()):.3f}"
    )

    print(f"\n=== monthly time-ordered folds, chosen on downside-weighted score (lam={args.lam})")
    print(f"{'fold':9s} {'n':>5s} {'chosen':44s} {'live bp':>8s} {'chosen bp':>10s}")
    fold_rows = []
    for m in months[1:]:
        tr = np.array([d[:7] < m for d in days.tolist()])
        ev = np.array([d[:7] == m for d in days.tolist()])
        best = max(scores, key=lambda i: day_score(days[tr], scores[i][tr], args.lam))
        lim = configs[best]
        fold_rows.append(
            {
                "fold": m,
                "n": int(ev.sum()),
                "config": label(lim),
                "stall_exit_min_gain": None
                if lim.stall_exit_min_gain is None
                else float(lim.stall_exit_min_gain),
                "stall_exit_minutes": lim.stall_exit_minutes,
                "breakeven_after_gain": None
                if lim.breakeven_after_gain is None
                else float(lim.breakeven_after_gain),
                "breakeven_buffer": float(lim.breakeven_buffer),
                "live_bp": float(base[ev].mean() * 1e4),
                "chosen_bp": float(scores[best][ev].mean() * 1e4),
            }
        )
        print(
            f"{m:9s} {int(ev.sum()):>5d} {label(lim):44s}"
            f" {base[ev].mean() * 1e4:>+8.0f} {scores[best][ev].mean() * 1e4:>+10.0f}"
        )

    # Single-rule sweeps at full sample, for the shape of each knob.
    print("\n=== each rule alone, full sample (bp, and downside-weighted daily score)")
    print(f"{'config':44s} {'bp':>8s} {'win':>7s} {'worst day':>11s} {'lam-score':>11s}")
    singles = []
    for i, lim in enumerate(configs):
        if (lim.stall_exit_min_gain is None) == (lim.breakeven_after_gain is None):
            continue
        r = scores[i]
        per = defaultdict(float)
        for d, v in zip(days.tolist(), r.tolist(), strict=True):
            per[d] += v
        singles.append(
            (
                label(lim),
                r.mean() * 1e4,
                float((r > 0).mean()),
                min(per.values()) * NOTIONAL,
                day_score(days, r, args.lam) * NOTIONAL,
            )
        )
    per_base = defaultdict(float)
    for d, v in zip(days.tolist(), base.tolist(), strict=True):
        per_base[d] += v
    print(
        f"{'LIVE DEFAULT (both off)':44s} {base.mean() * 1e4:>+8.0f}"
        f" {float((base > 0).mean()):>7.3f} {min(per_base.values()) * NOTIONAL:>+11,.0f}"
        f" {day_score(days, base, args.lam) * NOTIONAL:>+11,.0f}"
    )
    for name, bp, win, worst, sc in sorted(singles, key=lambda x: -x[4])[:14]:
        print(f"{name:44s} {bp:>+8.0f} {win:>7.3f} {worst:>+11,.0f} {sc:>+11,.0f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "folds": fold_rows,
                "live_bp": float(base.mean() * 1e4),
                "singles": [
                    {"config": n, "bp": bp, "win": w, "worst_day": wd, "lam_score": s}
                    for n, bp, w, wd, s in sorted(singles, key=lambda x: -x[4])
                ],
            },
            indent=1,
        )
    )
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
