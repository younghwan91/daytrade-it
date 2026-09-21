"""Learn exit rules from recorded entry paths, on monthly time-ordered folds.

uv run python scripts/eval/rl_exits.py --paths data/eval/rl_minimal/paths.jsonl \
    --dataset <bandit dataset.jsonl> --out data/eval/rl_minimal/exits.json     # simnode

Stages, in order, each gating the next:

1. **Twin verification.** The search scores configurations with a vectorized twin
   of `auto_trader.exit_rule_hit`; this asserts the twin returns the same exit
   minute as the live function on every path for a sample of configurations. If
   it does not, nothing else runs.
2. **Engine parity.** Re-scores the live-default limits and compares per-entry
   rewards against what `bandit_dataset` recorded through the real replay engine.
3. **MFE/MAE.** How much unrealized gain each entry reaches and gives back --
   whether there is anything for a profit-taking rule to harvest.
4. **Global exit search**, monthly time-ordered folds: pick the configuration on
   the months strictly before the evaluation month, score it on that month.
5. **Context-dependent exit**: the same search per context bucket, so the exit
   is a function of the entry's situation rather than one constant for all
   entries. Judged against the global choice out-of-sample, because a
   context-dependent rule that does not beat one constant is just more
   parameters.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import random
from collections import defaultdict
from decimal import Decimal
from pathlib import Path as FsPath

import numpy as np
from exit_fast import OPEN_MINUTE, Reduced, barrier, first_exit, reduce_path
from exit_lab import FLATTEN, Path, mfe_mae, simulate
from replay_baseline import BUY_COST, SELL_COST

from daytrade_it.application.trading.auto_trader import TradingLimits

ORB_WIDTHS: tuple[int, ...] = (30,)
NOTIONAL = 100_000.0

STOP_LOSS = (None, Decimal("0.010"), Decimal("0.015"), Decimal("0.020"), Decimal("0.030"))
#: Extended to +8% and +10% on prompt-opt's finding that the travel-distance lift
#: concentrates at FAR targets, not near ones: at +1% it is 1.47x raw / 1.12x
#: sigma-normalised, at +10% it is 22.2x / 9.23x. A grid that stopped at +5% could
#: not have found a configuration that monetises that, so it was extended rather
#: than left to conclude "no target works".
TAKE_PROFIT = (
    None,
    Decimal("0.010"),
    Decimal("0.015"),
    Decimal("0.020"),
    Decimal("0.030"),
    Decimal("0.050"),
    Decimal("0.080"),
    Decimal("0.100"),
)
#: Wider arms for the same reason: a trailing stop that stays in for a large move
#: is the other shape that could monetise far travel.
TRAIL = (
    (None, Decimal("0.01")),
    (Decimal("0.01"), Decimal("0.005")),
    (Decimal("0.01"), Decimal("0.01")),
    (Decimal("0.02"), Decimal("0.01")),
    (Decimal("0.03"), Decimal("0.01")),
    (Decimal("0.02"), Decimal("0.02")),
    (Decimal("0.05"), Decimal("0.02")),
    (Decimal("0.05"), Decimal("0.03")),
    (Decimal("0.08"), Decimal("0.03")),
)
STOP_TIME = (None, dt.timedelta(minutes=30), dt.timedelta(minutes=60))
STOP_ORB = (None, dt.timedelta(minutes=30))
#: The live default runs an early VWAP-break exit (gap 0.01, min gain 0.02). Leaving
#: it pinned would have kept "hold to close" -- switching every early exit off and
#: simply flattening at 15:00 -- outside the search space entirely. prompt-opt
#: measured that baseline at -34 bp against the live rules' -94 bp on the 4,496 rows
#: we share, so it is not a straw man: a configuration that cannot beat doing
#: nothing is worth knowing about.
VWAP_EXIT = (Decimal("0.01"), None)


def grid() -> list[TradingLimits]:
    """Every configuration searched. `vwap_exit_gap` stays at its adopted live value."""
    base = TradingLimits()
    out = []
    for sl in STOP_LOSS:
        for tp in TAKE_PROFIT:
            for after, gap in TRAIL:
                for st in STOP_TIME:
                    for orb in STOP_ORB:
                        for vw in VWAP_EXIT:
                            out.append(
                                dataclasses.replace(
                                    base,
                                    stop_loss_pct=sl,
                                    take_profit_pct=tp,
                                    trail_after_gain=after,
                                    trail_gap=gap,
                                    stop_time=st,
                                    stop_orb=orb,
                                    vwap_exit_gap=vw,
                                )
                            )
    return out


# ------------------------------------------------------------------ stacked scorer


class Stack:
    """Every reduced path concatenated, so one configuration scores in a few array ops."""

    def __init__(self, reduced: list[Reduced]) -> None:
        self.n = len(reduced)
        lens = [r.minute.size for r in reduced]
        self.starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(int)
        cat = np.concatenate
        self.minute = cat([r.minute for r in reduced])
        self.fill = cat([r.fill for r in reduced])
        self.close = cat([r.close for r in reduced])
        self.vwap = cat([r.vwap for r in reduced])
        self.peak = cat([r.peak for r in reduced])
        self.has_peak = cat([r.has_peak for r in reduced])
        self.last_minute = cat([r.last_minute for r in reduced])
        self.elapsed = cat([r.elapsed for r in reduced])
        self.entry = cat([np.full(n, r.entry) for n, r in zip(lens, reduced, strict=True)])
        self.flat = cat([np.full(n, r.flatten_minute) for n, r in zip(lens, reduced, strict=True)])
        self.orb = np.stack(
            [
                cat([np.full(n, r.orb_low[i]) for n, r in zip(lens, reduced, strict=True)])
                for i in range(len(ORB_WIDTHS))
            ]
        )
        self.entry_per_path = np.array([r.entry for r in reduced])
        self.valid = np.isfinite(self.vwap)
        self.idx = np.arange(self.minute.size)
        self._barriers: dict[tuple[str, str, int], np.ndarray] = {}

    def _per_element(self, per_path: np.ndarray) -> np.ndarray:
        """Broadcast one value per path across that path's decision minutes."""
        lens = np.diff(np.append(self.starts, self.minute.size))
        return np.repeat(per_path, lens)

    def _barrier(self, which: str, pct, sign: int) -> np.ndarray:
        """`exit_fast.barrier` over `self.entry` or `self.peak`, memoised.

        The live predicate computes every `x * (1 +/- pct)` threshold in Decimal,
        so an exactly-hit target is a HIT; float64 misses it and exits a minute
        late (see `exit_fast.barrier`). The search scores ~4,300 configurations
        over the same arrays, but only ~20 distinct `(array, pct, sign)` triples
        exist, so the Decimal work is done once per triple and never inside the
        configuration loop.
        """
        key = (which, str(pct), sign)
        hit = self._barriers.get(key)
        if hit is None:
            hit = barrier(self.entry if which == "entry" else self.peak, pct, sign)
            self._barriers[key] = hit
        return hit

    def rewards(
        self,
        lim: TradingLimits,
        stop_pct: np.ndarray | None = None,
        tp_pct: np.ndarray | None = None,
        trail_after: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reward per path. The three overrides take ONE VALUE PER PATH, which is how
        a context-dependent threshold enters: the same rule, a different level for
        each entry (e.g. a multiple of that stock's own volatility) instead of one
        constant for all of them. `None` keeps the scalar on `lim`.

        The scalar thresholds go through `_barrier`, which reproduces the live
        rule's Decimal arithmetic exactly. The per-path overrides do NOT and stay
        in float: no live rule computes `a x sigma`, so there is nothing to be a
        twin of, and `a x sigma` is a full-mantissa float that cannot land on a
        tick-exact price the way a round percentage of a round entry does.
        """
        e, v = self.entry, self.valid
        fire = self.minute == self.flat  # the flatten minute always exits
        if stop_pct is not None:
            fire |= (self.close <= e * (1 - self._per_element(stop_pct))) & v
        elif lim.stop_loss_pct is not None:
            fire |= (self.close <= self._barrier("entry", lim.stop_loss_pct, -1)) & v
        if lim.stop_vwap_gap is not None:
            fire |= (self.close < e) & (self.close < self.vwap * (1 - float(lim.stop_vwap_gap))) & v
        if lim.stop_time is not None:
            fire |= (
                (self.elapsed >= lim.stop_time.total_seconds() / 60)
                & (self.close < self._barrier("entry", lim.stop_time_min_gain, +1))
                & v
            )
        if lim.stop_orb is not None:
            w = int(lim.stop_orb.total_seconds() // 60)
            low = self.orb[ORB_WIDTHS.index(w)]
            fire |= (
                (self.last_minute >= OPEN_MINUTE + w) & (self.close < low) & np.isfinite(low) & v
            )
        if tp_pct is not None:
            fire |= (self.close >= e * (1 + self._per_element(tp_pct))) & v
        elif lim.take_profit_pct is not None:
            fire |= (self.close >= self._barrier("entry", lim.take_profit_pct, +1)) & v
        if trail_after is not None:
            fire |= (
                self.has_peak
                & (self.peak >= e * (1 + self._per_element(trail_after)))
                & (self.close <= self._barrier("peak", lim.trail_gap, -1))
                & v
            )
        elif lim.trail_after_gain is not None:
            fire |= (
                self.has_peak
                & (self.peak >= self._barrier("entry", lim.trail_after_gain, +1))
                & (self.close <= self._barrier("peak", lim.trail_gap, -1))
                & v
            )
        if lim.vwap_exit_gap is not None:
            fire |= (
                (self.close >= self._barrier("entry", lim.vwap_exit_min_gain, +1))
                & (self.close / self.vwap - 1 < -float(lim.vwap_exit_gap))
                & v
            )
        big = self.minute.size + 1
        first = np.minimum.reduceat(np.where(fire, self.idx, big), self.starts)
        first = np.minimum(first, self.minute.size - 1)
        proceeds = self.fill[first]
        buy = self.entry_per_path
        return (proceeds * (1 - float(SELL_COST)) - buy * (1 + float(BUY_COST))) / buy


# ------------------------------------------------------------------ helpers


SIGMA_BARS = 30
SIGMA_FLOOR = 0.0015
#: Volatility multiples searched for the context-dependent exit. The stop and the
#: target stop being fixed percentages and become multiples of the stock's own
#: pre-entry minute volatility, so "how far is too far" is set by the ticker.
SIGMA_STOP = (None, 1.0, 1.5, 2.0, 3.0, 4.0)
SIGMA_TP = (None, 1.5, 2.0, 3.0, 4.0, 6.0)


def entry_sigma(path: Path) -> float:
    """Stdev of 1-minute close-to-close returns over the bars completed BEFORE the fill.

    Decision-time only: every bar used ended before the entry minute, so the live
    daemon has exactly this (it already pulls the ticker's minute bars for
    `ret_5m`). Floored so a barely-traded opening minute cannot produce a
    zero-width stop.
    """
    closes = [
        float(b.close)
        for b in path.bars
        if b.ts.hour * 60 + b.ts.minute < path.buy_minute and b.close > 0
    ][-(SIGMA_BARS + 1) :]
    if len(closes) < 3:
        return float("nan")
    arr = np.array(closes)
    rets = arr[1:] / arr[:-1] - 1
    return max(float(rets.std()), SIGMA_FLOOR)


def describe(lim: TradingLimits) -> dict:
    def pct(v: Decimal | None) -> float | None:
        return None if v is None else float(v)

    def mins(v: dt.timedelta | None) -> int | None:
        return None if v is None else int(v.total_seconds() // 60)

    return {
        "stop_loss_pct": pct(lim.stop_loss_pct),
        "take_profit_pct": pct(lim.take_profit_pct),
        "trail_after_gain": pct(lim.trail_after_gain),
        "trail_gap": pct(lim.trail_gap) if lim.trail_after_gain is not None else None,
        "stop_time_min": mins(lim.stop_time),
        "stop_orb_min": mins(lim.stop_orb),
        "vwap_exit_gap": pct(lim.vwap_exit_gap),
    }


def short(lim: TradingLimits) -> str:
    d = describe(lim)
    bits = [
        f"vw={d['vwap_exit_gap']}",
        f"sl={d['stop_loss_pct']}",
        f"tp={d['take_profit_pct']}",
        f"tr={d['trail_after_gain']}/{d['trail_gap']}",
        f"st={d['stop_time_min']}",
        f"orb={d['stop_orb_min']}",
    ]
    return " ".join(bits)


def day_records(
    days: np.ndarray, rewards: np.ndarray, keep: np.ndarray | None = None
) -> list[dict]:
    """`bayes_judge`-shaped day records at a flat notional, over the kept entries."""
    if keep is None:
        keep = np.ones_like(rewards, dtype=bool)
    out = []
    for day in sorted(set(days.tolist())):
        sel = (days == day) & keep
        r = rewards[sel]
        out.append(
            {
                "day": day,
                "net": NOTIONAL * float(r.sum()),
                "trades": int(sel.sum()),
                "losers": int((r < 0).sum()),
                "worst_trade": NOTIONAL * float(r.min()) if r.size else 0.0,
                "errors": 0,
                "leftovers": 0,
            }
        )
    return out


def objective(days: np.ndarray, rewards: np.ndarray) -> float:
    """Total reward minus half the worst day -- `bandit_lodo.objective`'s shape."""
    per_day = defaultdict(float)
    for d, r in zip(days.tolist(), rewards.tolist(), strict=True):
        per_day[d] += r
    vals = list(per_day.values())
    return float(sum(vals) - 0.5 * max(0.0, -min(vals)))


# ------------------------------------------------------------------ main


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=FsPath, required=True)
    ap.add_argument(
        "--dataset", type=FsPath, required=True, help="bandit_dataset rewards, for parity"
    )
    ap.add_argument("--out", type=FsPath, required=True)
    ap.add_argument("--verify-configs", type=int, default=12)
    ap.add_argument(
        "--verify-paths",
        type=int,
        default=400,
        help="paths sampled for the exactness test (the live predicate is O(bars^2) per path)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from bayes_judge import judge

    rows = [json.loads(line) for line in args.paths.read_text().splitlines() if line]
    paths = [Path.from_row(r) for r in rows]
    flat_minute = FLATTEN.hour * 60 + FLATTEN.minute
    reduced, kept = [], []
    for p, row in zip(paths, rows, strict=True):
        r = reduce_path(p.bars, p.buy_minute, p.buy_price, flat_minute, ORB_WIDTHS)
        if r is not None:
            reduced.append(r)
            kept.append((p, row))
    print(f"paths={len(rows)}  usable={len(reduced)}  dropped={len(rows) - len(reduced)}")
    paths = [p for p, _ in kept]
    rows = [r for _, r in kept]
    days = np.array([r["day"] for r in rows])
    months = sorted({d[:7] for d in days.tolist()})
    stack = Stack(reduced)
    configs = grid()
    print(f"days={len(set(days.tolist()))} months={months} grid={len(configs)} configs")

    # -- 1. twin verification ------------------------------------------------
    rng = random.Random(args.seed)
    sample = [TradingLimits()] + rng.sample(configs, min(args.verify_configs, len(configs)))
    idx = rng.sample(range(len(paths)), min(args.verify_paths, len(paths)))
    vpaths = [(paths[i], reduced[i]) for i in idx]
    bad = 0
    for lim in sample:
        for p, r in vpaths:
            _, _, ref_minute = simulate(p, lim)
            _, _, fast_minute = first_exit(r, lim, ORB_WIDTHS)
            if ref_minute != fast_minute:
                bad += 1
                if bad <= 5:
                    print(
                        f"  MISMATCH {p.day} {p.ticker} {short(lim)}: {ref_minute} vs {fast_minute}"
                    )
    print(f"twin verification: {len(sample)} configs x {len(vpaths)} paths -> {bad} mismatches")
    if bad:
        raise SystemExit("twin does not match the live predicate; refusing to search")

    # -- 2. engine parity ----------------------------------------------------
    live = TradingLimits()
    lab = stack.rewards(live)
    hold = stack.rewards(dataclasses.replace(live, vwap_exit_gap=None))
    print(
        f"\nbaselines: live exits {lab.mean() * 1e4:+.0f} bp"
        f" (win {float((lab > 0).mean()):.3f})"
        f"   hold-to-close {hold.mean() * 1e4:+.0f} bp"
        f" (win {float((hold > 0).mean()):.3f})"
        f"   difference {(hold.mean() - lab.mean()) * 1e4:+.0f} bp"
    )

    def compare(label: str, ref: dict[tuple[str, str], float]) -> None:
        pairs = [
            (lab[i], ref[(r["day"], r["ticker"])])
            for i, r in enumerate(rows)
            if (r["day"], r["ticker"]) in ref
        ]
        if not pairs:
            print(f"{label}: no overlapping entries")
            return
        a = np.array([x for x, _ in pairs])
        b = np.array([y for _, y in pairs])
        print(
            f"{label} on {len(pairs)} entries: lab {a.mean() * 1e4:+.0f} bp vs"
            f" engine {b.mean() * 1e4:+.0f} bp, corr {np.corrcoef(a, b)[0, 1]:.4f},"
            f" median |diff| {np.median(np.abs(a - b)) * 1e4:.1f} bp,"
            f" |diff|>50bp on {(np.abs(a - b) > 0.005).mean():.3f}"
        )

    # Same policy the paths were recorded under (add-ons off): this is the parity gate.
    same = {
        (r["day"], r["ticker"]): r["engine_reward"]
        for r in rows
        if r.get("engine_reward") is not None
    }
    compare("engine parity (same policy, add-ons off)", same)
    # The live default additionally pyramids; the gap between the two is what
    # add-ons cost, not a lab error.
    with_addons = {}
    for line in args.dataset.read_text().splitlines():
        if not line:
            continue
        d = json.loads(line)
        if d.get("reward") is not None:
            with_addons[(d["day"], d["ticker"])] = d["reward"]
    compare("vs live default (add-ons ON)", with_addons)

    # -- 3. MFE / MAE --------------------------------------------------------
    exc = np.array([mfe_mae(p) for p in paths])
    mfe, mae = exc[:, 0], exc[:, 1]
    print("\nunrealized excursion between fill and 15:00 (bp):")
    print(f"{'quantile':>10s} {'MFE':>8s} {'MAE':>8s}")
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        print(f"{q:>10.2f} {np.quantile(mfe, q) * 1e4:>+8.0f} {np.quantile(mae, q) * 1e4:>+8.0f}")
    print(f"{'mean':>10s} {mfe.mean() * 1e4:>+8.0f} {mae.mean() * 1e4:>+8.0f}")
    for thr in (0.01, 0.02, 0.03, 0.05):
        reach = float((mfe >= thr).mean())
        realized = lab[mfe >= thr].mean() * 1e4 if reach else float("nan")
        print(
            f"  reached +{thr:.0%} unrealized: {reach:.3f} of entries;"
            f" live exits realized {realized:+.0f} bp on them"
        )

    # -- 4. global exit search on monthly time-ordered folds -----------------
    scores = {i: stack.rewards(c) for i, c in enumerate(configs)}
    print("\n=== global exit, monthly time-ordered folds")
    print(
        f"{'fold':9s} {'n':>5s} {'chosen config':52s}"
        f" {'live bp':>8s} {'hold bp':>8s} {'chosen bp':>10s}"
    )
    fold_rows = []
    cand_days, base_days = [], []
    for m in months[1:]:
        tr = np.array([d[:7] < m for d in days.tolist()])
        ev = np.array([d[:7] == m for d in days.tolist()])
        best = max(scores, key=lambda i: objective(days[tr], scores[i][tr]))
        lim = configs[best]
        ev_r, ev_live, ev_hold = scores[best][ev], lab[ev], hold[ev]
        fold_rows.append(
            {
                "fold": m,
                "n": int(ev.sum()),
                "config": describe(lim),
                "live_bp": float(ev_live.mean() * 1e4),
                "chosen_bp": float(ev_r.mean() * 1e4),
            }
        )
        cand_days += day_records(days[ev], ev_r)
        base_days += day_records(days[ev], ev_live)
        print(
            f"{m:9s} {int(ev.sum()):>5d} {short(lim):52s}"
            f" {ev_live.mean() * 1e4:>+8.0f} {ev_hold.mean() * 1e4:>+8.0f}"
            f" {ev_r.mean() * 1e4:>+10.0f}"
        )
    if not cand_days:
        print("fewer than two months: no time-ordered fold is possible")
        return
    oos_cand = {d["day"]: d for d in cand_days}
    oos_base = {d["day"]: d for d in base_days}
    hold_days = day_records(
        days[np.isin(days, [d["day"] for d in base_days])],
        hold[np.isin(days, [d["day"] for d in base_days])],
    )
    verdict_global = judge(oos_cand, oos_base)
    verdict_vs_hold = judge(oos_cand, {d["day"]: d for d in hold_days})
    print(
        f"pooled: live {sum(d['net'] for d in base_days):+,.0f} KRW,"
        f" learned {sum(d['net'] for d in cand_days):+,.0f} KRW"
    )
    print(
        f"bayes_judge vs live exits: p(mu>0)={verdict_global['p_mu_gt_0']:.3f}"
        f" mu={verdict_global['mu_mean']:+,.0f} days>=0={verdict_global['days_ge_0']}"
        f"/{verdict_global['n_days']}"
        f" worst {verdict_global['worst_day_candidate']:+,.0f} vs"
        f" {verdict_global['worst_day_baseline']:+,.0f}"
        f" losers {verdict_global['loser_ratio_candidate']:.3f} vs"
        f" {verdict_global['loser_ratio_baseline']:.3f} adopt={verdict_global['adopt']}"
    )
    for r in verdict_global["reasons"]:
        print(f"    - {r}")
    v = verdict_vs_hold
    print(
        f"bayes_judge vs HOLD-TO-CLOSE: p(mu>0)={v['p_mu_gt_0']:.3f}"
        f" mu={v['mu_mean']:+,.0f} days>=0={v['days_ge_0']}/{v['n_days']}"
        f" worst {v['worst_day_candidate']:+,.0f} vs {v['worst_day_baseline']:+,.0f}"
        f" losers {v['loser_ratio_candidate']:.3f} vs {v['loser_ratio_baseline']:.3f}"
        f" adopt={v['adopt']}"
    )
    for r in v["reasons"]:
        print(f"    - {r}")

    # -- 5. context-dependent exit: thresholds as multiples of entry volatility --
    sigma = np.array([entry_sigma(p) for p in paths])
    median_sigma = float(np.nanmedian(sigma))
    sigma = np.where(np.isnan(sigma), median_sigma, sigma)
    print(
        f"\nentry volatility (1-min sd over the {SIGMA_BARS} bars before the fill):"
        f" median {median_sigma * 1e4:.0f} bp,"
        f" p10 {np.quantile(sigma, 0.1) * 1e4:.0f}, p90 {np.quantile(sigma, 0.9) * 1e4:.0f} bp"
    )
    base = TradingLimits()
    ctx_grid = [(a, b) for a in SIGMA_STOP for b in SIGMA_TP]
    ctx_scores = {}
    for a, b in ctx_grid:
        ctx_scores[(a, b)] = stack.rewards(
            base,
            stop_pct=None if a is None else a * sigma,
            tp_pct=None if b is None else b * sigma,
        )
    print("\n=== context-dependent exit (stop = a x sigma, target = b x sigma)")
    print(
        f"{'fold':9s} {'n':>5s} {'a':>5s} {'b':>5s} {'live bp':>8s} {'fixed bp':>9s} {'ctx bp':>8s}"
    )
    ctx_days, fixed_days2 = [], []
    ctx_rows = []
    for row, m in zip(fold_rows, months[1:], strict=True):
        tr = np.array([d[:7] < m for d in days.tolist()])
        ev = np.array([d[:7] == m for d in days.tolist()])
        best = max(ctx_scores, key=lambda k: objective(days[tr], ctx_scores[k][tr]))
        ev_ctx = ctx_scores[best][ev]
        fixed_best = max(scores, key=lambda i: objective(days[tr], scores[i][tr]))
        ev_fixed = scores[fixed_best][ev]
        ctx_days += day_records(days[ev], ev_ctx)
        fixed_days2 += day_records(days[ev], ev_fixed)
        ctx_rows.append(
            {"fold": m, "a": best[0], "b": best[1], "ctx_bp": float(ev_ctx.mean() * 1e4)}
        )
        print(
            f"{m:9s} {int(ev.sum()):>5d} {str(best[0]):>5s} {str(best[1]):>5s}"
            f" {row['live_bp']:>+8.0f} {ev_fixed.mean() * 1e4:>+9.0f} {ev_ctx.mean() * 1e4:>+8.0f}"
        )
    v_ctx_live = judge({d["day"]: d for d in ctx_days}, {d["day"]: d for d in base_days})
    v_ctx_fixed = judge({d["day"]: d for d in ctx_days}, {d["day"]: d for d in fixed_days2})
    for label, v in (("ctx vs live exits", v_ctx_live), ("ctx vs fixed learned", v_ctx_fixed)):
        print(
            f"{label:22s} p(mu>0)={v['p_mu_gt_0']:.3f} mu={v['mu_mean']:+,.0f}"
            f" days>=0={v['days_ge_0']}/{v['n_days']}"
            f" worst {v['worst_day_candidate']:+,.0f} vs {v['worst_day_baseline']:+,.0f}"
            f" losers {v['loser_ratio_candidate']:.3f} vs {v['loser_ratio_baseline']:.3f}"
            f" adopt={v['adopt']}"
        )
        for r in v["reasons"]:
            print(f"    - {r}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "folds": fold_rows,
                "context_folds": ctx_rows,
                "verdict_global": verdict_global,
                "verdict_vs_hold": verdict_vs_hold,
                "verdict_ctx_vs_live": v_ctx_live,
                "verdict_ctx_vs_fixed": v_ctx_fixed,
                "mfe_mean_bp": float(mfe.mean() * 1e4),
                "mae_mean_bp": float(mae.mean() * 1e4),
            },
            indent=1,
        )
    )
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
