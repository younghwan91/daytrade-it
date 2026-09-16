"""Learn the stop decision: cut-or-hold each minute, tabular, no hand-written rule.

The user's instruction is directional and corrects everything tested before it:
cut what is FALLING, early and decisively -- not what is rising. Live has no
stop-loss at all; the 15:00 flatten is the only floor.

**Why this can transfer where the earlier rules did not.** Trailing, the stall
exit and the break-even stop all cut positions that were UP, so on a pool with
positive drift they converted return into win rate and lost. A stop acts only on
positions that are DOWN, so that mechanism does not apply. This is the first exit
family whose failure mode is not structurally guaranteed on a profitable pool,
which is why it is worth learning rather than assuming.

**Method: tabular optimal stopping by policy iteration on discretised state.**
No stop percentage, gap or threshold is written down anywhere. The policy decides
cut-or-hold from the state bin, and the bin boundaries are the only hand-set
quantity -- deliberately coarse, because with ~4,400 episodes the risk is
variance. A bin whose visit count is below `--min-visits` defaults to hold, so a
decision is never made off a handful of samples; visit counts are reported.

**One structural constraint, from the instruction rather than from the data**:
`cut` is only available while the position is below entry. That encodes "cut the
falling, not the rising" into the policy space instead of hoping the data
rediscovers it, and it makes it impossible for this policy to become another
winner-capping rule.

Live's VWAP-break exit stays ON throughout (it is the live default and it fires
on winners); when it fires the episode terminates regardless of the policy, using
main's own `vwap_exit_hit`.

State, all reconstructible from 1-minute bars at decision time:
  pnl        close vs entry
  drawdown   close vs the position's own post-entry peak
  slope      3-minute return -- the speed of the decline
  held       minutes since entry

NOT observable and therefore NOT in the state: whether fills keep coming, book
pressure, order-flow confirmation. We have 1-minute OHLCV and 1-share orders, so
there are no partial fills to read. Minute volume was tested as a proxy and is
reported separately; it is not order flow and this policy should not be described
as conditioning on it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path as FsPath

import numpy as np

from daytrade_it.application.trading.auto_trader import TradingLimits, vwap_exit_hit
from daytrade_it.infrastructure.brokers.base import MinuteBar

SELL_COST = 0.00215
BUY_COST = 0.00015
FLATTEN_MINUTE = 15 * 60
NOTIONAL = 100_000.0

PNL_EDGES = (-0.03, -0.015, 0.0, 0.015)
DD_EDGES = (-0.04, -0.02, -0.01)
SLOPE_EDGES = (-0.005, 0.0, 0.005)
HELD_EDGES = (30,)
#: Turnover this minute against the stock's own recent baseline: is participation
#: drying up as the price falls, or is this heavy selling? Three bins only --
#: adding a dimension multiplies the state space, and with ~660k decision points
#: the constraint is visits per bin, not expressiveness.
#: NOTE `minute_bars.trade_value` is exactly close x volume (verified: relative
#: error <= 1e-5, fully populated), i.e. a derived column rather than
#: exchange-reported turnover, so it is computed here from the bars directly.
TV_EDGES = (0.7, 1.5)


def _bin(value: float, edges: tuple[float, ...]) -> int:
    return int(np.searchsorted(np.asarray(edges), value, side="right"))


def state_of(pnl: float, dd: float, slope: float, held: float, tv_rel: float | None) -> tuple:
    base = (
        _bin(pnl, PNL_EDGES),
        _bin(dd, DD_EDGES),
        _bin(slope, SLOPE_EDGES),
        _bin(held, HELD_EDGES),
    )
    return base if tv_rel is None else (*base, _bin(tv_rel, TV_EDGES))


def _net(entry: float, proceeds: float) -> float:
    return (proceeds * (1 - SELL_COST) - entry * (1 + BUY_COST)) / entry


class Episode:
    """One entry reduced to a per-minute decision sequence."""

    __slots__ = ("day", "minutes", "states", "cut_vals", "forced", "terminal", "can_cut")

    def __init__(self, row: dict, upside_vwap_exit: bool = True, use_volume: bool = True) -> None:
        day = dt.date.fromisoformat(row["day"])
        buy = int(row["buy_minute"])
        entry = float(row["buy_price"])
        bars = [
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
        by_min = {b.ts.hour * 60 + b.ts.minute: b for b in bars}
        after = sorted(m for m in by_min if m > buy)
        at_or_after = [m for m in after if m >= FLATTEN_MINUTE]
        close_at = at_or_after[0] if at_or_after else (after[-1] if after else buy)
        real_flat = bool(at_or_after)

        lim = TradingLimits()
        self.day = row["day"]
        self.minutes: list[int] = []
        self.states: list[tuple] = []
        self.cut_vals: list[float] = []
        self.forced: list[bool] = []
        self.can_cut: list[bool] = []
        peak = float("-inf")
        for m in after:
            if m > close_at:
                break
            bar = by_min[m]
            prev = [by_min[x] for x in after if x < m] or [by_min[after[0]]]
            peak = max(peak, max(float(b.high) for b in prev))
            close = float(prev[-1].close)
            pnl = close / entry - 1
            dd = close / peak - 1 if peak > 0 else 0.0
            three = [x for x in after if x <= m - 3]
            base = float(by_min[three[-1]].close) if three else close
            slope = close / base - 1 if base > 0 else 0.0
            tv_rel = None
            if use_volume:
                # Turnover of the last completed minute against the median of the
                # 20 before it. Minute turnover -- NOT aggressor side, fill
                # strength or book pressure, none of which exist in the pool-wide
                # history.
                recent = prev[-1]
                tv_now = float(recent.close) * recent.volume
                hist = [float(b.close) * b.volume for b in prev[-21:-1]]
                med = float(np.median(hist)) if hist else 0.0
                tv_rel = (tv_now / med) if med > 0 else 1.0
            now = dt.datetime.combine(day, dt.time(m // 60, m % 60))
            # "수익은 길게": with `upside_vwap_exit` off the only upside exit is the
            # 15:00 flatten, so nothing caps a winner and the stop cannot be
            # flattered by an upside that was truncated for it.
            fired = False
            if upside_vwap_exit:
                fired, _c, _v = vwap_exit_hit(
                    bars, Decimal(str(entry)), now, lim.vwap_exit_gap, lim.vwap_exit_min_gain
                )
            self.minutes.append(m)
            self.states.append(state_of(pnl, dd, slope, m - buy, tv_rel))
            self.cut_vals.append(_net(entry, float(bar.open)))
            self.forced.append(bool(fired) or m == close_at)
            # The instruction's asymmetry, encoded structurally: only cut a loser.
            self.can_cut.append(pnl < 0)
        if not real_flat and self.cut_vals:
            self.cut_vals[-1] = _net(entry, float(by_min[close_at].close))
        self.terminal = self.cut_vals[-1] if self.cut_vals else 0.0


def evaluate(ep: Episode, policy: dict | None) -> float:
    """Realized reward of running `policy` on `ep` (None = live as-is: no stop)."""
    for i, s in enumerate(ep.states):
        if ep.forced[i]:
            return ep.cut_vals[i]
        if policy is not None and ep.can_cut[i] and policy.get(s, False):
            return ep.cut_vals[i]
    return ep.terminal


def learn(eps: list[Episode], lam: float, min_visits: int, sweeps: int = 4) -> tuple[dict, dict]:
    """Tabular policy iteration. Returns (policy, per-state visit counts)."""

    def weighted(r: float) -> float:
        return r if r > 0 else lam * r

    policy: dict = {}
    visits: dict = defaultdict(int)
    for _ in range(sweeps):
        q_cut: dict = defaultdict(float)
        q_hold: dict = defaultdict(float)
        n: dict = defaultdict(int)
        for ep in eps:
            k = len(ep.states)
            v_next = [0.0] * (k + 1)
            v_next[k] = weighted(ep.terminal)
            for i in range(k - 1, -1, -1):
                if ep.forced[i] or ep.can_cut[i] and policy.get(ep.states[i], False):
                    v_next[i] = weighted(ep.cut_vals[i])
                else:
                    v_next[i] = v_next[i + 1]
            for i, s in enumerate(ep.states):
                if ep.forced[i] or not ep.can_cut[i]:
                    continue
                q_cut[s] += weighted(ep.cut_vals[i])
                q_hold[s] += v_next[i + 1]
                n[s] += 1
        policy = {s: (q_cut[s] / n[s]) > (q_hold[s] / n[s]) for s in n if n[s] >= min_visits}
        visits = dict(n)
    return policy, visits


def day_records(eps: list[Episode], rewards: list[float]) -> list[dict]:
    by_day: dict[str, list[float]] = defaultdict(list)
    for ep, r in zip(eps, rewards, strict=True):
        by_day[ep.day].append(r)
    return [
        {
            "day": d,
            "net": NOTIONAL * float(sum(v)),
            "trades": len(v),
            "losers": int(sum(1 for x in v if x < 0)),
            "worst_trade": NOTIONAL * min(v),
            "errors": 0,
            "leftovers": 0,
        }
        for d, v in sorted(by_day.items())
    ]


class FixedStop(dict):
    """The best hand-set stop, in the same tabular form so the comparison is fair.

    A dict subclass rather than an enumerated table so it works whatever the state
    tuple's arity is (the volume ablation changes it).
    """

    def __init__(self, pct: float) -> None:
        super().__init__()
        self.pct = pct

    def get(self, state: tuple, default: bool = False) -> bool:  # noqa: FBT001,FBT002,ARG002
        lo = PNL_EDGES[state[0] - 1] if state[0] > 0 else -1.0
        return lo <= -self.pct


def fixed_stop_policy(pct: float) -> dict:
    return FixedStop(pct)


def summarize(name: str, eps: list[Episode], rewards: list[float], lam: float) -> dict:
    arr = np.array(rewards)
    days = day_records(eps, rewards)
    nets = [d["net"] for d in days]
    return {
        "name": name,
        "bp": float(arr.mean() * 1e4),
        "win": float((arr > 0).mean()),
        "worst_trade_bp": float(arr.min() * 1e4),
        "worst_day": float(min(nets)),
        "lam_score": float(sum(x if x > 0 else lam * x for x in nets)),
        "total": float(sum(nets)),
    }


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=FsPath, required=True)
    ap.add_argument("--dash-paths", type=FsPath, help="frozen-policy validation set")
    ap.add_argument("--out", type=FsPath, required=True)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--min-visits", type=int, default=50)
    ap.add_argument(
        "--no-volume", action="store_true", help="ablation: drop the turnover state dimension"
    )
    ap.add_argument(
        "--hold-to-close-upside",
        action="store_true",
        help="turn the VWAP-break exit off so winners run; the stop is then the only exit that varies",
    )
    args = ap.parse_args()

    from bayes_judge import judge

    rows = [json.loads(line) for line in args.paths.read_text().splitlines() if line]
    up = not args.hold_to_close_upside
    vol = not args.no_volume
    eps = [Episode(r, up, vol) for r in rows]
    eps = [e for e in eps if e.states]
    months = sorted({e.day[:7] for e in eps})
    print(f"state: pnl, drawdown, slope, held{', turnover' if vol else ' (volume ABLATED)'}")
    print(
        f"upside exit: {'VWAP break + 15:00 (live default)' if up else '15:00 flatten only (winners run)'}"
    )
    print(f"episodes={len(eps)} days={len({e.day for e in eps})} months={months}")
    live_all = [evaluate(e, None) for e in eps]
    print(
        f"live as-is (no stop): {np.mean(live_all) * 1e4:+.0f} bp, win {np.mean(np.array(live_all) > 0):.3f}"
    )

    print(f"\n=== monthly time-ordered folds (lam={args.lam}, min_visits={args.min_visits})")
    print(
        f"{'fold':9s} {'n':>5s} {'live bp':>9s} {'fixed bp':>9s} {'learned bp':>11s} {'cut%':>6s} {'states':>7s}"
    )
    cand, base, fixed_days = [], [], []
    fold_info = []
    for m in months[1:]:
        tr = [e for e in eps if e.day[:7] < m]
        ev = [e for e in eps if e.day[:7] == m]
        policy, visits = learn(tr, args.lam, args.min_visits)
        # best fixed stop, chosen on the training months only
        best_p, best_v = None, None
        for pct in (0.01, 0.015, 0.02, 0.03, 0.05):
            r = [evaluate(e, fixed_stop_policy(pct)) for e in tr]
            d = day_records(tr, r)
            # LATENT COMPOSITION: this selects on a day-level downside statistic,
            # the same shape as `bayes_judge`'s `worst_day` criterion. The results
            # below are reported descriptively, so nothing is currently gated on a
            # statistic this selected. Routing a candidate chosen here into the
            # gate would create a select-on-X / gate-on-X composition the gate
            # cannot detect. Weight by trade count first, or gate on `loser_ratio`.
            v = sum(x["net"] if x["net"] > 0 else args.lam * x["net"] for x in d)
            if best_v is None or v > best_v:
                best_p, best_v = pct, v
        r_live = [evaluate(e, None) for e in ev]
        r_fix = [evaluate(e, fixed_stop_policy(best_p)) for e in ev]
        r_learn = [evaluate(e, policy) for e in ev]
        cut_rate = float(np.mean([a != b for a, b in zip(r_live, r_learn, strict=True)]))
        cand += day_records(ev, r_learn)
        base += day_records(ev, r_live)
        fixed_days += day_records(ev, r_fix)
        fold_info.append(
            {
                "fold": m,
                "n": len(ev),
                "fixed_pct": best_p,
                "live_bp": float(np.mean(r_live) * 1e4),
                "fixed_bp": float(np.mean(r_fix) * 1e4),
                "learned_bp": float(np.mean(r_learn) * 1e4),
                "cut_rate": cut_rate,
                "states_used": sum(1 for v in policy.values() if v),
                "states_total": len(policy),
            }
        )
        print(
            f"{m:9s} {len(ev):>5d} {np.mean(r_live) * 1e4:>+9.0f} {np.mean(r_fix) * 1e4:>+9.0f}"
            f" {np.mean(r_learn) * 1e4:>+11.0f} {cut_rate:>6.3f}"
            f" {sum(1 for v in policy.values() if v):>3d}/{len(policy):<3d}"
        )

    v_live = judge({c["day"]: c for c in cand}, {b["day"]: b for b in base})
    v_fix = judge({c["day"]: c for c in cand}, {b["day"]: b for b in fixed_days})
    for label, v in (("learned vs live", v_live), ("learned vs best fixed", v_fix)):
        print(
            f"{label:22s} p(mu>0)={v['p_mu_gt_0']:.3f} mu={v['mu_mean']:+,.0f}"
            f" days>=0={v['days_ge_0']}/{v['n_days']}"
            f" worst {v['worst_day_candidate']:+,.0f} vs {v['worst_day_baseline']:+,.0f}"
            f" losers {v['loser_ratio_candidate']:.3f} vs {v['loser_ratio_baseline']:.3f}"
            f" adopt={v['adopt']}"
        )

    out: dict = {"folds": fold_info, "verdict_vs_live": v_live, "verdict_vs_fixed": v_fix}

    if args.dash_paths:
        print("\n=== frozen-policy validation on the dashboard pool")
        print(
            "(trained on every sweep month, applied unchanged -- a different pool AND a later period)"
        )
        d_rows = [json.loads(line) for line in args.dash_paths.read_text().splitlines() if line]
        d_eps = [e for e in (Episode(r, up, vol) for r in d_rows) if e.states]
        policy, visits = learn(eps, args.lam, args.min_visits)
        best_p, best_v = None, None
        for pct in (0.01, 0.015, 0.02, 0.03, 0.05):
            r = [evaluate(e, fixed_stop_policy(pct)) for e in eps]
            d = day_records(eps, r)
            v = sum(x["net"] if x["net"] > 0 else args.lam * x["net"] for x in d)
            if best_v is None or v > best_v:
                best_p, best_v = pct, v
        rows_out = [
            summarize("live as-is (no stop)", d_eps, [evaluate(e, None) for e in d_eps], args.lam),
            summarize(
                f"best fixed stop ({best_p:.1%})",
                d_eps,
                [evaluate(e, fixed_stop_policy(best_p)) for e in d_eps],
                args.lam,
            ),
            summarize("learned stop", d_eps, [evaluate(e, policy) for e in d_eps], args.lam),
        ]
        print(
            f"{'policy':26s} {'bp':>7s} {'win':>7s} {'worst trade':>12s} {'worst day':>11s} {'lam-score':>11s}"
        )
        for r in rows_out:
            print(
                f"{r['name']:26s} {r['bp']:>+7.0f} {r['win']:>7.3f}"
                f" {r['worst_trade_bp']:>+12.0f} {r['worst_day']:>+11,.0f} {r['lam_score']:>+11,.0f}"
            )
        active = sorted((s for s, cut in policy.items() if cut), key=lambda s: -visits.get(s, 0))
        print(
            f"\nlearned policy: cuts in {len(active)} of {len(policy)} states with >= {args.min_visits} visits"
        )
        print(f"{'(pnl,dd,slope,held)':24s} {'visits':>8s}")
        for s in active[:10]:
            print(f"{str(s):24s} {visits.get(s, 0):>8d}")
        out["dashboard"] = rows_out
        out["policy"] = {str(k): v for k, v in policy.items()}
        out["visits"] = {str(k): v for k, v in visits.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
