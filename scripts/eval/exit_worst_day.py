"""Was the sigma-scaled exit rejected by a defect in the worst-day gate?

`bayes_judge` gate 2 is `worst_day_candidate >= worst_day_baseline`, where each
side is `min(net)` over the fold's days. That is an extreme order statistic: on
84 days it is decided by exactly one of them. The sigma-scaled contextual exit
failed it by 16,254 KRW (-230,261 vs -214,007) while clearing p(mu>0) and
improving the loser ratio, so the whole rejection rests on that one day.

Reproduce against `exits3.json`, the last of the three recorded runs: the
`TAKE_PROFIT` grid was extended to +8%/+10% after `exits.json` and `exits2.json`
were written, which moves the 2026-05 fixed-arm choice and with it p(mu>0)
(0.9139 -> 0.9285) and days_ge_0 (45 -> 44). The worst-day pair is -230,261 vs
-214,007 in all three runs, so the gate that did the rejecting is unaffected.

Two distinct defects could produce a wrong rejection, and they call for
different repairs:

  raw      `min(net_d)`              -- the gate as it stands. One day decides.
  weighted `min(net_d / trades_d)`   -- the worst day's loss PER TRADE. Removes
                                        the size confound: at equal per-trade
                                        quality a 75-trade day posts a larger
                                        |net| than a 17-trade day purely
                                        because it traded more.
  robust   `quantile(net_d, 0.05)`   -- the 5th percentile of the daily-net
                                        distribution. Still a left-tail
                                        statistic, but read off ~4 days instead
                                        of 1.

All three keep the same gate direction (candidate >= baseline). If `weighted`
and `robust` agree with `raw`, the rejection stands on its merits. If they
disagree with each other, the one that flips names the defect that was
operating.

Stages 4 and 5 of `rl_exits.py` are re-run here to recover the per-day series,
which `exits.json` does not store. The regenerated verdict is asserted equal to
the published one before anything is concluded from it.

`--costs real` additionally re-prices every reward at the broker's own round
trip (`costs.round_trip`, 37-54 bp by market and by whether the fill is inside
the first five minutes) instead of the `replay_baseline` literal that
`Stack.rewards` applies (`SELL_COST + BUY_COST` = 23.0 bp, the legacy constant
with spread and slippage at zero). Both arms trade the same paths on the same
days, so a per-trade cost cancels out of the daily DELTA and cannot move
p(mu>0), mu or days_ge_0; it does move the daily NET levels, which is exactly
what the worst-day gate reads. Running it both ways says whether the gate's
verdict survives the correct cost.

uv run python scripts/eval/exit_worst_day.py \
    --paths ~/git/daytrade-it-rl/data/eval/rl_minimal/paths.jsonl \
    --published ~/git/daytrade-it-rl/data/eval/rl_minimal/exits.json \
    --out data/eval/rlnews/worst_day.json                              # simnode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as FsPath

import numpy as np

sys.path.insert(0, str(FsPath(__file__).resolve().parent))

from bayes_judge import judge  # noqa: E402
from costs import round_trip  # noqa: E402
from exit_fast import reduce_path  # noqa: E402
from exit_lab import FLATTEN, Path  # noqa: E402
from replay_baseline import BUY_COST, SELL_COST  # noqa: E402
from rl_exits import (  # noqa: E402
    ORB_WIDTHS,
    SIGMA_STOP,
    SIGMA_TP,
    Stack,
    day_records,
    entry_sigma,
    grid,
    objective,
)

from daytrade_it.application.trading.auto_trader import TradingLimits  # noqa: E402

#: Tolerance for the reproduction gate. The re-run is deterministic, so this is
#: floating-point slack, not a fudge factor.
EXACT = 1e-9

#: A fill in the first five minutes crosses the wider opening spread.
OPEN_WINDOW_MIN = 5


def real_cost(rows: list[dict], markets: dict[str, str]) -> np.ndarray:
    """Round-trip cost rate per path, from the broker's model rather than a literal."""
    return np.array(
        [
            round_trip(
                r["day"],
                markets.get(r["ticker"], "거래소"),
                at_open=(r["buy_minute"] - 9 * 60) < OPEN_WINDOW_MIN,
            )
            for r in rows
        ]
    )


def recost(reward: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """`reward` (charged the legacy literal) re-charged at `cost`.

    `Stack.rewards` returns ``g * (1 - SELL) - (1 + BUY)`` for a gross ratio
    ``g = proceeds / buy``. Invert for ``g``, then return the gross return less
    the real round trip, which is the convention `costs.round_trip` documents
    (``net = ret - round_trip(...)``).
    """
    gross = (reward + 1 + float(BUY_COST)) / (1 - float(SELL_COST))
    return gross - 1 - cost


def stock_markets() -> dict[str, str]:
    """code -> market label, from the kr_quant stock master (read-only)."""
    import edge_db  # noqa: PLC0415

    cur = edge_db.connect().cursor()
    cur.execute("SELECT code, market FROM stocks")
    return {code: market for code, market in cur.fetchall() if market}


def worst_raw(days: dict[str, dict]) -> float:
    """`min(net)` over days -- `bayes_judge`'s gate 2 exactly."""
    return min(float(v["net"]) for v in days.values())


def worst_weighted(days: dict[str, dict]) -> float:
    """`min(net / trades)` -- the worst day's loss per trade. Days with no trade skipped."""
    per = [float(v["net"]) / int(v["trades"]) for v in days.values() if int(v["trades"])]
    return min(per) if per else 0.0


def worst_robust(days: dict[str, dict], q: float = 0.05) -> float:
    """The `q` quantile of the daily nets -- a tail read off several days, not one."""
    return float(np.quantile([float(v["net"]) for v in days.values()], q))


PENALTIES = {"raw": worst_raw, "weighted": worst_weighted, "robust": worst_robust}


def gate(cand: dict[str, dict], base: dict[str, dict]) -> dict:
    """Each penalty applied to both sides, with the pass/fail the gate would give."""
    out = {}
    for name, fn in PENALTIES.items():
        c, b = fn(cand), fn(base)
        out[name] = {"candidate": c, "baseline": b, "passes": c >= b, "margin": c - b}
    return out


def bootstrap_gate(
    cand: dict[str, dict], base: dict[str, dict], n: int = 20_000, seed: int = 0
) -> dict:
    """How often each penalty passes when the days are resampled with replacement.

    The gate is a statement about the population of days, but it is computed on
    one draw of 84. Resampling days says how much of the verdict is the sample.
    """
    rng = np.random.default_rng(seed)
    keys = sorted(cand)
    c_net = np.array([float(cand[k]["net"]) for k in keys])
    b_net = np.array([float(base[k]["net"]) for k in keys])
    c_tr = np.array([int(cand[k]["trades"]) for k in keys], dtype=float)
    b_tr = np.array([int(base[k]["trades"]) for k in keys], dtype=float)
    idx = rng.integers(0, len(keys), size=(n, len(keys)))
    cn, bn, ct, bt = c_net[idx], b_net[idx], c_tr[idx], b_tr[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        cw = np.where(ct > 0, cn / ct, np.inf)
        bw = np.where(bt > 0, bn / bt, np.inf)
    return {
        "n_resamples": n,
        "raw": float((cn.min(axis=1) >= bn.min(axis=1)).mean()),
        "weighted": float((cw.min(axis=1) >= bw.min(axis=1)).mean()),
        "robust": float((np.quantile(cn, 0.05, axis=1) >= np.quantile(bn, 0.05, axis=1)).mean()),
    }


def rebuild(paths_file: FsPath, costs_mode: str = "legacy") -> dict:
    """Stages 4 and 5 of `rl_exits.main`, returning the per-day series they judge."""
    rows = [json.loads(line) for line in paths_file.read_text().splitlines() if line]
    parsed = [Path.from_row(r) for r in rows]
    flat_minute = FLATTEN.hour * 60 + FLATTEN.minute
    reduced, kept = [], []
    for p, row in zip(parsed, rows, strict=True):
        r = reduce_path(p.bars, p.buy_minute, p.buy_price, flat_minute, ORB_WIDTHS)
        if r is not None:
            reduced.append(r)
            kept.append((p, row))
    parsed = [p for p, _ in kept]
    rows = [r for _, r in kept]
    days = np.array([r["day"] for r in rows])
    months = sorted({d[:7] for d in days.tolist()})
    stack = Stack(reduced)
    configs = grid()
    live = TradingLimits()
    cost = real_cost(rows, stock_markets()) if costs_mode == "real" else None
    if cost is not None:
        print(
            f"real round trip: mean {cost.mean() * 1e4:.1f} bp,"
            f" legacy literal {(float(SELL_COST) + float(BUY_COST)) * 1e4:.1f} bp"
        )
    fix = (lambda r: r) if cost is None else (lambda r: recost(r, cost))
    lab = fix(stack.rewards(live))
    scores = {i: fix(stack.rewards(c)) for i, c in enumerate(configs)}

    sigma = np.array([entry_sigma(p) for p in parsed])
    sigma = np.where(np.isnan(sigma), float(np.nanmedian(sigma)), sigma)
    ctx_scores = {
        (a, b): fix(
            stack.rewards(
                live,
                stop_pct=None if a is None else a * sigma,
                tp_pct=None if b is None else b * sigma,
            )
        )
        for a in SIGMA_STOP
        for b in SIGMA_TP
    }

    base_days, fixed_days, ctx_days = [], [], []
    for m in months[1:]:
        tr = np.array([d[:7] < m for d in days.tolist()])
        ev = np.array([d[:7] == m for d in days.tolist()])
        fixed_best = max(scores, key=lambda i: objective(days[tr], scores[i][tr]))
        ctx_best = max(ctx_scores, key=lambda k: objective(days[tr], ctx_scores[k][tr]))
        base_days += day_records(days[ev], lab[ev])
        fixed_days += day_records(days[ev], scores[fixed_best][ev])
        ctx_days += day_records(days[ev], ctx_scores[ctx_best][ev])
    key = lambda recs: {d["day"]: d for d in recs}  # noqa: E731
    return {"live": key(base_days), "fixed": key(fixed_days), "ctx": key(ctx_days)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=FsPath, required=True)
    ap.add_argument("--published", type=FsPath, required=True)
    ap.add_argument("--out", type=FsPath, required=True)
    ap.add_argument(
        "--costs",
        choices=("legacy", "real"),
        default="legacy",
        help="legacy reproduces the recorded run; real re-prices at costs.round_trip",
    )
    args = ap.parse_args()

    # -- reproduction gate: nothing is concluded from a series that does not
    #    reproduce the published verdict it is supposed to explain. The gate is
    #    always run on the legacy-cost series, because that is the one the
    #    recorded run produced; `--costs real` is then reported beside it.
    legacy = rebuild(args.paths, "legacy")
    mine = judge(legacy["ctx"], legacy["fixed"])
    published = json.loads(args.published.read_text())["verdict_ctx_vs_fixed"]
    diffs = {
        k: (mine[k], published[k])
        for k in ("p_mu_gt_0", "mu_mean", "worst_day_candidate", "worst_day_baseline", "days_ge_0")
        if abs(float(mine[k]) - float(published[k])) > EXACT
    }
    if diffs:
        for k, (a, b) in diffs.items():
            print(f"  MISMATCH {k}: rebuilt {a!r} vs published {b!r}")
        raise SystemExit("rebuilt series does not reproduce exits.json; refusing to diagnose")
    print(f"reproduction gate: verdict_ctx_vs_fixed reproduced exactly on {mine['n_days']} days")

    series = legacy if args.costs == "legacy" else rebuild(args.paths, "real")
    ctx, fixed, live = series["ctx"], series["fixed"], series["live"]
    if args.costs == "real":
        v = judge(ctx, fixed)
        print(
            f"at the real round trip the delta is unchanged: p(mu>0)"
            f" {mine['p_mu_gt_0']:.6f} -> {v['p_mu_gt_0']:.6f},"
            f" mu {mine['mu_mean']:+,.0f} -> {v['mu_mean']:+,.0f},"
            f" days>=0 {mine['days_ge_0']} -> {v['days_ge_0']}"
        )
        mine = v

    trades = sorted(int(v["trades"]) for v in ctx.values())
    print(
        f"trades/day: min {trades[0]} p25 {trades[len(trades) // 4]}"
        f" median {trades[len(trades) // 2]} p75 {trades[3 * len(trades) // 4]}"
        f" max {trades[-1]} (spread {trades[-1] / max(trades[0], 1):.1f}x)"
    )

    result = {
        "costs": args.costs,
        "n_days": mine["n_days"],
        "verdict_ctx_vs_fixed": mine,
        "comparisons": {},
    }
    for label, cand, base in (("ctx_vs_fixed", ctx, fixed), ("ctx_vs_live", ctx, live)):
        g = gate(cand, base)
        bs = bootstrap_gate(cand, base)
        result["comparisons"][label] = {"gate": g, "bootstrap_pass_rate": bs}
        print(f"\n=== {label}")
        print(
            f"{'penalty':10s} {'candidate':>14s} {'baseline':>14s} {'margin':>12s} {'passes':>7s} {'boot':>6s}"
        )
        for name in PENALTIES:
            r = g[name]
            print(
                f"{name:10s} {r['candidate']:>14,.1f} {r['baseline']:>14,.1f}"
                f" {r['margin']:>+12,.1f} {str(r['passes']):>7s} {bs[name]:>6.3f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
