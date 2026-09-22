"""Stage 3: exit timing from stage 2's entries -- a learned stopping policy.

    .venv/bin/python scripts/eval/pipeline/stage3_exit.py \
        --in data/eval/pipeline/stage2_oof.jsonl --out data/eval/pipeline/stage3_oof.jsonl

**Problem.** From the minute after the stage-2 fill up to the live flatten,
every completed minute offers one choice: sell now, or keep holding. The
flatten minute is FORCED, exactly as live flattens. The decision minutes and
sell fills are `exit_fast.reduce_path`'s, so a policy that never sells is the
hold-to-flatten baseline and one that fires on the VWAP break is the live rule.

**Structural constraint.** The sell has two heads selected by the sign of the
position at the last completed close: below the entry a stop is a CUT, at or
above it a TAKE. A cut is therefore only ever taken below the entry price --
the network cannot learn a stop that fires on a winner as a "cut" -- and the
take head is what stage 3 adds over the earlier exit studies, which only had
cut/add. No add-on (pyramiding is off live).

**Everything else** -- trainer, expanding monthly folds, WLS weighting, the
within-day shuffle-noise best-of-N -- is `timing_core`, shared with stage 2.
The shuffle scrambles every channel except the clock and the position's own
P&L, which decides the head and is mechanically tied to the reward
(`seqexit_core.SCRAMBLEABLE` excludes it for the same reason). So the noise
policies can learn "cut when below entry" too, and a candidate has to beat
that, not just a blind one.

**Baselines**, side by side and each a paired, day-clustered difference: the
live rule, hold to the flatten, and the zero-parameter cut (sell at the first
minute the last close is below entry) -- the rule the earlier learned exit
tables collapsed to, which won in some months and lost to holding in others.

**Gate.** Driven by the live rule, and by the zero-parameter cut, this walk
must return `exit_fast.first_exit`'s exit minute and price on EVERY row, and
the forced last step must be the flatten; otherwise nothing is fitted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal
from pathlib import Path

# Single-threaded BLAS on the shared node, set before numpy loads.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing_core as tc  # noqa: E402

STAGE = 3
BASE = (
    "pnl",  # (close / entry - 1) / sigma_entry
    "dd",  # (close / peak since entry - 1) / sigma_entry
    "runup",  # (peak since entry / entry - 1) / sigma_entry
    "slope3",
    "slope10",
    "vwap_gap",
    "vsurge",
    "vtrend",
    "bar_rng",
    "day_ret",  # (close / prev close - 1) / own daily vol
)
CLOCK = ("elapsed", "tod")
MODELS = {
    # Fixed before the first fit; not searched. Same specs as stage 2.
    "linear": {"hidden": 0, "iters": 300, "lr": 0.05, "batch": 3000, "l2": 1e-3},
    "mlp16": {"hidden": 16, "iters": 400, "lr": 0.01, "batch": 3000, "l2": 1e-3},
}
BURN_IN_MONTHS = 2


def build(bars: tc.Bars, ax: dict, rows: list[dict], idx_of: dict, s1_names: list[str]):
    from exit_fast import barrier  # noqa: PLC0415

    gap = float(tc.LIVE.vwap_exit_gap)
    min_gain = tc.LIVE.vwap_exit_min_gain
    # Two heads by position sign: each channel appears once per head, zero on
    # the other side, plus a per-head bias. `pos`/`neg` are never scrambled.
    names = (
        tuple(f"{n}_{s}" for s in ("neg", "pos") for n in BASE + tuple(s1_names))
        + ("neg", "pos")
        + CLOCK
    )
    X, R, legal, minute, fill, pidx = ([] for _ in range(6))
    live_step, cut_step, starts, stops, keep, market = [], [], [], [], [], []
    off = 0
    for r_i, r in enumerate(rows):
        i = idx_of.get(r["key"])
        if i is None:
            continue
        mi, o, h, lo, c, v = bars.path(i)
        e_min, e_px = int(r["entry_minute"]), float(r["entry_price"])
        G = tc.exit_grid(mi, o, c, e_min)
        if G is None:
            continue
        k, j, f = G
        day, tic = r["day"], r["ticker"]
        pc = ax["prev_close"].get(f"{day}|{tic}")
        dv = ax["dvol"].get(f"{day}|{tic}", np.nan)
        cost = float(r["entry_cost"])
        vw, valid = tc.vwap_series(c, v)
        je = int(np.searchsorted(mi, e_min, side="left")) - 1
        sig = float(tc.running_sigma(c)[je]) if je >= 0 else tc.SIGMA_FLOOR
        cj = c[j]
        after = mi >= e_min + 1
        peak = np.maximum.accumulate(np.where(after, h, -np.inf))[j]
        has_peak = np.isfinite(peak)
        cum_vol = np.cumsum(v)
        mean_vol = cum_vol[j] / (j + 1)
        w3 = tc.trailing_mean(v, j, 3)
        p3 = tc.trailing_mean(v, np.maximum(j - 3, 0), 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            feat = {
                "pnl": (cj / e_px - 1) / sig,
                "dd": np.where(has_peak, cj / peak - 1, 0.0) / sig,
                "runup": np.where(has_peak, peak / e_px - 1, 0.0) / sig,
                "slope3": np.where(j >= 3, cj / c[np.maximum(j - 3, 0)] - 1, 0.0) / sig,
                "slope10": np.where(j >= 10, cj / c[np.maximum(j - 10, 0)] - 1, 0.0) / sig,
                "vwap_gap": np.where(valid[j], cj / vw[j] - 1, 0.0) / sig,
                "bar_rng": (h[j] - lo[j]) / cj / sig,
                "day_ret": ((cj / pc - 1) / dv) if pc and np.isfinite(dv) else np.zeros(j.size),
            }
            vol = {
                "vsurge": np.log(np.where(mean_vol > 0, w3 / mean_vol, 1.0) + 1e-3),
                "vtrend": np.where(j >= 5, np.log((w3 + 1.0) / (p3 + 1.0)), 0.0),
            }
        ch = {n: tc._squash(x) for n, x in feat.items()} | {
            n: np.clip(np.nan_to_num(x), -4, 4) for n, x in vol.items()
        }
        for n in s1_names:  # stage-1 OOF scores carried on the stage-2 row; fixed scale
            ch[n] = np.full(j.size, np.clip(10.0 * float(r[n]), -4, 4))
        neg = (cj < e_px).astype(float)  # strictly below the entry: a stop here is a cut
        pos = 1.0 - neg
        cols = [ch[n] * neg for n in BASE + tuple(s1_names)]
        cols += [ch[n] * pos for n in BASE + tuple(s1_names)]
        cols += [neg, pos, (mi[k] - e_min) / 60.0, (mi[k] - tc.OPEN_MINUTE) / 390.0]
        X.append(np.stack(cols, 1).astype(np.float32))
        R.append(f / e_px - 1.0 - cost)
        # live `_fires`: every stop needs a finite VWAP; the flatten is forced anyway
        legal.append(valid[j])
        minute.append(mi[k])
        fill.append(f)
        pidx.append(np.full(k.size, r_i))
        # the live rule, for the baseline and the gate
        with np.errstate(divide="ignore", invalid="ignore"):
            fire = (cj >= barrier(e_px, min_gain, +1)) & (cj / vw[j] - 1 < -gap) & valid[j]
        live_step.append(off + (int(np.argmax(fire)) if fire.any() else k.size - 1))
        # zero-parameter cut: the first minute the last close is strictly below entry
        below = (cj < e_px) & valid[j]
        cut_step.append(off + (int(np.argmax(below)) if below.any() else k.size - 1))
        starts.append(off)
        off += k.size
        stops.append(off)
        keep.append(r_i)
        market.append(r["market"])
    cat = np.concatenate
    book = tc.Book(
        np.asarray(starts),
        np.asarray(stops),
        cat(X),
        cat(R),
        cat(legal),
        cat(minute),
        cat(fill),
        cat(pidx),
        names,
    )
    return book, np.asarray(live_step), np.asarray(cut_step), np.asarray(keep)


#: The zero-parameter cut as a live `TradingLimits`: every rule off except a
#: stop-loss at 1e-12. `exit_fast.barrier` evaluates `entry * (1 - 1e-12)` in
#: Decimal, so on whole-KRW prices `close <= barrier` is exactly `close < entry`.
CUT_PCT = Decimal("1e-12")


def gate(bars, rows, idx_of, book, live_step, cut_step, keep) -> dict:
    """Both rule baselines against `exit_fast.first_exit`, on EVERY row: the live
    rule under `TradingLimits()` and the zero-parameter cut under a bare
    stop-loss; plus the forced last step must be the flatten minute."""
    import dataclasses  # noqa: PLC0415

    from exit_fast import first_exit, reduce_path  # noqa: PLC0415
    from exit_lab import Path as LabPath  # noqa: PLC0415
    from rl_exits import ORB_WIDTHS  # noqa: PLC0415

    bare = dataclasses.replace(
        tc.LIVE,
        stop_loss_pct=CUT_PCT,
        stop_vwap_gap=None,
        stop_time=None,
        stop_orb=None,
        take_profit_pct=None,
        trail_after_gain=None,
        vwap_exit_gap=None,
    )
    bad = 0
    ex = []
    for p, r_i in enumerate(keep):
        r = rows[r_i]
        lp = LabPath.from_row(bars.row(idx_of[r["key"]]))
        red = reduce_path(
            lp.bars,
            int(r["entry_minute"]),
            Decimal(str(float(r["entry_price"]))),
            tc.FLATTEN_MINUTE,
            ORB_WIDTHS,
        )
        px, _reason, em = first_exit(red, tc.LIVE, ORB_WIDTHS)
        cpx, _r2, cem = first_exit(red, bare, ORB_WIDTHS)
        t, tc_ = live_step[p], cut_step[p]
        flat_ok = red.flatten_minute == int(book.minute[book.stops[p] - 1])
        if (
            em != int(book.minute[t])
            or px != float(book.fill[t])
            or not flat_ok
            or cem != int(book.minute[tc_])
            or cpx != float(book.fill[tc_])
        ):
            bad += 1
            if len(ex) < 10:
                ex.append(
                    f"{r['day']} {r['ticker']}: exit_fast {em}/{px} vs {int(book.minute[t])}/{float(book.fill[t])}"
                )
    fired = float((live_step < book.stops - 1).mean())
    cut = float((cut_step < book.stops - 1).mean())
    tc.log(
        f"gate: {bad} mismatches over {len(keep)} rows (live rule + zero-parameter cut); "
        f"live rule exits early on {fired:.1%}, cut on {cut:.1%}"
    )
    return {
        "rows": len(keep),
        "mismatches": bad,
        "examples": ex,
        "frac_live_rule_fires": fired,
        "frac_cut_fires": cut,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--paths", nargs="*", default=[str(p) for p in tc.DEFAULT_PATHS])
    ap.add_argument("--models", default="linear,mlp16")
    ap.add_argument("--noise", type=int, default=10)
    ap.add_argument("--gate-only", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.inp).read_text().splitlines() if line]
    tc.config.assert_no_holdout_labels({r["day"] for r in rows}, STAGE)
    bars = tc.load_bars(args.paths, STAGE)
    ax = tc.aux(bars)
    idx_of = {str(k): i for i, k in enumerate(bars.key.tolist())}
    s1_names = sorted(k for k in rows[0] if k.startswith("s1_")) if rows else []
    assert not any(tc.excluded(r["day"]) for r in rows)
    entry_rule = sorted({r["entry_rule"] for r in rows})
    tc.log(f"{len(rows)} stage-2 rows, entry rule {entry_rule}")

    book, live_step, cut_step, keep = build(bars, ax, rows, idx_of, s1_names)
    tc.log(f"book: {book.n_paths} positions, {book.n_steps} steps")
    out_dir = Path(args.out).parent
    g = gate(bars, rows, idx_of, book, live_step, cut_step, keep)
    (out_dir / "stage3_gate.json").write_text(json.dumps(g, indent=1))
    if g["mismatches"]:
        sys.exit("stage 3 gate FAILED: simulated live exit differs from exit_fast; refusing to fit")
    if args.gate_only:
        return
    ge = tc.grad_check()
    tc.log(f"trainer grad check: max rel err {ge:.2e}")
    if ge > 1e-5:
        sys.exit("analytic stopping gradient disagrees with finite differences; refusing to fit")

    days = np.array([rows[r]["day"] for r in keep])
    months = np.array([d[:7] for d in days])
    fold_months = sorted(set(months.tolist()))[BURN_IN_MONTHS:]
    oof = np.isin(months, fold_months)
    R = book.R
    hold = book.stops - 1
    best = np.array(
        [
            book.starts[p] + int(np.argmax(R[book.starts[p] : book.stops[p]]))
            for p in range(book.n_paths)
        ]
    )
    dvol = np.array([ax["dvol"].get(f"{rows[r]['day']}|{rows[r]['ticker']}", np.nan) for r in keep])
    dvol = np.nan_to_num(dvol, nan=np.nanmedian(dvol))

    def weights(train):
        return tc.wls_weights(dvol, R[hold], train)

    def summ(step, label):
        return tc.day_cluster(days[oof], R[step][oof]) | {"label": label}

    report = {
        "entry_rule": entry_rule,
        "positions": int(book.n_paths),
        "steps": int(book.n_steps),
        "oof_positions": int(oof.sum()),
        "oof_days": int(np.unique(days[oof]).size),
        "fold_months": fold_months,
        "gate": g,
        "features": list(book.names),
        "noise_seeds": args.noise,
        "baselines": {
            "hold": summ(hold, "hold to the live flatten"),
            "live_rule": summ(live_step, "live exit: VWAP break + flatten"),
            "first_below": summ(
                cut_step, "zero-parameter cut: first minute the close is below entry"
            ),
            "hindsight_best": summ(best, "hindsight-best exit (upper bound, not a policy)"),
        },
        "models": {},
    }
    bases = {"live_rule": live_step, "hold": hold, "first_below": cut_step}
    for b in ("live_rule", "first_below"):
        report["baselines"][b]["diff_vs_hold"] = tc.day_cluster(
            days[oof], (R[bases[b]] - R[hold])[oof]
        )
    for k, v in report["baselines"].items():
        tc.log(f"baseline {k}: {v['bp']:+.1f}bp +- {v['se_bp']:.1f}")

    keep_cols = {"neg", "pos", *CLOCK} | {n for n in book.names if n.startswith(("pnl_", "s1_"))}
    scr_cols = [c for c, n in enumerate(book.names) if n not in keep_cols]
    rep_path = out_dir / "stage3_report.json"
    choices = {}
    for name in args.models.split(","):
        real, ch = tc.evaluate(
            name,
            book,
            R,
            bases,
            days,
            months,
            fold_months,
            MODELS[name],
            weights,
            scr_cols,
            args.noise,
        )
        real["frac_cut"] = float(
            ((book.X[ch, book.names.index("neg")] > 0) & (ch < hold))[oof].mean()
        )
        real["frac_take"] = float(
            ((book.X[ch, book.names.index("pos")] > 0) & (ch < hold))[oof].mean()
        )
        report["models"][name] = real
        choices[name] = ch
        rep_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))

    # Pre-registered: noise best-of-N AND > 2 SE above live rule, hold AND the cut.
    chosen = next(
        (n for n in args.models.split(",") if tc.passes(report["models"][n])), "live_rule"
    )
    report["chosen"] = chosen
    rep_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    pick = choices.get(chosen, live_step)
    with Path(args.out).open("w") as f:
        for p, r_i in enumerate(keep):
            r = rows[r_i]
            t = pick[p] if (oof[p] and pick[p] >= 0) else live_step[p]
            rec = {
                k: r[k]
                for k in (
                    "key",
                    "day",
                    "ticker",
                    "market",
                    "month",
                    "entry_rule",
                    "entry_minute",
                    "entry_price",
                    "entry_cost",
                )
            }
            rec |= {
                "split": "oof" if oof[p] else "burnin",
                "exit_rule": chosen if oof[p] else "live_rule",
                "exit_minute": int(book.minute[t]),
                "exit_price": float(book.fill[t]),
                "net": float(R[t]),
                "net_live_rule": float(R[live_step[p]]),
                "net_hold": float(R[hold[p]]),
                "net_first_below": float(R[cut_step[p]]),
            }
            f.write(json.dumps(rec) + "\n")
    tc.log(f"chosen exit rule: {chosen}; wrote {len(keep)} rows -> {args.out}")


if __name__ == "__main__":
    main()
