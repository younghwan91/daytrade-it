"""Stage 2: entry timing within stage-1-selected news -- a learned stopping policy.

    .venv/bin/python scripts/eval/pipeline/stage2_entry.py \
        --in data/eval/pipeline/stage1_oof.jsonl --out data/eval/pipeline/stage2_oof.jsonl

**Problem.** From the news decision minute (`buy_minute`) up to the live
`entry_end`, every completed minute offers one choice: buy now, or keep
watching. The last candidate minute is FORCED -- stage 2 decides WHEN, never
WHETHER; whether is stage 1's job. The immediate buy at `buy_minute` at the
recorded `buy_price` is the first candidate, so the baseline is inside the
action set.

**Policy.** `timing_core.Policy`: a logit over the minute's state, trained on
the exact expected reward of the stochastic stopping policy, evaluated
deterministically. No threshold, window or holding cap is an output; the
decision is the learned function.

**Reward.** Net return of the trade entered at that minute and exited by the
LIVE exit (`--exit live`: `exit_fast.first_exit` under `TradingLimits()`, i.e.
VWAP-break + flatten) or held to the flatten (`--exit hold`), less
`costs.round_trip` at the entry minute. No add-on: pyramiding is off live.

**Folds.** Expanding window by calendar month: each month is scored by a model
fitted only on earlier months; the first two months are burn-in.

**Noise.** Beside every candidate, the identical procedure on features whose
alignment with their own path is destroyed by a fixed within-day donor
permutation (clock channels kept), `--noise` seeds fixed in advance.
Best-of-N on noise is the bar a candidate has to clear.

**Stage-1 link.** When `--in` exists, the population is stage 1's `selection`
split (out-of-fold scores, never in-sample) and all six target scores are state
inputs; `--no-s1` keeps the population and withholds the scores (ablation).
When it does not exist, the population is every path and the run says so.

**Evidence window.** Forward-holdout days never enter. Spent windows (consumed,
stage 1's confirm range) are fitted on and reported under "all months", never
as evidence. Search and evidence months, the win-probability candidates and the
threshold rule are frozen in `PREREG` before any run.

**Win probability.** Per candidate, a calibrated out-of-fold P(net > 0) of the
immediate trade; a candidate is traded only when it clears the pre-registered
threshold (abstain is the default), and the timing/exit numbers are reported
on that selected set as well as on everyone.

**Baselines.** Immediate (the recorded open print at the news minute) and
next-minute (the next bar's open), so a gain that is only "skip the noisy open
print" is not claimed as timing; results are also split 09:00 vs later news.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from decimal import Decimal
from multiprocessing import get_context
from pathlib import Path

# Single-threaded BLAS on the shared node, set before numpy loads.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing_core as tc  # noqa: E402

STAGE = 2
FEATURES = (
    "r_news",  # (close / news price - 1) / sigma
    "dd_hi",  # (close / high since news - 1) / sigma
    "up_lo",  # (close / low since news - 1) / sigma
    "slope3",  # 3-bar return / sigma
    "slope10",  # 10-bar return / sigma
    "vwap_gap",  # (close / session VWAP - 1) / sigma
    "vsurge",  # log(last-3 mean volume / session mean volume per bar)
    "vtrend",  # log(last-3 / previous-3 mean volume)
    "r_open",  # (close / day open - 1) / sigma
    "bar_rng",  # last bar (high - low) / close / sigma
    "gap_prev",  # (day open / prev close - 1) / own daily vol
    "day_ret",  # (close / prev close - 1) / own daily vol
    "elapsed",  # minutes since the news / 60        (clock)
    "tod",  # (minute - 09:00) / 390                    (clock)
)
CLOCK = ("elapsed", "tod")
#: Every stage-1 target's out-of-fold score is a state input. Which one matters
#: is left to the fit rather than picked by reading stage 1's selection/confirm
#: results (lead instruction). Scores sit on a common ~+-0.1 scale, so one
#: FIXED factor (not estimated) brings them to the other inputs' range.
S1_TARGETS = ("mfe", "ret_close", "mfe_n", "ret_n", "rr", "lrr")
S1_SCALE = 10.0
MODELS = {
    # Fixed before the first fit; not searched.
    "linear": {"hidden": 0, "iters": 300, "lr": 0.05, "batch": 3000, "l2": 1e-3},
    "mlp16": {"hidden": 16, "iters": 400, "lr": 0.01, "batch": 3000, "l2": 1e-3},
}
BURN_IN_MONTHS = 2


# ------------------------------------------------------------------ book


def build(bars: tc.Bars, ax: dict, keep: np.ndarray, s1: dict):
    """Entry stopping book over the paths `keep`, plus per-step live/hold rewards."""
    from exit_fast import barrier  # noqa: PLC0415

    gap = float(tc.LIVE.vwap_exit_gap)
    min_gain = tc.LIVE.vwap_exit_min_gain
    Xs, R_live, R_hold, legal, minute, fill, pidx, ex_min, ex_fill = ([] for _ in range(9))
    starts, stops, kept = [], [], []
    market, dvol_p = [], []
    off = 0
    names = FEATURES + (tuple(f"s1_{t}" for t in S1_TARGETS) if s1 is not None else ())
    for i in keep:
        mi, o, h, lo, c, v = bars.path(i)
        bm, bp = int(bars.buy_minute[i]), float(bars.buy_price[i])
        if bm >= tc.ENTRY_END_MINUTE:
            continue  # live refuses the entry
        G = tc.exit_grid(mi, o, c, bm)
        if G is None:
            continue
        kG, jG, fG = G
        day, tic = str(bars.day[i]), str(bars.ticker[i])
        mk = ax["markets"].get(tic, "거래소")
        pc = ax["prev_close"].get(f"{day}|{tic}")
        dv = ax["dvol"].get(f"{day}|{tic}", np.nan)

        # candidates: the news minute at buy_price, then every later bar minute
        # before entry_end that still leaves at least one exit decision minute
        later = np.flatnonzero((mi > bm) & (mi < tc.ENTRY_END_MINUTE))
        cm = np.concatenate([[bm], mi[later]]).astype(np.int64)
        cf = np.concatenate([[bp], o[later]])
        has_exit = mi[kG][-1] > cm
        cm, cf = cm[has_exit], cf[has_exit]
        if cm.size == 0 or cm[0] != bm:
            continue

        # ---- live exit from every candidate: first VWAP break after it, else flatten
        vw, valid = tc.vwap_series(c, v)
        with np.errstate(divide="ignore", invalid="ignore"):
            condv = (c[jG] / vw[jG] - 1 < -gap) & valid[jG]
        B = barrier(cf, min_gain, +1)
        mk_min = mi[kG]
        M = (mk_min[None, :] > cm[:, None]) & condv[None, :] & (c[jG][None, :] >= B[:, None])
        anyf = M.any(1)
        g = np.where(anyf, M.argmax(1), kG.size - 1)
        cost = np.array([tc.cost_of(day, mk, int(m)) for m in cm])
        R_live.append(fG[g] / cf - 1.0 - cost)
        R_hold.append(fG[-1] / cf - 1.0 - cost)
        ex_min.append(mk_min[g])
        ex_fill.append(fG[g])

        # ---- state at each candidate, from bars strictly before it
        j = np.searchsorted(mi, cm, side="left") - 1
        has = j >= 0
        jj = np.maximum(j, 0)
        sig = np.where(has, tc.running_sigma(c)[jj], tc.SIGMA_FLOOR)
        cj = c[jj]
        after_news = mi >= bm
        rmax = np.maximum.accumulate(np.where(after_news, h, -np.inf))[jj]
        rmin = np.minimum.accumulate(np.where(after_news, lo, np.inf))[jj]
        cum_vol = np.cumsum(v)
        mean_vol = cum_vol[jj] / (jj + 1)
        w3 = tc.trailing_mean(v, jj, 3)
        p3 = tc.trailing_mean(v, np.maximum(jj - 3, 0), 3)
        c3 = c[np.maximum(jj - 3, 0)]
        c10 = c[np.maximum(jj - 10, 0)]
        with np.errstate(divide="ignore", invalid="ignore"):
            f = {
                "r_news": (cj / bp - 1) / sig,
                "dd_hi": np.where(np.isfinite(rmax), cj / rmax - 1, 0.0) / sig,
                "up_lo": np.where(np.isfinite(rmin), cj / rmin - 1, 0.0) / sig,
                "slope3": np.where(jj >= 3, cj / c3 - 1, 0.0) / sig,
                "slope10": np.where(jj >= 10, cj / c10 - 1, 0.0) / sig,
                "vwap_gap": np.where(valid[jj], cj / vw[jj] - 1, 0.0) / sig,
                "vsurge": np.log(np.where(mean_vol > 0, w3 / mean_vol, 1.0) + 1e-3),
                "vtrend": np.where(jj >= 5, np.log((w3 + 1.0) / (p3 + 1.0)), 0.0),
                "r_open": (cj / o[0] - 1) / sig,
                "bar_rng": (h[jj] - lo[jj]) / cj / sig,
                "gap_prev": np.full(
                    cm.size, (o[0] / pc - 1) / dv if pc and np.isfinite(dv) else 0.0
                ),
                "day_ret": ((cj / pc - 1) / dv) if pc and np.isfinite(dv) else np.zeros(cm.size),
            }
        for k in list(f):
            f[k] = (
                tc._squash(np.where(has, f[k], 0.0))
                if k not in ("vsurge", "vtrend")
                else np.clip(np.where(has, np.nan_to_num(f[k]), 0.0), -4, 4)
            )
        f["elapsed"] = (cm - bm) / 60.0
        f["tod"] = (cm - tc.OPEN_MINUTE) / 390.0
        if s1 is not None:
            sc = s1[str(bars.key[i])]
            for t, val in zip(S1_TARGETS, sc, strict=True):
                f[f"s1_{t}"] = np.full(cm.size, np.clip(S1_SCALE * val, -4, 4))
        Xs.append(np.stack([f[n] for n in names], 1).astype(np.float32))
        lg = np.ones(cm.size, bool)
        legal.append(lg)
        minute.append(cm)
        fill.append(cf)
        pidx.append(np.full(cm.size, i))
        starts.append(off)
        off += cm.size
        stops.append(off)
        kept.append(i)
        market.append(mk)
        dvol_p.append(dv)

    cat = np.concatenate
    book = tc.Book(
        starts=np.asarray(starts),
        stops=np.asarray(stops),
        X=cat(Xs),
        R=cat(R_live),
        legal=cat(legal),
        minute=cat(minute),
        fill=cat(fill),
        path=cat(pidx),
        names=names,
    )
    extra = {
        "R_live": book.R,
        "R_hold": cat(R_hold),
        "ex_min": cat(ex_min),
        "ex_fill": cat(ex_fill),
        "kept": np.asarray(kept),
        "market": np.asarray(market),
        "dvol": np.asarray(dvol_p),
    }
    return book, extra


# ------------------------------------------------------------------ gate

_G: dict = {}


def _gate_chunk(ids: list[int]) -> list[tuple]:
    from exit_fast import first_exit, reduce_path  # noqa: PLC0415
    from exit_lab import Path as LabPath  # noqa: PLC0415
    from rl_exits import ORB_WIDTHS  # noqa: PLC0415

    bars, book, extra = _G["bars"], _G["book"], _G["extra"]
    bad = []
    for p in ids:
        i = int(extra["kept"][p])
        lp = LabPath.from_row(bars.row(i))
        for t in range(book.starts[p], book.stops[p]):
            m = int(book.minute[t])
            red = reduce_path(
                lp.bars, m, Decimal(str(float(book.fill[t]))), tc.FLATTEN_MINUTE, ORB_WIDTHS
            )
            if red is None:
                bad.append((p, t, "none"))
                continue
            px, _reason, em = first_exit(red, tc.LIVE, ORB_WIDTHS)
            if em != int(extra["ex_min"][t]) or px != float(extra["ex_fill"][t]):
                bad.append(
                    (p, t, f"{em}/{px} vs {int(extra['ex_min'][t])}/{float(extra['ex_fill'][t])}")
                )
    return bad


def gate(bars, book, extra, procs: int, cache: Path) -> dict:
    """EVERY (path, candidate entry): simulated live exit == `exit_fast.first_exit`.

    `exit_fast.first_exit` is itself held against the live `exit_rule_hit` by
    `rlnews_verify_twin.py`, so this places the stage-2 reward against live
    code transitively. Cached by the book's content hash.
    """
    import hashlib  # noqa: PLC0415

    hsh = hashlib.md5(
        book.minute.tobytes() + book.fill.tobytes() + extra["ex_min"].tobytes()
    ).hexdigest()[:12]
    if cache.exists():
        prev = json.loads(cache.read_text())
        if prev.get("hash") == hsh:
            tc.log(f"gate cached: {prev['mismatches']} mismatches over {prev['steps']} steps")
            return prev
    _G.update(bars=bars, book=book, extra=extra)
    ids = np.array_split(np.arange(book.n_paths), procs * 8)
    t0 = time.time()
    with get_context("fork").Pool(procs) as pool:
        bad = [b for part in pool.imap_unordered(_gate_chunk, [list(x) for x in ids]) for b in part]
    last_min = np.repeat(np.array([extra["ex_min"][s - 1] for s in book.stops]), book.lens())
    fired = float((extra["ex_min"] < last_min).mean())
    out = {
        "hash": hsh,
        "paths": int(book.n_paths),
        "steps": int(book.n_steps),
        "mismatches": len(bad),
        "frac_steps_exiting_before_flatten": round(fired, 4),
        "examples": [list(map(str, b)) for b in bad[:10]],
        "seconds": round(time.time() - t0),
    }
    cache.write_text(json.dumps(out, indent=1))
    tc.log(
        f"gate: {len(bad)} mismatches over {book.n_steps} (path, entry) pairs, "
        f"{fired:.1%} exit on the VWAP break before the flatten, {out['seconds']}s"
    )
    return out


PREREG = tc.REPO / "docs/research/frozen/2026-09-23-stage23-prereg.json"
WIN_GRID = np.round(np.arange(0.30, 0.8001, 0.01), 2)


def win_inputs(book: tc.Book, which: str) -> np.ndarray:
    X0 = book.X[book.starts]
    if which == "all":
        return X0
    return X0[:, [book.names.index(f"s1_{which}")]]


def choose_win_rule(Xs: dict, y, net, months, fold_months, search, evidence) -> dict:
    """The pre-registered selection: OOF P(win) per candidate input set, then the
    (candidate, threshold) with the best net per candidate on `search` months."""
    probs = {k: tc.oof_prob(X, y, months, fold_months) for k, X in Xs.items()}
    best = ("abstain", 1.01, 0.0)
    for k, p in probs.items():
        for t in WIN_GRID:
            v = float(np.where(search & (p >= t), net, 0.0)[search].mean())
            if v > best[2] or (v == best[2] and k == best[0] and t > best[1]):
                best = (k, float(t), v)
    k, t, v = best
    p = probs.get(k, np.zeros(y.size))
    sel = (p >= t) if k != "abstain" else np.zeros(y.size, bool)
    return {
        "probs": probs,
        "model": k,
        "threshold": t,
        "search_net_per_candidate_bp": 1e4 * v,
        "selected": sel,
        "evidence_net_per_candidate_bp": 1e4 * float(np.where(sel, net, 0.0)[evidence].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exit", choices=("live", "hold"), default="live")
    ap.add_argument("--models", default="linear,mlp16")
    ap.add_argument("--noise", type=int, default=10, help="shuffle-noise seeds, fixed in advance")
    ap.add_argument("--gate-procs", type=int, default=6)
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--select-only", action="store_true", help="win-probability selection only")
    ap.add_argument("--limit-days", type=int, default=0, help="smoke runs: first N days only")
    ap.add_argument("--tag", default="", help="suffix for report files (parallel runs)")
    args = ap.parse_args()
    prereg = json.loads(PREREG.read_text())

    bars = tc.load_bars(tc.price_paths(), STAGE)
    ax = tc.aux(bars)
    # Stage-1 scores are decision-time features: no label is read from --in.
    scores = {}
    with Path(args.inp).open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("score") is None or tc.excluded(r["day"]):
                continue
            assert r["split"] in ("selection", "confirm", "post_train"), r["split"]
            assert r["trained_through"] < r["day"], r["key"]
            scores[r["key"]] = [float(r[f"score_{t}"]) for t in S1_TARGETS]
    keep = np.flatnonzero(np.isin(bars.key, list(scores)))
    if args.limit_days:
        dd = sorted(set(bars.day[keep].tolist()))[: args.limit_days]
        keep = keep[np.isin(bars.day[keep], dd)]
    tc.log(
        f"population: {keep.size} path rows with a stage-1 OOF score ({len(scores)} scored pairs)"
    )

    book, extra = build(bars, ax, keep, scores)
    tc.log(
        f"book: {book.n_paths} paths, {book.n_steps} steps, median {int(np.median(book.lens()))} per path"
    )
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    g = gate(bars, book, extra, args.gate_procs, out_dir / "stage2_gate.json")
    if g["mismatches"]:
        sys.exit("stage 2 gate FAILED: simulated live exit differs from exit_fast; refusing to fit")
    if args.gate_only:
        return

    kept = extra["kept"]
    days = bars.day[kept]
    months = np.array([d[:7] for d in days])
    fold_months = sorted(set(months.tolist()))[BURN_IN_MONTHS:]
    oof = np.isin(months, fold_months)
    ev_day = np.array([tc.evidence_day(d) for d in days])
    search = np.isin(months, prereg["search_months"])
    evidence = np.isin(months, prereg["evidence_months"]) & ev_day
    assert not (search & evidence).any()
    R = extra["R_live"] if args.exit == "live" else extra["R_hold"]
    first = book.starts
    net0 = extra["R_live"][first]  # the win target is always immediate entry + live exit
    y = (net0 > 0).astype(float)

    # ---- win probability and the pre-registered selection -------------------
    Xs = {k: win_inputs(book, k) for k in prereg["win_probability"]["candidates"]}
    rule = choose_win_rule(Xs, y, net0, months, fold_months, search, evidence)
    sel = rule["selected"]
    p_ch = rule["probs"].get(rule["model"], np.full(y.size, np.nan))
    win = {
        "model": rule["model"],
        "threshold": rule["threshold"],
        "search_net_per_candidate_bp": rule["search_net_per_candidate_bp"],
        "evidence_net_per_candidate_bp": rule["evidence_net_per_candidate_bp"],
        "candidates": {},
    }
    for k, p in rule["probs"].items():
        ok = ~np.isnan(p)
        win["candidates"][k] = {
            s_name: {
                "reliability": tc.reliability(p[m], y[m]),
                "trade_curve": tc.trade_curve(p[m], net0[m], days[m], WIN_GRID[::5]),
            }
            for s_name, m in (
                ("search", search & ok),
                ("evidence", evidence & ok),
                ("all_oof", oof & ok),
            )
        }
    tc.log(
        f"win rule: {rule['model']} p>={rule['threshold']}  search {win['search_net_per_candidate_bp']:+.1f}bp/cand  "
        f"evidence {win['evidence_net_per_candidate_bp']:+.1f}bp/cand, trade rate {sel[evidence].mean():.3f}"
    )
    noise = []
    for s in range(1, args.noise + 1):
        donor = tc.within_day_donors(days, seed=2000 + s)
        rn = choose_win_rule(
            {k: X[donor] for k, X in Xs.items()},
            y,
            net0,
            months,
            fold_months,
            search,
            evidence,
        )
        noise.append(rn["evidence_net_per_candidate_bp"])
        tc.log(
            f"  win noise {s}: {rn['model']} p>={rn['threshold']} evidence {noise[-1]:+.1f}bp/cand"
        )
    win["noise_evidence_net_per_candidate_bp"] = noise
    win["beats_noise"] = bool(noise) and win["evidence_net_per_candidate_bp"] > max(noise)
    for s_name, m in (("evidence", evidence), ("all_oof", oof)):
        win[f"{s_name}_selected"] = {"trade_rate": float(sel[m].mean()), "n": int((sel & m).sum())}
        if (sel & m).sum() >= 2:
            win[f"{s_name}_selected"] |= {
                "net_per_trade": tc.day_cluster(days[sel & m], net0[sel & m]),
                "hit_rate": float(y[sel & m].mean()),
            }
    report = {
        "population": int(book.n_paths),
        "steps": int(book.n_steps),
        "exit": args.exit,
        "prereg": str(PREREG.relative_to(tc.REPO)),
        "fold_months": fold_months,
        "gate": g,
        "features": list(book.names),
        "win": win,
        "noise_seeds": args.noise,
        "models": {},
    }
    rep_path = out_dir / f"stage2_report{args.tag}.json"
    rep_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))

    masks = {
        "evidence": evidence,
        "evidence_selected": evidence & sel,
        "all_oof": oof,
        "all_oof_selected": oof & sel,
    }
    second = np.minimum(book.starts + 1, book.stops - 1)
    bases = {"immediate": first, "next_minute": second}
    best = np.array(
        [
            book.starts[p] + int(np.argmax(R[book.starts[p] : book.stops[p]]))
            for p in range(book.n_paths)
        ]
    )
    report["baselines"] = {}
    for mname, m in masks.items():
        if m.sum() < 2:
            continue
        report["baselines"][mname] = {
            "immediate": tc.day_cluster(days[m], R[first][m]),
            "next_minute": tc.day_cluster(days[m], R[second][m]),
            "last_candidate": tc.day_cluster(days[m], R[book.stops - 1][m]),
            "hindsight_best": tc.day_cluster(days[m], R[best][m]),
        }
        tc.log(
            f"baselines [{mname}] "
            + "  ".join(
                f"{k} {v['bp']:+.1f}+-{v['se_bp']:.1f}"
                for k, v in report["baselines"][mname].items()
            )
        )
    rep_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    if args.select_only:
        return

    ge = tc.grad_check()
    tc.log(f"trainer grad check: max rel err {ge:.2e}")
    if ge > 1e-5:
        sys.exit("analytic stopping gradient disagrees with finite differences; refusing to fit")
    dvol = np.nan_to_num(extra["dvol"], nan=np.nanmedian(extra["dvol"]))

    def weights(train):
        return tc.wls_weights(dvol, R[book.starts], train)

    clock_cols = [book.names.index(c) for c in CLOCK]
    scr_cols = [
        k
        for k in range(len(book.names))
        if k not in clock_cols and not book.names[k].startswith("s1_")
    ]
    at_open = np.array([int(bars.buy_minute[i]) == tc.OPEN_MINUTE for i in kept])
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
            masks,
        )
        for grp, gm in (("news_at_0900", at_open), ("news_after_0900", ~at_open)):
            m = evidence & gm
            real[f"evidence_{grp}"] = {
                "n": int(m.sum()),
                **{
                    f"det_vs_{b}": tc.day_cluster(days[m], (R[ch] - R[st])[m])
                    for b, st in bases.items()
                },
            }
        report["models"][name] = real
        choices[name] = ch
        rep_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))

    chosen = next(
        (n for n in args.models.split(",") if tc.passes(report["models"][n])), "immediate"
    )
    report["chosen"] = chosen
    if chosen != "immediate":
        all_w = weights(np.ones(book.n_paths, bool))
        (out_dir / f"stage2_model{args.tag}.json").write_text(
            json.dumps(tc.fit_final(book, R, all_w, MODELS[chosen]))
        )
    rep_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))

    with Path(args.out).open("w") as f:
        for p in range(book.n_paths):
            i = int(kept[p])
            ch = (
                choices[chosen][p]
                if chosen != "immediate" and choices[chosen][p] >= 0
                else first[p]
            )
            rec = {
                "key": str(bars.key[i]),
                "day": str(bars.day[i]),
                "ticker": str(bars.ticker[i]),
                "market": str(extra["market"][p]),
                "month": months[p],
                "split": "oof" if oof[p] else "burnin",
                "evidence": bool(evidence[p]),
                "search": bool(search[p]),
                "p_win": None if np.isnan(p_ch[p]) else float(p_ch[p]),
                "win_model": rule["model"],
                "selected": bool(sel[p]),
                "news_minute": int(bars.buy_minute[i]),
                "news_price": float(bars.buy_price[i]),
                "entry_rule": chosen,
                "entry_minute": int(book.minute[ch]),
                "entry_price": float(book.fill[ch]),
                "entry_cost": float(
                    tc.cost_of(str(bars.day[i]), str(extra["market"][p]), int(book.minute[ch]))
                ),
            }
            rec |= {f"s1_{t}": v for t, v in zip(S1_TARGETS, scores[str(bars.key[i])], strict=True)}
            for name, chs in choices.items():
                rec[f"entry_minute_{name}"] = int(book.minute[chs[p] if chs[p] >= 0 else first[p]])
            f.write(json.dumps(rec) + "\n")
    tc.log(f"chosen entry rule: {chosen}; wrote {book.n_paths} rows -> {args.out}")


if __name__ == "__main__":
    main()
