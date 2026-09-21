"""Step 2: causal (expanding-window) regime estimation, and what conditioning
on it does.

Three variants of the state path, for the same model family:

  EXPANDING  -- parameters re-fit every `--refit` days on data STRICTLY BEFORE
                the segment, then FILTERED forward. Admissible. This is the
                only one a trading conclusion may rest on.
  FULLFIT    -- one EM fit on the whole span, then FILTERED. The filter is
                causal but THE PARAMETERS SAW THE FUTURE. Optimistic variant.
  SMOOTHED   -- full-sample EM + forward-backward. Uses the whole sequence at
                every t. NOT ADMISSIBLE; reported as an upper bound only.

The confirm span (DAILY_CONFIRM_WINDOW.json, frozen before any search) is
scored exactly once, at the end, by `--stage confirm`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/young/git/daytrade-it-rlnews")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "scripts" / "eval"))
from bayes_judge import judge  # noqa: E402
from costs import round_trip  # noqa: E402
from hmm_core import (  # noqa: E402
    filter_states,
    fit_em,
    relabel,
    smooth_states,
    visit_counts,
)

SERIES = REPO / "data/eval/rlnews/hmm_daily_series.jsonl"
WIN = json.loads((REPO / "data/eval/rlnews/DAILY_CONFIRM_WINDOW.json").read_text())
BP = 10_000.0
PRIOR_SD_BP = 50.0

OBS_SETS = {
    "legs": ["o_night", "o_intra"],
    "vol": ["o_rv5", "o_rv21", "o_disp"],
    "compact": ["o_intra", "o_disp", "o_rv21", "o_breadth"],
    "legs5vol": ["o_night5", "o_intra5", "o_rv21", "o_disp"],
    "full": ["o_night", "o_intra", "o_disp", "o_breadth", "o_rv5", "o_rv21", "o_logtv", "o_dlogtv"],
}
VOL_KEY = {
    "legs": "o_intra",
    "vol": "o_rv21",
    "compact": "o_rv21",
    "legs5vol": "o_rv21",
    "full": "o_rv21",
}
KS = (2, 3, 4)


def load():
    return [json.loads(line) for line in SERIES.read_text().splitlines() if line.strip()]


def design(rows, cols):
    return np.array([[r[c] for c in cols] for r in rows], float)


def expanding_filter(rows, cols, K, burn=750, refit=63, seed=7):
    """Causal state path. alpha[t] uses obs 0..t; the parameters in force at t
    were fit on rows[0:t0] with t0 <= t. Nothing from >= t0 enters them."""
    X_raw = design(rows, cols)
    T = len(rows)
    by = cols.index(VOL_KEY_FOR(cols))
    alpha = np.full((T, K), np.nan)
    fits = []
    t0 = burn
    while t0 < T:
        t1 = min(t0 + refit, T)
        tr = X_raw[:t0]
        mu_, sd_ = tr.mean(0), tr.std(0) + 1e-12
        p = relabel(fit_em((tr - mu_) / sd_, K, seed=seed), by=by)
        a, _, _ = filter_states(p, (X_raw[:t1] - mu_) / sd_)
        alpha[t0:t1] = a[t0:t1]
        fits.append({"t0": t0, "day": rows[t0]["day"], "A": p["A"].tolist(), "n_fit": t0})
        t0 = t1
    return alpha, fits


def VOL_KEY_FOR(cols):
    for name, c in OBS_SETS.items():
        if c == cols:
            return VOL_KEY[name]
    return cols[0]


def fullfit_paths(rows, cols, K, seed=7):
    X = design(rows, cols)
    mu_, sd_ = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu_) / sd_
    p = relabel(fit_em(Z, K, seed=seed), by=cols.index(VOL_KEY_FOR(cols)))
    a, _, _ = filter_states(p, Z)
    g = smooth_states(p, Z)
    return a, g, p


def cost_bp(day):
    return round_trip(day, "거래소", at_open=True) * BP


def nets_bp(rows):
    """One equal-weight universe basket held overnight: buy close d-1, sell
    open d. Gross = t_night, one round trip charged on day d."""
    return np.array([r["t_night"] * BP - cost_bp(r["day"]) for r in rows])


def day_recs(rows, net_bp, mask):
    out = {}
    for r, n, m in zip(rows, net_bp, mask, strict=True):
        v = float(n) if m else 0.0
        out[r["day"]] = {
            "day": r["day"],
            "net": v,
            "trades": 1 if m else 0,
            "losers": 1 if (m and v < 0) else 0,
            "worst_trade": v if m else 0.0,
        }
    return out


def hard(alpha):
    return np.where(np.isnan(alpha).any(1), -1, alpha.argmax(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["search", "confirm"], default="search")
    ap.add_argument("--refit", type=int, default=63)
    ap.add_argument("--burn", type=int, default=750)
    ap.add_argument("--shuffles", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load()
    lo_s, hi_s = WIN["track_c_daily"]["search"]
    lo_c, hi_c = WIN["track_c_daily"]["confirm"]
    print(f"series {rows[0]['day']}..{rows[-1]['day']} n={len(rows)}")
    print(f"round trip on {rows[-1]['day']}: {cost_bp(rows[-1]['day']):.1f} bp")

    net_all = nets_bp(rows)
    gross_all = np.array([r["t_night"] * BP for r in rows])
    intra_all = np.array([r["t_intra"] * BP for r in rows])
    days = [r["day"] for r in rows]
    yr = np.array([int(d[:4]) for d in days])
    in_s = np.array([lo_s <= d <= hi_s for d in days])
    in_c = np.array([lo_c <= d <= hi_c for d in days])

    configs = [(o, K) for o in OBS_SETS for K in KS]
    print(
        f"\nN configurations searched = {len(configs)} "
        f"({len(OBS_SETS)} observation sets x {len(KS)} state counts)\n"
    )

    store = {}
    for o, K in configs:
        cols = OBS_SETS[o]
        a, fits = expanding_filter(rows, cols, K, burn=args.burn, refit=args.refit)
        store[(o, K)] = {"alpha": a, "fits": fits}
        print(f"  expanding filter done {o} K={K}  ({len(fits)} refits)", flush=True)

    # ---------------- the search: pick config + state subset on SEARCH span --
    results = []
    for o, K in configs:
        a = store[(o, K)]["alpha"]
        lab = hard(a)
        m = in_s & (lab >= 0)
        per = {k: net_all[m & (lab == k)] for k in range(K)}
        pick = {k for k, v in per.items() if v.size >= 20 and v.mean() > 0}
        m & np.isin(lab, sorted(pick)) if pick else np.zeros_like(m)
        delta = (net_all[m] * np.isin(lab[m], sorted(pick))).mean() - net_all[m].mean()
        d_, v_ = visit_counts(lab[m])
        results.append(
            {
                "obs": o,
                "K": K,
                "pick": sorted(pick),
                "n_search": int(m.sum()),
                "delta_bp": float(delta),
                "state_mean_net_bp": {
                    k: float(v.mean()) if v.size else None for k, v in per.items()
                },
                "state_days": d_,
                "state_visits": v_,
            }
        )
    results.sort(key=lambda r: -r["delta_bp"])
    print("\n--- search-span delta (conditioned - unconditioned), bp/day")
    for r in results:
        print(
            f"  {r['obs']:<9} K={r['K']}  delta {r['delta_bp']:+7.2f}  "
            f"pick {r['pick']}  days {r['state_days']}  visits {r['state_visits']}"
        )
    best = results[0]
    print(
        f"\nBEST config: {best['obs']} K={best['K']} pick={best['pick']} "
        f"delta={best['delta_bp']:+.2f} bp/day"
    )

    # ---------------- selection premium on shuffled targets ------------------
    rng = np.random.default_rng(11)
    np.where(in_s & (hard(store[configs[0]]["alpha"]) >= 0))[0]
    labs = {c: hard(store[c]["alpha"]) for c in configs}

    def best_of_n(net_vec, blocks=None):  # noqa: ARG001  -- kept for call-site symmetry
        bestd = -1e9
        for o, K in configs:
            lab = labs[(o, K)]
            m = in_s & (lab >= 0)
            nv = net_vec[m]
            lv = lab[m]
            per = {k: nv[lv == k] for k in range(K)}
            pick = [k for k, v in per.items() if v.size >= 20 and v.mean() > 0]
            d = (nv * np.isin(lv, pick)).mean() - nv.mean()
            bestd = max(bestd, d)
        return bestd

    def circ_block_shuffle(vec, mask, block, rng):
        out = vec.copy()
        v = vec[mask]
        n = len(v)
        nb = int(np.ceil(n / block))
        starts = rng.integers(0, n, nb)
        pieces = [np.take(v, np.arange(s, s + block) % n) for s in starts]
        out[mask] = np.concatenate(pieces)[:n]
        return out

    mask_all_s = in_s & (labs[configs[0]] >= 0)
    null_iid, null_blk = [], []
    for _i in range(args.shuffles):
        perm = net_all.copy()
        v = perm[mask_all_s]
        perm[mask_all_s] = rng.permutation(v)
        null_iid.append(best_of_n(perm))
        null_blk.append(best_of_n(circ_block_shuffle(net_all, mask_all_s, 21, rng)))
    null_iid = np.array(null_iid)
    null_blk = np.array(null_blk)
    print(
        f"\n--- selection premium, best-of-{len(configs)} on a shuffled target "
        f"({args.shuffles} draws)"
    )
    for nm, arr in (("iid permutation", null_iid), ("21-day circular block", null_blk)):
        print(
            f"  {nm:<22} mean {arr.mean():+6.2f}  p50 {np.percentile(arr, 50):+6.2f}  "
            f"p90 {np.percentile(arr, 90):+6.2f}  p95 {np.percentile(arr, 95):+6.2f}  "
            f"max {arr.max():+6.2f}   P(null >= observed) = "
            f"{float((arr >= best['delta_bp']).mean()):.3f}"
        )

    # ---------------- ordering of next-day legs by state, year by year -------
    o, K = best["obs"], best["K"]
    lab = labs[(o, K)]
    print(f"\n--- next-day legs by FILTERED state ({o} K={K}), search span, by year")
    order_rank = sorted(
        range(K),
        key=lambda k: (
            -float(gross_all[in_s & (lab == k)].mean()) if (in_s & (lab == k)).sum() else 0
        ),
    )
    print(f"  pooled search-span rank on overnight (best->worst): {order_rank}")
    ord_ok_n = ord_ok_i = ny = 0
    for y in sorted(set(yr[in_s])):
        mm = in_s & (yr == y) & (lab >= 0)
        if mm.sum() < 60:
            continue
        ny += 1
        gn = {
            k: float(gross_all[mm & (lab == k)].mean()) if (mm & (lab == k)).sum() >= 5 else None
            for k in range(K)
        }
        gi = {
            k: float(intra_all[mm & (lab == k)].mean()) if (mm & (lab == k)).sum() >= 5 else None
            for k in range(K)
        }
        cn = [gn[k] for k in order_rank if gn[k] is not None]
        ci = [gi[k] for k in order_rank if gi[k] is not None]
        okn = all(cn[i] >= cn[i + 1] for i in range(len(cn) - 1))
        oki = all(ci[i] >= ci[i + 1] for i in range(len(ci) - 1))
        ord_ok_n += okn
        ord_ok_i += oki
        print(
            f"  {y}  n={int(mm.sum()):3d}  night "
            + " ".join(f"s{k}:{'  . ' if gn[k] is None else f'{gn[k]:+6.1f}'}" for k in range(K))
            + f"  ordered={okn}   intra "
            + " ".join(f"s{k}:{'  . ' if gi[k] is None else f'{gi[k]:+6.1f}'}" for k in range(K))
            + f"  ordered={oki}"
        )
    print(f"  overnight ordering held {ord_ok_n}/{ny} years; intraday {ord_ok_i}/{ny} years")

    # ---------------- filtered vs fullfit vs smoothed ------------------------
    cols = OBS_SETS[o]
    a_f, g_s, p_full = fullfit_paths(rows, cols, K)
    lab_full, lab_sm = hard(a_f), g_s.argmax(1)
    print(f"\n--- state-path variants ({o} K={K}), search span")
    for nm, lb in (
        ("EXPANDING filtered (admissible)", lab),
        ("FULLFIT filtered (params saw future)", lab_full),
        ("SMOOTHED (NOT admissible)", lab_sm),
    ):
        m = in_s & (lb >= 0)
        per = {k: net_all[m & (lb == k)] for k in range(K)}
        pick = [k for k, v in per.items() if v.size >= 20 and v.mean() > 0]
        d = (net_all[m] * np.isin(lb[m], pick)).mean() - net_all[m].mean()
        dd, vv = visit_counts(lb[m])
        print(f"  {nm:<38} pick {pick}  delta {d:+6.2f} bp/day  days {dd} visits {vv}")
    agree = float((lab[in_s & (lab >= 0)] == lab_sm[in_s & (lab >= 0)]).mean())
    agree2 = float((lab[in_s & (lab >= 0)] == lab_full[in_s & (lab >= 0)]).mean())
    print(f"  expanding-filtered vs smoothed label agreement: {agree:.3f}")
    print(f"  expanding-filtered vs fullfit-filtered agreement: {agree2:.3f}")

    # ---------------- judge on the search span -------------------------------
    m = in_s & (lab >= 0)
    rs = [r for r, ok in zip(rows, m, strict=True) if ok]
    nb = net_all[m]
    cand = day_recs(rs, nb, np.isin(lab[m], best["pick"]))
    base = day_recs(rs, nb, np.ones(len(rs), bool))
    j = judge(cand, base, prior_sd=PRIOR_SD_BP)
    print(
        f"\n--- bayes_judge, SEARCH span (in-sample for the rule): "
        f"p_mu>0={j['p_mu_gt_0']:.3f} mu={j['mu_mean']:+.2f} bp/day "
        f"adopt={j['adopt']} n={j['n_days']}"
    )
    print(
        f"    unconditioned mean net {nb.mean():+.2f} bp/day; "
        f"conditioned {float((nb * np.isin(lab[m], best['pick'])).mean()):+.2f} bp/day"
    )

    # ---------------- HELD-OUT CONFIRM, scored once --------------------------
    mc = in_c & (lab >= 0)
    rc = [r for r, ok in zip(rows, mc, strict=True) if ok]
    nc = net_all[mc]
    candc = day_recs(rc, nc, np.isin(lab[mc], best["pick"]))
    basec = day_recs(rc, nc, np.ones(len(rc), bool))
    jc = judge(candc, basec, prior_sd=PRIOR_SD_BP)
    dc, vc = visit_counts(lab[mc])
    print(f"\n--- CONFIRM span {lo_c}..{hi_c} n={int(mc.sum())} (frozen before search)")
    print(
        f"    unconditioned mean net {nc.mean():+.2f} bp/day; "
        f"conditioned {float((nc * np.isin(lab[mc], best['pick'])).mean()):+.2f} bp/day"
    )
    print(
        f"    delta {float((nc * np.isin(lab[mc], best['pick'])).mean() - nc.mean()):+.2f} bp/day  "
        f"p_mu>0={jc['p_mu_gt_0']:.3f} adopt={jc['adopt']}"
    )
    print(f"    confirm-span state days {dc} visits {vc}")
    for k in range(K):
        s = nc[lab[mc] == k]
        gk = gross_all[mc][lab[mc] == k]
        ik = intra_all[mc][lab[mc] == k]
        if s.size:
            print(
                f"      s{k}: n={s.size:3d} night {gk.mean():+6.1f} "
                f"intra {ik.mean():+6.1f} net {s.mean():+6.1f} bp"
            )

    out = Path(args.out or REPO / "data/eval/rlnews/hmm_causal.json")
    out.write_text(
        json.dumps(
            {
                "confirm_window": WIN["track_c_daily"],
                "cost_bp_last_day": cost_bp(rows[-1]["day"]),
                "n_configs": len(configs),
                "search_results": results,
                "best": best,
                "null_best_of_n": {
                    "iid": {
                        "mean": float(null_iid.mean()),
                        "p95": float(np.percentile(null_iid, 95)),
                        "max": float(null_iid.max()),
                        "p_ge_observed": float((null_iid >= best["delta_bp"]).mean()),
                    },
                    "block21": {
                        "mean": float(null_blk.mean()),
                        "p95": float(np.percentile(null_blk, 95)),
                        "max": float(null_blk.max()),
                        "p_ge_observed": float((null_blk >= best["delta_bp"]).mean()),
                    },
                    "draws": args.shuffles,
                },
                "ordering_years": {
                    "overnight_ok": ord_ok_n,
                    "intraday_ok": ord_ok_i,
                    "n_years": ny,
                },
                "agreement": {"filtered_vs_smoothed": agree, "filtered_vs_fullfit": agree2},
                "judge_search": j,
                "judge_confirm": jc,
                "confirm_state_days": dc,
                "confirm_state_visits": vc,
            },
            indent=1,
            default=str,
        )
        + "\n"
    )
    print("\nwrote", out)


if __name__ == "__main__":
    main()
