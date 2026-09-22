"""rlnews entry bandit, v3 feature sets, WLS-weighted target.

    python data/eval/rlnews/bandit4_wls_run.py prior [n_null]

Same folds, same grid, same variants, same reward as bandit3_run.py.  The only
change: `bandit.fit` is given sample weights w ∝ sigma^(-2*beta), with `beta`
estimated ON THE TRAINING ROWS OF THAT FIT ONLY (log-log regression of
per-decile reward sd on decile median sigma).  Weights are normalized to mean 1
inside each fit so the ridge prior's effective strength is comparable.

Evaluation is always on actual net bp: weights change how the model is fitted,
never what a trade is worth.
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "scripts/eval")
sys.path.insert(0, "data/eval/rlnews")
from bandit3_run import (  # noqa: E402
    OUT,
    VARIANTS,
    _mp,
    boot_se,
    build_matrix,
    miss_cols_for,
    shuffle_within_day,
    stat,
)
from bandit_lodo import GRID, LAMS, objective  # noqa: E402

from daytrade_it.application.trading import bandit  # noqa: E402

BETA_LO, BETA_HI = 0.0, 2.0
N_BINS = 10


def _sigma(rows):
    return np.array([r["features"].get("sigma", np.nan) for r in rows], dtype=float)


def est_beta(rows, n_bins=N_BINS):
    """sd(reward) ∝ sigma^beta, by log-log regression on `rows` (training only)."""
    sig = _sigma(rows)
    rew = np.array([r["reward"] for r in rows], dtype=float)
    ok = np.isfinite(sig) & np.isfinite(rew) & (sig > 0)
    sig, rew = sig[ok], rew[ok]
    if sig.size < 200:
        return 0.0, {"raw": None, "reason": "too few usable rows", "n": int(sig.size)}
    edges = np.quantile(sig, np.linspace(0, 1, n_bins + 1))
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        m = (sig >= edges[i]) & (sig <= edges[i + 1] if i == n_bins - 1 else sig < edges[i + 1])
        if m.sum() < 30:
            continue
        sd = float(rew[m].std())
        if sd <= 0:
            continue
        xs.append(np.log(float(np.median(sig[m]))))
        ys.append(np.log(sd))
        ns.append(int(m.sum()))
    if len(xs) < 4:
        return 0.0, {"raw": None, "reason": "too few usable deciles", "bins": len(xs)}
    x, y = np.array(xs), np.array(ys)
    beta_raw = float(np.polyfit(x, y, 1)[0])
    r = float(np.corrcoef(x, y)[0, 1])
    beta = float(min(max(beta_raw, BETA_LO), BETA_HI))
    return beta, {
        "raw": beta_raw,
        "clamped": beta != beta_raw,
        "r_loglog": r,
        "n_bins": len(xs),
        "n_rows": int(sig.size),
    }


def weights_for(rows, beta):
    """w ∝ sigma^(-2 beta), mean 1.  Non-finite / non-positive sigma -> median sigma."""
    sig = _sigma(rows)
    bad = ~np.isfinite(sig) | (sig <= 0)
    if bad.all():
        return np.ones(len(rows)), int(bad.sum())
    sig = np.where(bad, np.median(sig[~bad]), sig)
    w = sig ** (-2.0 * beta)
    w = w / w.mean()
    return w, int(bad.sum())


def fit_rows_w(rows, feats, miss, lam, min_prob=0.5, min_edge=0.0, beta=None):
    X, names = build_matrix(rows, feats, miss)
    y = np.array([r["reward"] for r in rows], dtype=float)
    w = None
    if beta is not None:
        w, _ = weights_for(rows, beta)
    return bandit.fit(
        X, y, lam=lam, min_prob=min_prob, min_edge=min_edge, feature_names=names, weights=w
    )


def dispersion(rows, beta, n_q=5, resid=None):
    """sd(reward) and sd(sqrt(w)*reward) across sigma quintiles; ratio max/min and Q5/Q1."""
    sig = _sigma(rows)
    rew = np.array([r["reward"] for r in rows], dtype=float)
    w, _ = weights_for(rows, beta)
    ok = np.isfinite(sig) & np.isfinite(rew) & (sig > 0)
    sig_o, rew_o, w_o = sig[ok], rew[ok], w[ok]
    res_o = None if resid is None else np.asarray(resid, dtype=float)[ok]
    q = np.quantile(sig_o, np.linspace(0, 1, n_q + 1))
    raw, wt, rs = [], [], []
    for i in range(n_q):
        m = (sig_o >= q[i]) & (sig_o <= q[i + 1] if i == n_q - 1 else sig_o < q[i + 1])
        raw.append(float(rew_o[m].std()) * 1e4)
        wt.append(float((np.sqrt(w_o[m]) * rew_o[m]).std()) * 1e4)
        if res_o is not None:
            rs.append(float(res_o[m].std()) * 1e4)
    out = {
        "sd_raw_bp": raw,
        "sd_weighted_bp": wt,
        "ratio_raw_q5_q1": raw[-1] / raw[0],
        "ratio_weighted_q5_q1": wt[-1] / wt[0],
        "ratio_raw_max_min": max(raw) / min(raw),
        "ratio_weighted_max_min": max(wt) / min(wt),
    }
    if rs:
        out["sd_wresid_bp"] = rs
        out["ratio_wresid_q5_q1"] = rs[-1] / rs[0]
        out["ratio_wresid_max_min"] = max(rs) / min(rs)
    return out


def choose_w(rows, feats, miss, weighted):
    months = sorted({r["month"] for r in rows})
    if len(months) < 2:
        return {"lam": LAMS[0], "min_prob": 0.7, "min_edge": 0.005, "inner": "insufficient"}
    val_month = months[-1]
    inner_train = [r for r in rows if r["month"] < val_month]
    inner_val = [r for r in rows if r["month"] == val_month]
    beta_in = None
    beta_diag = None
    if weighted:
        beta_in, beta_diag = est_beta(inner_train)
    val_days = sorted({r["day"] for r in inner_val})
    val_rewards = np.array([r["reward"] for r in inner_val])
    val_day = np.array([r["day"] for r in inner_val])
    best, best_obj = None, -np.inf
    for lam in LAMS:
        model = fit_rows_w(inner_train, feats, miss, lam, beta=beta_in)
        m, p = _mp(model, inner_val, feats, miss)
        for p_min in sorted({g[1] for g in GRID}, reverse=True):
            for e_min in sorted({g[2] for g in GRID}, reverse=True):
                mask = (p >= p_min) & (m >= e_min)
                sums = dict.fromkeys(val_days, 0.0)
                for d, v in zip(val_day[mask], val_rewards[mask], strict=True):
                    sums[d] += float(v)
                obj = objective([sums[d] for d in val_days])
                if obj > best_obj + 1e-12:
                    best_obj, best = obj, (lam, p_min, e_min)
    lam, p_min, e_min = best
    return {
        "lam": lam,
        "min_prob": p_min,
        "min_edge": e_min,
        "inner_objective": best_obj,
        "val_month": val_month,
        "inner_beta": beta_in,
        "inner_beta_diag": beta_diag,
    }


def run_variant_w(rows, feats, *, weighted=True, keep_w=True, keep_disp=True):
    miss = miss_cols_for(rows, feats)
    months = sorted({r["month"] for r in rows})
    folds, sel_all, rem_all = [], [], []
    for m in months[1:]:
        train = [r for r in rows if r["month"] < m]
        ev = [r for r in rows if r["month"] == m]
        if len({r["day"] for r in train}) < 20:
            continue
        beta, bdiag = est_beta(train) if weighted else (None, None)
        hp = choose_w(train, feats, miss, weighted)
        model = fit_rows_w(train, feats, miss, hp["lam"], hp["min_prob"], hp["min_edge"], beta=beta)
        mev, pev = _mp(model, ev, feats, miss)
        mask = (pev >= model.min_prob) & (mev >= model.min_edge)
        sel = [r for r, on in zip(ev, mask, strict=True) if on]
        rem = [r for r, on in zip(ev, mask, strict=True) if not on]
        sel_all += sel
        rem_all += rem
        f = {
            "month": m,
            "n_train_rows": len(train),
            "n_train_days": len({r["day"] for r in train}),
            "beta": beta,
            "beta_diag": bdiag,
            "inner_beta": hp.get("inner_beta"),
            "lam": hp["lam"],
            "min_prob": hp["min_prob"],
            "min_edge": hp["min_edge"],
            "n_eval_pairs": len(ev),
            "n_selected": len(sel),
            "sel_rate": len(sel) / len(ev),
            "pt_sel_bp": 1e4 * float(np.mean([r["reward"] for r in sel])) if sel else None,
            "pt_rem_bp": 1e4 * float(np.mean([r["reward"] for r in rem])) if rem else None,
            "pt_all_bp": 1e4 * float(np.mean([r["reward"] for r in ev])),
        }
        if keep_disp and weighted:
            mtr, _ = _mp(model, train, feats, miss)
            ytr = np.array([r["reward"] for r in train], dtype=float)
            wtr, _ = weights_for(train, beta)
            f["dispersion_train"] = dispersion(train, beta, resid=np.sqrt(wtr) * (ytr - mtr))
        if keep_w:
            f["w"] = {
                n: float(v)
                for n, v in zip(("_intercept", *model.feature_names), model.w, strict=True)
            }
        folds.append(f)
    res = {
        "folds": folds,
        "n_folds": len(folds),
        "betas": [f["beta"] for f in folds],
        "lams": [f["lam"] for f in folds],
        "lam_at_grid_max_every_fold": bool(folds) and all(f["lam"] == max(LAMS) for f in folds),
        "n_zero_trade_folds": sum(1 for f in folds if f["n_selected"] == 0),
        "n_selected": len(sel_all),
        "n_available": len(sel_all) + len(rem_all),
        "pt_sel_bp": 1e4 * float(np.mean([r["reward"] for r in sel_all])) if sel_all else None,
        "pt_rem_bp": 1e4 * float(np.mean([r["reward"] for r in rem_all])) if rem_all else None,
        "diff_bp": (
            1e4
            * (
                float(np.mean([r["reward"] for r in sel_all]))
                - float(np.mean([r["reward"] for r in rem_all]))
            )
        )
        if sel_all and rem_all
        else None,
    }
    return res, sel_all, rem_all


def main() -> None:
    period = sys.argv[1]
    n_null = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    win = json.loads((OUT / "CONFIRM_WINDOWS.json").read_text())[f"track_a_{period}"]
    confirm_days = set(win["confirm"])
    search_hi = min(confirm_days)

    rows = [json.loads(line) for line in (OUT / f"feats3_{period}.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["has_mkt"]]
    search = [r for r in rows if r["day"] < search_hi]
    assert not ({r["day"] for r in search} & confirm_days)

    beta_pool, pool_diag = est_beta(search)
    report = {
        "period": period,
        "weighting": "w ∝ sigma^(-2 beta), mean 1, beta from training rows of each fit",
        "beta_clamp": [BETA_LO, BETA_HI],
        "search_range": [min(r["day"] for r in search), max(r["day"] for r in search)],
        "n_search_rows": len(search),
        "n_search_days": len({r["day"] for r in search}),
        "pooled_beta_reference_only": {"beta": beta_pool, "diag": pool_diag},
        "pooled_dispersion_reference_only": dispersion(search, beta_pool),
        "variants": {},
    }

    for name, feats in VARIANTS.items():
        res, sel, rem = run_variant_w(search, feats, weighted=True)
        if sel and rem:
            res["diff_bp_se_day_clustered"] = boot_se(sel, rem)
        report["variants"][name] = res
        print(
            f"{name}: sel {res['n_selected']}/{res['n_available']} "
            f"pt_sel={res['pt_sel_bp']} pt_rem={res['pt_rem_bp']} "
            f"lams={res['lams']} betas={res['betas']} zero={res['n_zero_trade_folds']}",
            file=sys.stderr,
        )

    rng = np.random.default_rng(20260922)
    null_best, null_detail = [], []
    for i in range(n_null):
        sh = shuffle_within_day(search, rng)
        per = {}
        for name, feats in VARIANTS.items():
            r, _, _ = run_variant_w(sh, feats, weighted=True, keep_w=False, keep_disp=False)
            per[name] = {
                "stat": stat(r),
                "n_selected": r["n_selected"],
                "pt_sel_bp": r["pt_sel_bp"],
                "lams": r["lams"],
                "betas": r["betas"],
                "n_zero_trade_folds": r["n_zero_trade_folds"],
            }
        b = max(per.values(), key=lambda x: x["stat"])
        null_best.append(b["stat"])
        null_detail.append(per)
        print(f"null {i}: best_stat={b['stat']:.2f}", file=sys.stderr)
    report["selection_premium"] = {
        "N": n_null,
        "shuffle": "reward permuted within each day; beta re-estimated inside every shuffled fit",
        "searched_over": list(VARIANTS),
        "statistic": "pooled per-trade diff bp (selected - remainder), 0 when nothing selected",
        "null_best_of_search": null_best,
        "null_best_max": float(max(null_best)) if null_best else None,
        "null_best_p90": float(np.percentile(null_best, 90)) if null_best else None,
        "null_best_median": float(np.median(null_best)) if null_best else None,
        "observed_best": {n: stat(report["variants"][n]) for n in VARIANTS},
        "detail": null_detail,
    }

    (OUT / f"bandit4_{period}_wls_search.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False)
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "selection_premium"}, indent=1)[:3000]
    )


if __name__ == "__main__":
    main()
