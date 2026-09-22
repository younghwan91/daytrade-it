"""rlnews entry bandit, WLS search, feats5 MA encoding (bucketed ma_frac).

    python data/eval/rlnews/bandit5_wls_run.py prior new [n_null] [out_dir]
    python data/eval/rlnews/bandit5_wls_run.py prior old 0 /tmp/x   # reproduce bandit4

Identical to bandit4_wls_run.py + bandit4_wls_null_strat.py in every respect
(folds, grid, inner time-ordered split, beta per fit on training rows only,
reward, evaluation on actual net bp, stratified null with the same RNG seed and
therefore the SAME permutations) except the MA columns:

    old: ma_count_*  + ma_dist_norm_* + ma_n_avail_*   (feats3_prior.jsonl)
    new: ma_frac_*   + ma_dist_norm_* + ma_n_avail_*   (feats5_prior.jsonl)

SEARCH WINDOW ONLY.  Rows on or after the first track_a confirm day are dropped
before anything else happens; the confirm slice is never read into a model or
an evaluation.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, "scripts/eval")
sys.path.insert(0, "data/eval/rlnews")
from bandit3_run import BASE10_NORM, BASE10_RAW, OUT, SIZE, boot_se, stat  # noqa: E402
from bandit4_wls_null_strat import shuffle_within_day_sigma  # noqa: E402
from bandit4_wls_run import _sigma, dispersion, est_beta, run_variant_w  # noqa: E402

ENC = {
    "old": (
        "feats3",
        ("ma_count_daily", "ma_dist_norm_daily", "ma_n_avail_daily"),
        ("ma_count_intraday", "ma_dist_norm_intraday", "ma_n_avail_intraday"),
    ),
    "new": (
        "feats5",
        ("ma_frac_daily", "ma_dist_norm_daily", "ma_n_avail_daily"),
        ("ma_frac_intraday", "ma_dist_norm_intraday", "ma_n_avail_intraday"),
    ),
}

_G: dict = {}


def variants(ma_d, ma_i):
    return {
        "V0_old_raw10": BASE10_RAW,
        "V1_norm10_daily_size": BASE10_NORM + ma_d + SIZE,
        "V2_norm10_intraday_size": BASE10_NORM + ma_i + SIZE,
        "V3_norm10_both_size": BASE10_NORM + ma_d + ma_i + SIZE,
    }


def stat_within(res):
    """Selection-weighted mean of WITHIN-FOLD (pt_sel - pt_rem), bp; 0 if nothing selected.

    Secondary statistic.  The pre-registered `stat` pools selected vs remainder
    across folds, so it mixes month-level means into the comparison: selecting
    anything in a good month scores well, anything in a bad month scores badly.
    This one compares each fold's selected rows only against that fold's own
    remainder.
    """
    num, den = 0.0, 0
    for f in res["folds"]:
        if f["n_selected"] and f["pt_rem_bp"] is not None:
            num += f["n_selected"] * (f["pt_sel_bp"] - f["pt_rem_bp"])
            den += f["n_selected"]
    return num / den if den else 0.0


def _null_one(i):
    search, rewards, VAR = _G["search"], _G["rewards"][i], _G["VAR"]
    sh = [dict(r) for r in search]
    for r, v in zip(sh, rewards, strict=True):
        r["reward"] = v
    per = {}
    for name, feats in VAR.items():
        r, _, _ = run_variant_w(sh, feats, weighted=True, keep_w=True, keep_disp=False)
        per[name] = {
            "stat": stat(r),
            "stat_within": stat_within(r),
            "n_selected": r["n_selected"],
            "pt_sel_bp": r["pt_sel_bp"],
            "lams": r["lams"],
            "betas": r["betas"],
            "n_zero_trade_folds": r["n_zero_trade_folds"],
            "fold_n_selected": [f["n_selected"] for f in r["folds"]],
            "fold_pt_sel_bp": [f["pt_sel_bp"] for f in r["folds"]],
            "fold_pt_rem_bp": [f["pt_rem_bp"] for f in r["folds"]],
        }
    return i, per


def main() -> None:
    period, enc = sys.argv[1], sys.argv[2]
    n_null = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    out_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else OUT
    tag, ma_d, ma_i = ENC[enc]
    VAR = variants(ma_d, ma_i)

    win = json.loads((OUT / "CONFIRM_WINDOWS.json").read_text())[f"track_a_{period}"]
    confirm_days = set(win["confirm"])
    search_hi = min(confirm_days)
    search = []
    with (OUT / f"{tag}_{period}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["day"] >= search_hi:  # confirm slice never enters memory
                continue
            if r["has_mkt"]:
                search.append(r)
    assert not ({r["day"] for r in search} & confirm_days)
    assert max(r["day"] for r in search) < search_hi

    beta_pool, pool_diag = est_beta(search)
    report = {
        "period": period,
        "encoding": enc,
        "feature_file": str(OUT / f"{tag}_{period}.jsonl"),
        "ma_daily": list(ma_d),
        "ma_intraday": list(ma_i),
        "confirm_slice_touched": False,
        "confirm_first_day_excluded_from": search_hi,
        "weighting": "w ∝ sigma^(-2 beta), mean 1, beta from training rows of each fit",
        "search_range": [min(r["day"] for r in search), max(r["day"] for r in search)],
        "n_search_rows": len(search),
        "n_search_days": len({r["day"] for r in search}),
        "pooled_beta_reference_only": {"beta": beta_pool, "diag": pool_diag},
        "pooled_dispersion_reference_only": dispersion(search, beta_pool),
        "variants": {},
    }
    for name, feats in VAR.items():
        res, sel, rem = run_variant_w(search, feats, weighted=True)
        if sel and rem:
            res["diff_bp_se_day_clustered"] = boot_se(sel, rem)
        res["stat_within_fold_bp"] = stat_within(res)
        report["variants"][name] = res
        print(
            f"{enc} {name}: sel {res['n_selected']}/{res['n_available']} "
            f"pt_sel={res['pt_sel_bp']} pt_rem={res['pt_rem_bp']} lams={res['lams']} "
            f"sel_by_fold={[f['n_selected'] for f in res['folds']]}",
            file=sys.stderr,
        )
    (out_dir / f"bandit5_{period}_wls_search_{enc}.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False)
    )
    if n_null <= 0:
        return

    # ---- stratified null: identical permutations to bandit4_wls_null_strat ----
    sig = _sigma(search)
    fin = sig[np.isfinite(sig) & (sig > 0)]
    edges = np.quantile(fin, [0.2, 0.4, 0.6, 0.8])
    sig_q = [int(np.searchsorted(edges, s)) if np.isfinite(s) and s > 0 else -1 for s in sig]
    rng = np.random.default_rng(20260922)
    rewards = []
    for _ in range(n_null):
        sh = shuffle_within_day_sigma(search, sig_q, rng)
        rewards.append([r["reward"] for r in sh])
    _G.update(search=search, rewards=rewards, VAR=VAR)
    with mp.get_context("fork").Pool(min(10, n_null)) as pool:
        got = dict(pool.imap_unordered(_null_one, range(n_null)))
    detail = [got[i] for i in range(n_null)]
    null_best = [max(p["stat"] for p in per.values()) for per in detail]
    obs = {n: stat(report["variants"][n]) for n in VAR}
    obs_best = max(obs.values())
    null_best_w = [max(p["stat_within"] for p in per.values()) for per in detail]
    obs_w = {n: stat_within(report["variants"][n]) for n in VAR}
    obs_w_best = max(obs_w.values())
    out = {
        "N": n_null,
        "encoding": enc,
        "shuffle": "reward permuted within (day x sigma quintile); beta re-estimated inside each run; "
        "same RNG seed as bandit4_wls_null_strat.py -> same permutations",
        "confirm_slice_touched": False,
        "statistic": "pooled per-trade diff bp (selected - remainder), 0 when nothing selected; "
        "max over the 4 variants",
        "observed_by_variant": obs,
        "observed_best": obs_best,
        "null_best_of_search": null_best,
        "null_best_max": float(max(null_best)),
        "null_best_p90": float(np.percentile(null_best, 90)),
        "null_best_median": float(np.median(null_best)),
        "n_null_ge_observed": int(sum(1 for x in null_best if x >= obs_best)),
        "p_value_plus1": (1 + sum(1 for x in null_best if x >= obs_best)) / (n_null + 1),
        "secondary_within_fold": {
            "statistic": "selection-weighted mean of within-fold (pt_sel - pt_rem) bp, max over variants",
            "observed_by_variant": obs_w,
            "observed_best": obs_w_best,
            "null_best_of_search": null_best_w,
            "null_best_max": float(max(null_best_w)),
            "null_best_p90": float(np.percentile(null_best_w, 90)),
            "null_best_median": float(np.median(null_best_w)),
            "n_null_ge_observed": int(sum(1 for x in null_best_w if x >= obs_w_best)),
            "p_value_plus1": (1 + sum(1 for x in null_best_w if x >= obs_w_best)) / (n_null + 1),
            "per_variant_null": {n: sorted(per[n]["stat_within"] for per in detail) for n in VAR},
        },
        "detail": detail,
    }
    (out_dir / f"bandit5_{period}_wls_null_strat_{enc}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "detail"}, indent=1))


if __name__ == "__main__":
    main()
