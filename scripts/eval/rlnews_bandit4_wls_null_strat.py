"""Second null for the WLS run: permute reward within (day x sigma quintile).

The plain within-day permutation destroys the cross-sectional sigma->dispersion
relation, so `beta` re-estimated inside a shuffled run collapses to ~0.05 and
the weighting is effectively inert: that null therefore cannot tell us whether
the WEIGHTED search manufactures selection on noise.  Permuting inside
(day, sigma quintile) strata keeps the heteroscedasticity (and so keeps beta
near its real value) while still destroying any feature->reward link.

    python data/eval/rlnews/bandit4_wls_null_strat.py prior [n_null]
"""

from __future__ import annotations

import collections
import json
import sys

import numpy as np

sys.path.insert(0, "scripts/eval")
sys.path.insert(0, "data/eval/rlnews")
from bandit3_run import OUT, VARIANTS, stat  # noqa: E402
from bandit4_wls_run import _sigma, run_variant_w  # noqa: E402


def shuffle_within_day_sigma(rows, sig_q, rng):
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by[(r["day"], sig_q[i])].append(i)
    new = [dict(r) for r in rows]
    for idxs in by.values():
        vals = [rows[i]["reward"] for i in idxs]
        rng.shuffle(vals)
        for i, v in zip(idxs, vals, strict=True):
            new[i]["reward"] = v
    return new


def main() -> None:
    period = sys.argv[1]
    n_null = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    win = json.loads((OUT / "CONFIRM_WINDOWS.json").read_text())[f"track_a_{period}"]
    search_hi = min(win["confirm"])
    rows = [json.loads(line) for line in (OUT / f"feats3_{period}.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["has_mkt"]]
    search = [r for r in rows if r["day"] < search_hi]

    sig = _sigma(search)
    fin = sig[np.isfinite(sig) & (sig > 0)]
    edges = np.quantile(fin, [0.2, 0.4, 0.6, 0.8])
    sig_q = [int(np.searchsorted(edges, s)) if np.isfinite(s) and s > 0 else -1 for s in sig]

    rng = np.random.default_rng(20260922)
    null_best, detail = [], []
    for i in range(n_null):
        sh = shuffle_within_day_sigma(search, sig_q, rng)
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
        detail.append(per)
        print(
            f"strat null {i}: best={b['stat']:.2f} betas={per['V1_norm10_daily_size']['betas']}",
            file=sys.stderr,
        )

    out = {
        "N": n_null,
        "shuffle": "reward permuted within (day x sigma quintile); beta re-estimated inside each run",
        "null_best_of_search": null_best,
        "null_best_max": float(max(null_best)),
        "null_best_p90": float(np.percentile(null_best, 90)),
        "null_best_median": float(np.median(null_best)),
        "detail": detail,
    }
    (OUT / f"bandit4_{period}_wls_null_strat.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "detail"}, indent=1))


if __name__ == "__main__":
    main()
