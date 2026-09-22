"""Size-aware null check for the feats5 WLS search (search window only).

Statistic per variant: selected-minus-remainder difference of DAY-DEMEANED
reward (day mean over all candidates that day), pooled over the eval folds,
divided by its day-clustered standard error -> a t.  Max over the 4 variants.
Same 20 (day x sigma quintile) permutations as bandit4/bandit5 strat nulls,
beta re-estimated inside every shuffled fit.

    python data/eval/rlnews/bandit5_null_tstat.py [n_null]
"""

from __future__ import annotations

import collections
import json
import multiprocessing as mp
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, "scripts/eval")
sys.path.insert(0, "data/eval/rlnews")
from bandit3_run import OUT  # noqa: E402
from bandit4_wls_null_strat import shuffle_within_day_sigma  # noqa: E402
from bandit4_wls_run import _sigma, run_variant_w  # noqa: E402
from bandit5_wls_run import ENC, variants  # noqa: E402

_G: dict = {}


def tstat(sel, rem):
    if not sel or not rem:
        return 0.0, 0.0, 0
    allr = sel + rem
    dsum, dn = collections.Counter(), collections.Counter()
    for r in allr:
        dsum[r["day"]] += r["reward"]
        dn[r["day"]] += 1
    dm = {d: dsum[d] / dn[d] for d in dn}
    ns, nr = len(sel), len(rem)
    u = collections.Counter()
    diff = 0.0
    for r in sel:
        v = (r["reward"] - dm[r["day"]]) / ns
        u[r["day"]] += v
        diff += v
    for r in rem:
        v = -(r["reward"] - dm[r["day"]]) / nr
        u[r["day"]] += v
        diff += v
    se = float(np.sqrt(sum(x * x for x in u.values())))
    return (diff / se if se > 0 else 0.0), 1e4 * diff, ns


def _one(i):
    search, VAR = _G["search"], _G["VAR"]
    rows = search
    if i >= 0:
        rows = [dict(r) for r in search]
        for r, v in zip(rows, _G["rewards"][i], strict=True):
            r["reward"] = v
    per = {}
    for name, feats in VAR.items():
        _, sel, rem = run_variant_w(rows, feats, weighted=True, keep_w=False, keep_disp=False)
        t, d, n = tstat(sel, rem)
        per[name] = {"t": t, "diff_daydemeaned_bp": d, "n_selected": n}
    return i, per


def main() -> None:
    n_null = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    win = json.loads((OUT / "CONFIRM_WINDOWS.json").read_text())["track_a_prior"]
    hi = min(win["confirm"])
    tag, ma_d, ma_i = ENC["new"]
    search = []
    with (OUT / f"{tag}_prior.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["day"] < hi and r["has_mkt"]:
                search.append(r)
    assert max(r["day"] for r in search) < hi
    sig = _sigma(search)
    fin = sig[np.isfinite(sig) & (sig > 0)]
    edges = np.quantile(fin, [0.2, 0.4, 0.6, 0.8])
    sig_q = [int(np.searchsorted(edges, s)) if np.isfinite(s) and s > 0 else -1 for s in sig]
    rng = np.random.default_rng(20260922)
    rewards = [
        [r["reward"] for r in shuffle_within_day_sigma(search, sig_q, rng)] for _ in range(n_null)
    ]
    _G.update(search=search, rewards=rewards, VAR=variants(ma_d, ma_i))
    with mp.get_context("fork").Pool(min(11, n_null + 1)) as pool:
        got = dict(pool.imap_unordered(_one, range(-1, n_null)))
    obs = got[-1]
    null = [got[i] for i in range(n_null)]
    obs_best = max(p["t"] for p in obs.values())
    null_best = [max(p["t"] for p in per.values()) for per in null]
    out = {
        "encoding": "new",
        "confirm_slice_touched": False,
        "statistic": "t of selected-minus-remainder day-demeaned reward, day-clustered SE, "
        "pooled over eval folds; max over variants",
        "N": n_null,
        "observed": obs,
        "observed_best_t": obs_best,
        "null_best_t": null_best,
        "null_best_t_sorted": sorted(null_best),
        "null_best_t_max": max(null_best),
        "null_best_t_p90": float(np.percentile(null_best, 90)),
        "null_best_t_median": float(np.median(null_best)),
        "n_null_ge_observed": int(sum(1 for x in null_best if x >= obs_best)),
        "p_value_plus1": (1 + sum(1 for x in null_best if x >= obs_best)) / (n_null + 1),
        "null_per_variant_t_sorted": {n: sorted(per[n]["t"] for per in null) for n in obs},
        "detail": null,
    }
    (OUT / "bandit5_prior_wls_null_strat_tstat.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "detail"}, indent=1))


if __name__ == "__main__":
    main()
