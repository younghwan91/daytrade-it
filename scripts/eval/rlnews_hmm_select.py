"""Step 1: choose K and the observation set on TIME-ORDERED out-of-sample
one-step predictive log-likelihood, inside the SEARCH span only.

The confirm span (DAILY_CONFIRM_WINDOW.json, frozen before any of this ran) is
never loaded here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/young/git/daytrade-it-rlnews")
sys.path.insert(0, str(Path(__file__).parent))
from hmm_core import filter_states, fit_em, n_params, relabel, visit_counts  # noqa: E402

SERIES = REPO / "data/eval/rlnews/hmm_daily_series.jsonl"
WIN = json.loads((REPO / "data/eval/rlnews/DAILY_CONFIRM_WINDOW.json").read_text())

OBS_SETS = {
    "legs": ["o_night", "o_intra"],
    "vol": ["o_rv5", "o_rv21", "o_disp"],
    "compact": ["o_intra", "o_disp", "o_rv21", "o_breadth"],
    "legs5vol": ["o_night5", "o_intra5", "o_rv21", "o_disp"],
    "full": [
        "o_night",
        "o_intra",
        "o_disp",
        "o_breadth",
        "o_rv5",
        "o_rv21",
        "o_logtv",
        "o_dlogtv",
    ],
}
VOL_KEY = {
    "legs": "o_intra",
    "vol": "o_rv21",
    "compact": "o_rv21",
    "legs5vol": "o_rv21",
    "full": "o_rv21",
}


def load():
    rows = [json.loads(line) for line in SERIES.read_text().splitlines() if line.strip()]
    lo, hi = WIN["track_c_daily"]["search"]
    search = [r for r in rows if lo <= r["day"] <= hi]
    return rows, search


def design(rows, cols):
    return np.array([[r[c] for c in cols] for r in rows], float)


def main():
    rows, search = load()
    print(f"search span {search[0]['day']}..{search[-1]['day']}  n={len(search)}")
    cut = int(len(search) * 0.70)
    tr, te = search[:cut], search[cut:]
    print(
        f"inner fit {tr[0]['day']}..{tr[-1]['day']} n={len(tr)} | "
        f"inner oos {te[0]['day']}..{te[-1]['day']} n={len(te)}\n"
    )

    res = []
    for name, cols in OBS_SETS.items():
        Xtr = design(tr, cols)
        mu_, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-12  # fit-window stats only
        Xtr = (Xtr - mu_) / sd_
        Xall = (design(search, cols) - mu_) / sd_
        by = cols.index(VOL_KEY[name])
        for K in (2, 3, 4):
            p = fit_em(Xtr, K, seed=7)
            p = relabel(p, by=by)
            # one-step predictive LL on the held-out tail, params fit on the head
            _, _, pll = filter_states(p, Xall)
            oos = float(pll[cut:].mean())
            ins = float(pll[:cut].mean())
            bic = -2 * p["loglik"] + n_params(K, len(cols)) * np.log(len(tr))
            a_tr, _, _ = filter_states(p, Xtr)
            days, visits = visit_counts(a_tr.argmax(1))
            res.append(
                {
                    "obs": name,
                    "K": K,
                    "oos_pll": oos,
                    "ins_pll": ins,
                    "bic": float(bic),
                    "days": days,
                    "visits": visits,
                    "npar": n_params(K, len(cols)),
                }
            )
            print(
                f"{name:<9} K={K}  oosPLL/day {oos:+8.4f}  insPLL/day {ins:+8.4f}  "
                f"BIC {bic:10.1f}  npar {n_params(K, len(cols)):4d}  "
                f"days {days}  visits {visits}"
            )
    best = max(res, key=lambda r: r["oos_pll"])
    print(
        f"\nBEST by time-ordered OOS predictive LL: obs={best['obs']} K={best['K']} "
        f"({best['oos_pll']:+.4f}/day)"
    )
    out = REPO / "data/eval/rlnews/hmm_select.json"
    out.write_text(
        json.dumps(
            {
                "inner_fit": [tr[0]["day"], tr[-1]["day"]],
                "inner_oos": [te[0]["day"], te[-1]["day"]],
                "results": res,
                "best": best,
            },
            indent=1,
        )
        + "\n"
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
