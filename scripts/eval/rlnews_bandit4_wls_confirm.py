"""ONE-SHOT confirm-slice evaluation, WLS-weighted fit.

    python data/eval/rlnews/bandit4_wls_confirm.py prior <variant>

Identical to bandit3_confirm.py except the fit is weighted: beta is estimated
on the training rows (days <= 2026-01-27) only, weights w ∝ sigma^(-2 beta),
mean 1.  Evaluation is on actual net bp.

NOTE: the confirm slice was already spent once (V1, unweighted).  This is its
second use and it is correspondingly weaker evidence.
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "scripts/eval")
sys.path.insert(0, "data/eval/rlnews")
from bandit3_run import OUT, VARIANTS, _mp, boot_se, miss_cols_for  # noqa: E402
from bandit4_wls_run import choose_w, dispersion, est_beta, fit_rows_w  # noqa: E402


def main() -> None:
    period, name = sys.argv[1], sys.argv[2]
    feats = VARIANTS[name]
    win = json.loads((OUT / "CONFIRM_WINDOWS.json").read_text())[f"track_a_{period}"]
    confirm_days = set(win["confirm"])
    search_hi = min(confirm_days)

    rows = [json.loads(line) for line in (OUT / f"feats3_{period}.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["has_mkt"]]
    train = [r for r in rows if r["day"] < search_hi]
    ev = [r for r in rows if r["day"] in confirm_days]

    miss = miss_cols_for(train, feats)
    beta, bdiag = est_beta(train)
    hp = choose_w(train, feats, miss, True)
    model = fit_rows_w(train, feats, miss, hp["lam"], hp["min_prob"], hp["min_edge"], beta=beta)
    m, p = _mp(model, ev, feats, miss)
    mask = (p >= model.min_prob) & (m >= model.min_edge)
    sel = [r for r, on in zip(ev, mask, strict=True) if on]
    rem = [r for r, on in zip(ev, mask, strict=True) if not on]

    def agg(xs):
        if not xs:
            return None
        return {
            "n": len(xs),
            "net_bp": 1e4 * float(np.mean([r["reward"] for r in xs])),
            "gross_bp": 1e4 * float(np.mean([r["ret_close"] for r in xs])),
            "cost_bp": 1e4 * float(np.mean([r["cost"] for r in xs])),
        }

    mtr, _ = _mp(model, train, feats, miss)
    ytr = np.array([r["reward"] for r in train], dtype=float)
    from bandit4_wls_run import weights_for

    wtr, _ = weights_for(train, beta)
    out = {
        "period": period,
        "variant": name,
        "weighted": True,
        "beta_train": beta,
        "beta_diag": bdiag,
        "dispersion_train": dispersion(train, beta, resid=np.sqrt(wtr) * (ytr - mtr)),
        "hp": hp,
        "train_rows": len(train),
        "train_days": len({r["day"] for r in train}),
        "confirm_window": [win["confirm"][0], win["confirm"][-1]],
        "confirm_rows": len(ev),
        "confirm_days": len({r["day"] for r in ev}),
        "n_selected": len(sel),
        "sel_rate": len(sel) / max(1, len(ev)),
        "selected": agg(sel),
        "remainder": agg(rem),
        "all": agg(ev),
        "diff_bp": (
            1e4
            * (
                float(np.mean([r["reward"] for r in sel]))
                - float(np.mean([r["reward"] for r in rem]))
            )
        )
        if sel and rem
        else None,
        "diff_bp_se_day_clustered": boot_se(sel, rem) if sel and rem else None,
        "pass_bar": "per-trade net reward > 0 (gross clears the round-trip toll)",
        "passed": bool(sel) and float(np.mean([r["reward"] for r in sel])) > 0,
        "days_with_trades": len({r["day"] for r in sel}),
        "slice_uses": "SECOND use of the frozen confirm slice (first: V1 unweighted)",
    }
    (OUT / f"bandit4_{period}_wls_confirm_{name}.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False)
    )
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
