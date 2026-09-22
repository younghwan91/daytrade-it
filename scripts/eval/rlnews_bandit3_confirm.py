"""ONE-SHOT confirm-slice evaluation for the v3 feature sets.

    python data/eval/rlnews/bandit3_confirm.py prior <variant>

Trains on every search-range day (<= 2026-01-27) with hyperparameters chosen by
the same inner time-ordered split, then evaluates the frozen confirm slice
(2026-01-28..2026-03-16) exactly once.  The pass bar was fixed in advance:
per-trade NET reward > 0, i.e. gross hold-to-close return clears the ~36.7 bp
round-trip toll.
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "data/eval/rlnews")
from bandit3_run import (  # noqa: E402
    OUT,
    VARIANTS,
    _mp,
    boot_se,
    choose,
    fit_rows,
    miss_cols_for,
)


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
    hp = choose(train, feats, miss)
    model = fit_rows(train, feats, miss, hp["lam"], hp["min_prob"], hp["min_edge"])
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

    out = {
        "period": period,
        "variant": name,
        "features": list(feats),
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
    }
    (OUT / f"bandit3_{period}_confirm_{name}.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False)
    )
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
