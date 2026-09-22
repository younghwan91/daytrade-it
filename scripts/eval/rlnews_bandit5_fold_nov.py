"""Decompose the one unpinned fold (new encoding, V3, 2025-11): day-timing vs
within-day ranking, day-clustered SE.  Search window only.

    python data/eval/rlnews/bandit5_fold_nov.py
"""

from __future__ import annotations

import collections
import json
import sys

import numpy as np

sys.path.insert(0, "scripts/eval")
sys.path.insert(0, "data/eval/rlnews")
from bandit3_run import OUT  # noqa: E402
from bandit4_wls_run import run_variant_w  # noqa: E402
from bandit5_wls_run import ENC, variants  # noqa: E402


def main() -> None:
    win = json.loads((OUT / "CONFIRM_WINDOWS.json").read_text())["track_a_prior"]
    hi = min(win["confirm"])
    tag, ma_d, ma_i = ENC["new"]
    search = []
    with (OUT / f"{tag}_prior.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["day"] < hi and r["has_mkt"]:
                search.append(r)
    feats = variants(ma_d, ma_i)["V3_norm10_both_size"]
    res, sel, rem = run_variant_w(search, feats, weighted=True, keep_w=False, keep_disp=False)
    m = "2025-11"
    s = [r for r in sel if r["month"] == m]
    q = [r for r in rem if r["month"] == m]
    by_s, by_q = collections.defaultdict(list), collections.defaultdict(list)
    for r in s:
        by_s[r["day"]].append(r["reward"])
    for r in q:
        by_q[r["day"]].append(r["reward"])
    days = sorted(set(by_s) | set(by_q))
    per_day = []
    for d in days:
        a, b = by_s.get(d, []), by_q.get(d, [])
        per_day.append(
            {
                "day": d,
                "n_sel": len(a),
                "n_rem": len(b),
                "sel_share": len(a) / (len(a) + len(b)),
                "pt_sel_bp": 1e4 * float(np.mean(a)) if a else None,
                "pt_rem_bp": 1e4 * float(np.mean(b)) if b else None,
                "pt_day_bp": 1e4 * float(np.mean(a + b)),
            }
        )
    # day-timing part: selection-weighted mean of day means minus remainder-weighted
    day_mean = {p["day"]: p["pt_day_bp"] for p in per_day}
    timing = float(
        np.mean([day_mean[r["day"]] for r in s]) - np.mean([day_mean[r["day"]] for r in q])
    )
    # within-day part: day-demeaned rewards
    within = float(
        np.mean([1e4 * r["reward"] - day_mean[r["day"]] for r in s])
        - np.mean([1e4 * r["reward"] - day_mean[r["day"]] for r in q])
    )
    total = 1e4 * float(np.mean([r["reward"] for r in s]) - np.mean([r["reward"] for r in q]))
    rng = np.random.default_rng(7)
    boots, boots_w = [], []
    idx = np.arange(len(days))
    for _ in range(4000):
        pick = rng.choice(idx, size=len(days), replace=True)
        a = [v for i in pick for v in by_s.get(days[i], [])]
        b = [v for i in pick for v in by_q.get(days[i], [])]
        if a and b:
            boots.append(1e4 * (np.mean(a) - np.mean(b)))
        # within-day, day-clustered
        aw = [1e4 * v - day_mean[days[i]] for i in pick for v in by_s.get(days[i], [])]
        bw = [1e4 * v - day_mean[days[i]] for i in pick for v in by_q.get(days[i], [])]
        if aw and bw:
            boots_w.append(np.mean(aw) - np.mean(bw))
    n_days_with_sel = sum(1 for p in per_day if p["n_sel"])
    top = sorted(per_day, key=lambda p: -p["n_sel"])[:5]
    out = {
        "variant": "V3_norm10_both_size",
        "encoding": "new",
        "fold": m,
        "confirm_slice_touched": False,
        "n_sel": len(s),
        "n_rem": len(q),
        "n_days": len(days),
        "n_days_with_selection": n_days_with_sel,
        "diff_total_bp": total,
        "diff_day_timing_bp": timing,
        "diff_within_day_bp": within,
        "se_total_day_clustered_bp": float(np.std(boots)),
        "se_within_day_day_clustered_bp": float(np.std(boots_w)),
        "top5_days_by_n_sel": top,
        "share_of_selections_in_top5_days": sum(p["n_sel"] for p in top) / len(s),
        "per_day": per_day,
    }
    (OUT / "bandit5_prior_wls_fold_2025-11_V3.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_day"}, indent=1))


if __name__ == "__main__":
    main()
