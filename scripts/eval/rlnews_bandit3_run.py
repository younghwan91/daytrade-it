"""rlnews entry bandit, v3 feature sets: MA position + size, per-stock normalized.

    python data/eval/rlnews/bandit3_run.py prior

Honors the frozen confirm slice (data/eval/rlnews/CONFIRM_WINDOWS.json,
track_a_prior): everything is searched on days <= 2026-01-27; the confirm slice
is evaluated ONCE at the end.  Monthly time-ordered folds; hyperparameters from
an inner time-ordered split of the training months.  No LODO.
Reward = ret_close - round_trip(day, market, at_open=act_min<5).
Lab-level hold-to-close study, not an engine replay.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts/eval")
from bandit_lodo import GRID, LAMS, objective  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

from daytrade_it.application.trading import bandit  # noqa: E402

OUT = Path("/home/young/git/daytrade-it-rlnews/data/eval/rlnews")
NOTIONAL = 100_000.0

BASE10_RAW = (
    "act_min",
    "premarket",
    "sigma",
    "sig_q",
    "val_q",
    "log_prev_value",
    "ret_5m",
    "mkt_move",
    "n_articles_today_before",
    "minutes_since_prev_article",
)
BASE10_NORM = (
    "act_min",
    "premarket",
    "sigma_rel",
    "sig_q",
    "val_q",
    "value_rel",
    "ret_5m_norm",
    "mkt_move",
    "news_rate_rel",
    "gap_rel",
)
MA_D = ("ma_count_daily", "ma_dist_norm_daily", "ma_n_avail_daily")
MA_I = ("ma_count_intraday", "ma_dist_norm_intraday", "ma_n_avail_intraday")
SIZE = ("log_mcap", "size_q", "is_kosdaq")

VARIANTS: dict[str, tuple[str, ...]] = {
    "V0_old_raw10": BASE10_RAW,
    "V1_norm10_daily_size": BASE10_NORM + MA_D + SIZE,
    "V2_norm10_intraday_size": BASE10_NORM + MA_I + SIZE,
    "V3_norm10_both_size": BASE10_NORM + MA_D + MA_I + SIZE,
}


def design(rows, feats):
    """Feature matrix with an explicit missing-indicator per column that has
    any NaN; NaN -> 0 AFTER the indicator, so nothing is imputed silently."""
    X = np.array([[r["features"][f] for f in feats] for r in rows], dtype=float)
    return X


def build_matrix(rows, feats, miss_cols):
    X = design(rows, feats)
    bad = ~np.isfinite(X)
    parts = [np.where(bad, 0.0, X)]
    names = list(feats)
    for j in miss_cols:
        parts.append(bad[:, [j]].astype(float))
        names.append(feats[j] + "__miss")
    return np.hstack(parts), tuple(names)


def miss_cols_for(rows, feats):
    X = design(rows, feats)
    bad = ~np.isfinite(X)
    return tuple(j for j in range(X.shape[1]) if bad[:, j].any())


def fit_rows(rows, feats, miss, lam, min_prob=0.5, min_edge=0.0):
    X, names = build_matrix(rows, feats, miss)
    y = np.array([r["reward"] for r in rows], dtype=float)
    return bandit.fit(X, y, lam=lam, min_prob=min_prob, min_edge=min_edge, feature_names=names)


def _mp(model, rows, feats, miss):
    X, _ = build_matrix(rows, feats, miss)
    Z = np.hstack([np.ones((X.shape[0], 1)), (X - model.means) / model.stds])
    m = Z @ model.w
    scale = np.sqrt((model.b / model.a) * (1.0 + np.einsum("ij,jk,ik->i", Z, model.V, Z)))
    return m, student_t.cdf(m / scale, 2.0 * model.a)


def choose(rows, feats, miss):
    months = sorted({r["month"] for r in rows})
    if len(months) < 2:
        return {"lam": LAMS[0], "min_prob": 0.7, "min_edge": 0.005, "inner": "insufficient"}
    val_month = months[-1]
    inner_train = [r for r in rows if r["month"] < val_month]
    inner_val = [r for r in rows if r["month"] == val_month]
    val_days = sorted({r["day"] for r in inner_val})
    val_rewards = np.array([r["reward"] for r in inner_val])
    val_day = np.array([r["day"] for r in inner_val])
    best, best_obj = None, -np.inf
    for lam in LAMS:
        model = fit_rows(inner_train, feats, miss, lam)
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
    }


def run_variant(rows, feats, *, keep_w=True):
    """Monthly time-ordered folds over `rows` (search range only)."""
    miss = miss_cols_for(rows, feats)
    months = sorted({r["month"] for r in rows})
    folds, sel_all, rem_all = [], [], []
    for m in months[1:]:
        train = [r for r in rows if r["month"] < m]
        ev = [r for r in rows if r["month"] == m]
        if len({r["day"] for r in train}) < 20:
            continue
        hp = choose(train, feats, miss)
        model = fit_rows(train, feats, miss, hp["lam"], hp["min_prob"], hp["min_edge"])
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
        if keep_w:
            f["w"] = {
                n: float(v)
                for n, v in zip(("_intercept", *model.feature_names), model.w, strict=True)
            }
        folds.append(f)
    res = {
        "folds": folds,
        "n_folds": len(folds),
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


def stat(res):
    return res["diff_bp"] if res["diff_bp"] is not None else 0.0


def boot_se(sel, rem, n=2000, seed=7):
    days = sorted({r["day"] for r in sel} | {r["day"] for r in rem})
    by_s, by_r = collections.defaultdict(list), collections.defaultdict(list)
    for r in sel:
        by_s[r["day"]].append(r["reward"])
    for r in rem:
        by_r[r["day"]].append(r["reward"])
    rng = np.random.default_rng(seed)
    idx = np.arange(len(days))
    out = []
    for _ in range(n):
        pick = rng.choice(idx, size=len(days), replace=True)
        s = [v for i in pick for v in by_s[days[i]]]
        rr = [v for i in pick for v in by_r[days[i]]]
        if s and rr:
            out.append(np.mean(s) - np.mean(rr))
    return float(np.std(out) * 1e4) if out else float("nan")


def shuffle_within_day(rows, rng):
    by_day = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_day[r["day"]].append(i)
    new = [dict(r) for r in rows]
    for idxs in by_day.values():
        vals = [rows[i]["reward"] for i in idxs]
        rng.shuffle(vals)
        for i, v in zip(idxs, vals, strict=True):
            new[i]["reward"] = v
    return new


def describe(rows, feats):
    X = design(rows, feats)
    out = {}
    for j, f in enumerate(feats):
        col = X[:, j]
        fin = col[np.isfinite(col)]
        vals, cnt = np.unique(fin, return_counts=True) if fin.size else (np.array([]), np.array([]))
        out[f] = {
            "n": int(col.size),
            "frac_missing": float(1.0 - fin.size / col.size),
            "var": float(np.var(fin)) if fin.size else None,
            "sd": float(np.std(fin)) if fin.size else None,
            "mean": float(np.mean(fin)) if fin.size else None,
            "p05": float(np.percentile(fin, 5)) if fin.size else None,
            "median": float(np.median(fin)) if fin.size else None,
            "p95": float(np.percentile(fin, 95)) if fin.size else None,
            "n_distinct": int(vals.size),
            "modal_share": float(cnt.max() / fin.size) if fin.size else None,
        }
        if vals.size <= 10 and fin.size:
            out[f]["dist"] = {
                str(round(float(v), 4)): round(float(c) / fin.size, 4)
                for v, c in zip(vals, cnt, strict=True)
            }
    return out


def main() -> None:
    period = sys.argv[1]
    n_null = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    win = json.loads((OUT / "CONFIRM_WINDOWS.json").read_text())[f"track_a_{period}"]
    confirm_days = set(win["confirm"])
    search_hi = min(confirm_days)

    rows = [json.loads(line) for line in (OUT / f"feats3_{period}.jsonl").open(encoding="utf-8")]
    dropped = sorted({r["day"] for r in rows if not r["has_mkt"]})
    rows = [r for r in rows if r["has_mkt"]]
    search = [r for r in rows if r["day"] < search_hi]
    confirm = [r for r in rows if r["day"] in confirm_days]
    assert not ({r["day"] for r in search} & confirm_days)

    report = {
        "period": period,
        "confirm_window": [win["confirm"][0], win["confirm"][-1]],
        "search_range": [min(r["day"] for r in search), max(r["day"] for r in search)],
        "n_search_rows": len(search),
        "n_search_days": len({r["day"] for r in search}),
        "n_confirm_rows": len(confirm),
        "n_confirm_days": len({r["day"] for r in confirm}),
        "dropped_days_no_mkt": dropped,
        "feature_diagnostics": describe(
            search,
            tuple(sorted(set(BASE10_RAW) | set(BASE10_NORM) | set(MA_D) | set(MA_I) | set(SIZE))),
        ),
        "variants": {},
    }

    sels = {}
    for name, feats in VARIANTS.items():
        res, sel, rem = run_variant(search, feats)
        if sel and rem:
            res["diff_bp_se_day_clustered"] = boot_se(sel, rem)
        report["variants"][name] = res
        sels[name] = (sel, rem)
        print(
            f"{name}: sel {res['n_selected']}/{res['n_available']} "
            f"pt_sel={res['pt_sel_bp']} pt_rem={res['pt_rem_bp']} "
            f"lams={res['lams']} zero_folds={res['n_zero_trade_folds']}",
            file=sys.stderr,
        )

    # --- selection premium: identical search on within-day shuffled rewards
    rng = np.random.default_rng(20260922)
    null_best, null_detail = [], []
    for i in range(n_null):
        sh = shuffle_within_day(search, rng)
        per = {}
        for name, feats in VARIANTS.items():
            r, _, _ = run_variant(sh, feats, keep_w=False)
            per[name] = {
                "stat": stat(r),
                "n_selected": r["n_selected"],
                "pt_sel_bp": r["pt_sel_bp"],
                "lams": r["lams"],
                "n_zero_trade_folds": r["n_zero_trade_folds"],
            }
        b = max(per.values(), key=lambda x: x["stat"])
        null_best.append(b["stat"])
        null_detail.append(per)
        print(f"null {i}: best_stat={b['stat']:.2f}", file=sys.stderr)
    report["selection_premium"] = {
        "N": n_null,
        "shuffle": "reward permuted within each day (day structure preserved)",
        "searched_over": list(VARIANTS),
        "statistic": "pooled per-trade diff bp (selected - remainder), 0 when nothing selected",
        "null_best_of_search": null_best,
        "null_best_max": float(max(null_best)) if null_best else None,
        "null_best_p90": float(np.percentile(null_best, 90)) if null_best else None,
        "null_best_median": float(np.median(null_best)) if null_best else None,
        "observed_best": {n: stat(report["variants"][n]) for n in VARIANTS},
        "detail": null_detail,
    }

    (OUT / f"bandit3_{period}_search.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False)
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("feature_diagnostics", "selection_premium")
            },
            indent=1,
            ensure_ascii=False,
        )[:4000]
    )


if __name__ == "__main__":
    main()
