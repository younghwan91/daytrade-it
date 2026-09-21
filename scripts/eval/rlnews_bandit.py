"""Lab-level contextual-bandit entry study on hold-to-close net returns.

    bandit_rlnews.py <period>          period in {prior, judged}

Monthly time-ordered folds: train on every row from a day strictly before the
evaluated month; hyperparameters (lam, min_prob, min_edge) chosen on the
training months only, by an inner TIME-ORDERED split (last training month as
validation).  No LODO anywhere.  Reward = ret_close - round_trip(...).
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts/eval")
from bandit_lodo import GRID, LAMS, objective  # noqa: E402
from bayes_judge import judge  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

from daytrade_it.application.trading import bandit  # noqa: E402

FEATS: tuple[str, ...] = (
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
NOTIONAL = 100_000.0
OUT = Path("/home/young/git/daytrade-it-rlnews/data/eval/rlnews")


def fit_rows(rows, lam, min_prob=0.5, min_edge=0.0):
    X = np.array([[r["features"][f] for f in FEATS] for r in rows], dtype=float)
    y = np.array([r["reward"] for r in rows], dtype=float)
    return bandit.fit(X, y, lam=lam, min_prob=min_prob, min_edge=min_edge, feature_names=FEATS)


def _mp(model, rows):
    """Vectorized (m, p) for every row under the NIG posterior predictive.

    Same formula as `bandit.predict`; `scipy.stats.t.cdf` replaces the module's
    scalar `student_t_cdf` (verified to agree to <1e-9 on a sample).
    """
    X = np.array([[r["features"][f] for f in FEATS] for r in rows], dtype=float)
    Z = np.hstack([np.ones((X.shape[0], 1)), (X - model.means) / model.stds])
    m = Z @ model.w
    scale = np.sqrt((model.b / model.a) * (1.0 + np.einsum("ij,jk,ik->i", Z, model.V, Z)))
    p = student_t.cdf(m / scale, 2.0 * model.a)
    return m, p


def choose(rows) -> dict:
    """Grid pick on the training rows only, via an inner time-ordered split."""
    months = sorted({r["month"] for r in rows})
    if len(months) < 2:
        return {"lam": LAMS[0], "min_prob": 0.7, "min_edge": 0.005, "inner": "insufficient"}
    val_month = months[-1]
    inner_train = [r for r in rows if r["month"] < val_month]
    inner_val = [r for r in rows if r["month"] == val_month]
    val_days = sorted({r["day"] for r in inner_val})
    best, best_obj = None, -np.inf
    val_rewards = np.array([r["reward"] for r in inner_val])
    val_day = np.array([r["day"] for r in inner_val])
    for lam in LAMS:
        model = fit_rows(inner_train, lam)
        m, p = _mp(model, inner_val)
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


def day_rec(day, rows_sel):
    net = sum(r["reward"] for r in rows_sel) * NOTIONAL
    return {
        "day": day,
        "net": net,
        "trades": len(rows_sel),
        "losers": sum(1 for r in rows_sel if r["reward"] < 0),
        "worst_trade": min((r["reward"] * NOTIONAL for r in rows_sel), default=0.0),
        "errors": 0,
        "leftovers": 0,
    }


def boot_se(sel, rem, n=2000, seed=7):
    """Day-clustered bootstrap se of mean(sel) - mean(rem), in bp."""
    days = sorted({r["day"] for r in sel} | {r["day"] for r in rem})
    by_s = collections.defaultdict(list)
    by_r = collections.defaultdict(list)
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
        if not s or not rr:
            continue
        out.append(np.mean(s) - np.mean(rr))
    return float(np.std(out) * 1e4) if out else float("nan")


def main() -> None:
    period = sys.argv[1]
    with (OUT / f"feats_{period}.jsonl").open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    dropped_days = sorted({r["day"] for r in rows if not r["has_mkt"]})
    rows = [r for r in rows if r["has_mkt"]]
    months = sorted({r["month"] for r in rows})

    folds, models, all_cand, all_base, sel_all, rem_all = [], {}, [], [], [], []
    for m in months[1:]:
        train = [r for r in rows if r["month"] < m]
        ev = [r for r in rows if r["month"] == m]
        if len({r["day"] for r in train}) < 20:
            print(f"fold {m}: skipped ({len(train)} train rows)", file=sys.stderr)
            continue
        hp = choose(train)
        model = fit_rows(train, hp["lam"], hp["min_prob"], hp["min_edge"])
        models[m] = model
        m_ev, p_ev = _mp(model, ev)
        mask = (p_ev >= model.min_prob) & (m_ev >= model.min_edge)
        sel = [r for r, on in zip(ev, mask, strict=True) if on]
        rem = [r for r, on in zip(ev, mask, strict=True) if not on]
        days = sorted({r["day"] for r in ev})
        by_sel = collections.defaultdict(list)
        by_all = collections.defaultdict(list)
        for r in sel:
            by_sel[r["day"]].append(r)
        for r in ev:
            by_all[r["day"]].append(r)
        cand = [day_rec(d, by_sel[d]) for d in days]
        base = [day_rec(d, by_all[d]) for d in days]
        all_cand += cand
        all_base += base
        sel_all += sel
        rem_all += rem
        folds.append(
            {
                "month": m,
                "n_train_rows": len(train),
                "n_train_days": len({r["day"] for r in train}),
                "hp": hp,
                "n_eval_pairs": len(ev),
                "n_selected": len(sel),
                "sel_rate": len(sel) / len(ev),
                "per_trade_net_bp_selected": 1e4 * float(np.mean([r["reward"] for r in sel]))
                if sel
                else None,
                "per_trade_net_bp_remainder": 1e4 * float(np.mean([r["reward"] for r in rem]))
                if rem
                else None,
                "per_trade_net_bp_all": 1e4 * float(np.mean([r["reward"] for r in ev])),
                "cand_total_net": sum(x["net"] for x in cand),
                "base_total_net": sum(x["net"] for x in base),
                "w": {n: float(v) for n, v in zip(("_intercept", *FEATS), model.w, strict=True)},
            }
        )
        print(
            f"fold {m}: train {len(train)}r/{folds[-1]['n_train_days']}d "
            f"lam={hp['lam']} p={hp['min_prob']} e={hp['min_edge']} "
            f"sel {len(sel)}/{len(ev)} "
            f"pt_sel={folds[-1]['per_trade_net_bp_selected']} "
            f"pt_rem={folds[-1]['per_trade_net_bp_remainder']}",
            file=sys.stderr,
        )

    verdict = judge({r["day"]: r for r in all_cand}, {r["day"]: r for r in all_base})
    signs = {}
    for nm in ("_intercept", *FEATS):
        s = {f["month"]: ("+" if f["w"][nm] > 0 else "-" if f["w"][nm] < 0 else "0") for f in folds}
        signs[nm] = {
            "by_fold": s,
            "consistent": len(set(s.values())) == 1,
            "modal_share": max(collections.Counter(s.values()).values()) / max(1, len(s)),
            "mean_abs_w": float(np.mean([abs(f["w"][nm]) for f in folds])),
        }
    lams = [f["hp"]["lam"] for f in folds]
    summary = {
        "period": period,
        "dropped_days_no_mkt": dropped_days,
        "n_days": len({r["day"] for r in rows}),
        "n_pairs": len(rows),
        "n_folds": len(folds),
        "lambdas": {f["month"]: f["hp"]["lam"] for f in folds},
        "lam_at_grid_max_every_fold": all(x == max(LAMS) for x in lams),
        "lam_at_grid_extreme_every_fold": all(x in (max(LAMS), min(LAMS)) for x in lams),
        "min_probs": {f["month"]: f["hp"]["min_prob"] for f in folds},
        "min_edges": {f["month"]: f["hp"]["min_edge"] for f in folds},
        "pooled": {
            "n_selected": len(sel_all),
            "n_available": len(sel_all) + len(rem_all),
            "sel_rate": len(sel_all) / max(1, len(sel_all) + len(rem_all)),
            "per_trade_net_bp_selected": 1e4 * float(np.mean([r["reward"] for r in sel_all]))
            if sel_all
            else None,
            "per_trade_net_bp_remainder": 1e4 * float(np.mean([r["reward"] for r in rem_all]))
            if rem_all
            else None,
            "per_trade_net_bp_all": 1e4 * float(np.mean([r["reward"] for r in sel_all + rem_all])),
            "diff_bp": (
                1e4
                * (
                    float(np.mean([r["reward"] for r in sel_all]))
                    - float(np.mean([r["reward"] for r in rem_all]))
                )
                if sel_all and rem_all
                else None
            ),
            "diff_bp_se_day_clustered": boot_se(sel_all, rem_all) if sel_all and rem_all else None,
            "cand_total_net": sum(x["net"] for x in all_cand),
            "base_total_net": sum(x["net"] for x in all_base),
            "days_cand_zero_trades": sum(1 for x in all_cand if x["trades"] == 0),
        },
        "judge": verdict,
        "signs": signs,
        "folds": folds,
    }
    (OUT / f"bandit_{period}_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False)
    )
    (OUT / f"bandit_{period}_oof.json").write_text(
        json.dumps({"candidate": all_cand, "baseline": all_base}, indent=1)
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=1, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
