"""Breadth + generalisation search over the rlnews feature families.

Time-ordered expanding folds on the SEARCH window only. Nested inner split for
hyperparameters. Identical search re-run on within-day-shuffled targets to price
the selection premium. The frozen confirm window is touched once, by a separate
invocation (`--confirm`).

Usage:
  model.py search  <prior|judged> [--shuffles R]
  model.py confirm <prior|judged> --family F --model M --config JSON
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

OUT = Path("/home/young/git/daytrade-it-rlnews/data/eval/rlnews")
CONF = json.loads((OUT / "CONFIRM_WINDOWS.json").read_text())

BASE = [
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
]
VOL = [
    "log1p_cum_value",
    "rvol_cum",
    "rvol_last",
    "rvol_5",
    "vol_accel",
    "vol_share_5",
    "traded_frac",
    "turnover_vs_prevday",
    "log1p_last_bar_value",
]
FREE = [
    "fanout",
    "pub_to_act_min",
    "move_since_pub",
    "move_before_pub",
    "ret_since_open",
    "prevday_val_rank",
    "prevday_abs_move",
    "prevday_range",
    "gap_open",
    "log_entry",
    "tick_bp",
    "dow",
    "is_kosdaq",
    "n_bars_done",
]
FREE_CATS = ["sector", "session", "title_tag"]
V2_ORD = {
    "subject": ["mention", "partial", "main"],
    "persistence": ["one_off", "multi_quarter", "structural"],
    "freshness": ["repeat", "follow_up", "new"],
    "scale_vs_size": ["unknown", "small", "medium", "large"],
}
V2_NUM = ["specificity", "surprise", "sentiment_direction", "sentiment_strength"]
V2_FEATS = list(V2_ORD) + V2_NUM + ["reports_price_move", "material_type"]

TOP_FRAC = 0.05


def load(period: str):
    rows = []
    with open(OUT / f"feats2_{period}.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build(rows, has_v2: bool):
    n = len(rows)
    num_names = BASE + VOL + FREE
    X = {k: np.array([r["features"][k] for r in rows], dtype=float) for k in num_names}
    cat_raw = {c: [r["cats"][c] for r in rows] for c in FREE_CATS}
    if has_v2:
        for k, order in V2_ORD.items():
            m = {v: i for i, v in enumerate(order)}
            X["v2_" + k] = np.array([float(m.get(r["v2"][k], -1)) for r in rows])
        for k in V2_NUM:
            X["v2_" + k] = np.array([float(r["v2"][k]) for r in rows])
        X["v2_reports_price_move"] = np.array(
            [float(bool(r["v2"]["reports_price_move"])) for r in rows]
        )
        cat_raw["v2_material_type"] = [r["v2"]["material_type"] for r in rows]
    day = np.array([r["day"] for r in rows])
    reward = np.array([r["reward"] for r in rows])
    alpha = np.array([(r["alpha"] if r["alpha"] is not None else np.nan) for r in rows])
    return X, cat_raw, day, reward, alpha, n


def codes(values, min_n=100):
    from collections import Counter

    c = Counter(values)
    keep = sorted([k for k, v in c.items() if v >= min_n], key=lambda k: -c[k])
    m = {k: i + 1 for i, k in enumerate(keep)}
    return np.array([m.get(v, 0) for v in values], dtype=float), len(m) + 1


def onehot(values, min_n=100):
    arr, k = codes(values, min_n)
    M = np.zeros((len(values), k))
    M[np.arange(len(values)), arr.astype(int)] = 1.0
    return M[:, 1:]


def families(has_v2: bool):
    f = {
        "base10": (BASE, []),
        "base+vol": (BASE + VOL, []),
        "base+free": (BASE + FREE, FREE_CATS),
        "base+vol+free": (BASE + VOL + FREE, FREE_CATS),
    }
    if has_v2:
        v2n = ["v2_" + k for k in list(V2_ORD) + V2_NUM] + ["v2_reports_price_move"]
        f["base+v2"] = (BASE + v2n, ["v2_material_type"])
        f["base+vol+free+v2"] = (BASE + VOL + FREE + v2n, FREE_CATS + ["v2_material_type"])
    return f


GBM_CFGS = [
    {"max_leaf_nodes": ln, "learning_rate": lr, "min_samples_leaf": 100}
    for ln in (7, 15, 31)
    for lr in (0.03, 0.06)
]
RIDGE_CFGS = [{"alpha": a} for a in (1.0, 10.0, 100.0, 1000.0)]


def fit_predict(model, cfg, Xtr, ytr, Xte, cat_mask):
    if model == "gbm":
        m = HistGradientBoostingRegressor(
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=0,
            l2_regularization=1.0,
            categorical_features=cat_mask if cat_mask.any() else None,
            **cfg,
        )
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    m = Ridge(**cfg)
    m.fit((Xtr - mu) / sd, ytr)
    return m.predict((Xte - mu) / sd)


def sel_metric(pred, day, alpha, reward, frac=TOP_FRAC):
    """Mean alpha / reward of the top-`frac` predictions of each day, in bp."""
    a, r, _w = [], [], 0
    for d in np.unique(day):
        i = np.where(day == d)[0]
        k = max(1, int(round(frac * len(i))))
        top = i[np.argsort(-pred[i])[:k]]
        a.extend(alpha[top])
        r.extend(reward[top])
    a = np.array(a)
    r = np.array(r)
    return float(np.nanmean(a) * 1e4), float(np.nanmean(r) * 1e4), len(r)


def make_matrix(X, cat_raw, num_names, cat_names, gbm: bool):
    parts = [np.column_stack([X[k] for k in num_names])]
    mask = [False] * len(num_names)
    for c in cat_names:
        if gbm:
            arr, _ = codes(cat_raw[c])
            parts.append(arr[:, None])
            mask.append(True)
        else:
            M = onehot(cat_raw[c])
            parts.append(M)
            mask.extend([False] * M.shape[1])
    return np.column_stack(parts), np.array(mask)


def run_search(period, shuffles, out_path, target="alpha"):
    rows = load(period)
    has_v2 = "v2" in rows[0]
    confirm = set(CONF["track_a_" + period]["confirm"])
    rows = [r for r in rows if r["day"] not in confirm]
    X, cat_raw, day, reward, alpha, n = build(rows, has_v2)
    fam = families(has_v2)
    days = np.unique(day)
    L = len(days)
    bounds = [int(round(L * i / 6)) for i in range(1, 7)]
    folds = [(days[: bounds[i]], days[bounds[i] : bounds[i + 1]]) for i in range(5)]

    mats = {}
    for fname, (nn, cn) in fam.items():
        for gbm in (True, False):
            mats[(fname, gbm)] = make_matrix(X, cat_raw, nn, cn, gbm)

    y_all = reward if target == "reward" else np.where(np.isnan(alpha), reward, alpha)

    def one_pass(y):
        res = {}
        for fname in fam:
            for model, cfgs in (("gbm", GBM_CFGS), ("ridge", RIDGE_CFGS)):
                M, mask = mats[(fname, model == "gbm")]
                fold_a, fold_r, picks = [], [], []
                for tr_days, te_days in folds:
                    itr = np.isin(day, tr_days)
                    ite = np.isin(day, te_days)
                    cut = tr_days[int(0.8 * len(tr_days))]
                    i_in = itr & (day < cut)
                    i_val = itr & (day >= cut)
                    best, bcfg = -1e18, None
                    for cfg in cfgs:
                        p = fit_predict(model, cfg, M[i_in], y[i_in], M[i_val], mask)
                        s, _, _ = sel_metric(p, day[i_val], y[i_val], reward[i_val])
                        if s > best:
                            best, bcfg = s, cfg
                    p = fit_predict(model, bcfg, M[itr], y[itr], M[ite], mask)
                    a, r, _ = sel_metric(p, day[ite], alpha[ite], reward[ite])
                    fold_a.append(a)
                    fold_r.append(r)
                    picks.append(bcfg)
                res[(fname, model)] = {
                    "fold_alpha_bp": [round(x, 2) for x in fold_a],
                    "mean_alpha_bp": round(float(np.mean(fold_a)), 2),
                    "mean_reward_bp": round(float(np.mean(fold_r)), 2),
                    "picks": picks,
                }
        return res

    t0 = time.time()
    real = one_pass(y_all)
    print(f"real pass {time.time() - t0:.0f}s", file=sys.stderr)

    base_a = float(np.nanmean(alpha) * 1e4)
    base_r = float(np.nanmean(reward) * 1e4)

    prem = []
    rng = np.random.default_rng(7)
    for rep in range(shuffles):
        ysh = y_all.copy()
        for d in days:
            i = np.where(day == d)[0]
            ysh[i] = ysh[rng.permutation(i)]
        rs = one_pass(ysh)
        prem.append(
            {
                "alpha": max(v["mean_alpha_bp"] for v in rs.values()),
                "reward": max(v["mean_reward_bp"] for v in rs.values()),
            }
        )
        print(f"shuffle {rep} best {prem[-1]}", file=sys.stderr)

    out = {
        "period": period,
        "target": target,
        "n_pairs": len(rows),
        "n_search_days": L,
        "all_pairs_alpha_bp": round(base_a, 2),
        "all_pairs_reward_bp": round(base_r, 2),
        "N_configs": len(fam) * (len(GBM_CFGS) + len(RIDGE_CFGS)),
        "n_families": len(fam),
        "results": {f"{k[0]}|{k[1]}": v for k, v in real.items()},
        "shuffle_best_of_N": prem,
        "shuffle_mean": {
            k: round(float(np.mean([x[k] for x in prem])), 2) for k in ("alpha", "reward")
        }
        if prem
        else None,
        "shuffle_p90": {
            k: round(float(np.percentile([x[k] for x in prem], 90)), 2) for k in ("alpha", "reward")
        }
        if prem
        else None,
    }
    Path(out_path).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1))


def run_confirm(period, fname, model, out_path, target="alpha"):
    rows = load(period)
    has_v2 = "v2" in rows[0]
    confirm = set(CONF["track_a_" + period]["confirm"])
    X, cat_raw, day, reward, alpha, n = build(rows, has_v2)
    fam = families(has_v2)
    nn, cn = fam[fname]
    M, mask = make_matrix(X, cat_raw, nn, cn, model == "gbm")
    y_all = reward if target == "reward" else np.where(np.isnan(alpha), reward, alpha)
    ite = np.isin(day, sorted(confirm))
    itr = ~ite
    tr_days = np.unique(day[itr])
    cut = tr_days[int(0.8 * len(tr_days))]
    i_in = itr & (day < cut)
    i_val = itr & (day >= cut)
    cfgs = GBM_CFGS if model == "gbm" else RIDGE_CFGS
    best, bcfg = -1e18, None
    for cfg in cfgs:
        p = fit_predict(model, cfg, M[i_in], y_all[i_in], M[i_val], mask)
        s, _, _ = sel_metric(p, day[i_val], y_all[i_val], reward[i_val])
        if s > best:
            best, bcfg = s, cfg
    p = fit_predict(model, bcfg, M[itr], y_all[itr], M[ite], mask)
    a, r, k = sel_metric(p, day[ite], alpha[ite], reward[ite])
    months = {}
    dte = day[ite]
    for mo in sorted({d[:7] for d in dte}):
        sel = np.array([d[:7] == mo for d in dte])
        aa, rr, kk = sel_metric(p[sel], dte[sel], alpha[ite][sel], reward[ite][sel])
        months[mo] = {"alpha_bp": round(aa, 2), "reward_bp": round(rr, 2), "n": kk}
    daily = {}
    for d in sorted(set(dte)):
        sel = dte == d
        aa, rr, kk = sel_metric(p[sel], dte[sel], alpha[ite][sel], reward[ite][sel])
        daily[d] = [round(aa, 2), round(rr, 2), kk]
    dl = sorted(daily)
    A = np.array([daily[d][0] for d in dl])
    R = np.array([daily[d][1] for d in dl])
    W = np.array([daily[d][2] for d in dl], dtype=float)
    rng = np.random.default_rng(3)
    ba, br = [], []
    for _ in range(4000):
        j = rng.integers(0, len(dl), len(dl))
        ba.append(np.average(A[j], weights=W[j]))
        br.append(np.average(R[j], weights=W[j]))
    boot = {"alpha_se_bp": round(float(np.std(ba)), 2), "reward_se_bp": round(float(np.std(br)), 2)}
    out = {
        "period": period,
        "target": target,
        "family": fname,
        "model": model,
        "cfg": bcfg,
        "confirm_days": len(confirm),
        "n_selected": k,
        "confirm_alpha_bp": round(a, 2),
        "confirm_reward_bp": round(r, 2),
        "all_pairs_alpha_bp": round(float(np.nanmean(alpha[ite]) * 1e4), 2),
        "all_pairs_reward_bp": round(float(np.nanmean(reward[ite]) * 1e4), 2),
        "by_month": months,
        "daily_alpha_bp": daily,
        "days_positive_alpha": int((A > 0).sum()),
        "days_positive_reward": int((R > 0).sum()),
        "day_clustered_se": boot,
    }
    Path(out_path).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("period")
    ap.add_argument("--shuffles", type=int, default=10)
    ap.add_argument("--family")
    ap.add_argument("--model")
    ap.add_argument("--target", default="alpha")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "search":
        run_search(a.period, a.shuffles, a.out, a.target)
    else:
        run_confirm(a.period, a.family, a.model, a.out, a.target)
