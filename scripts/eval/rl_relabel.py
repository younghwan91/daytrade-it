"""Can the entry features rank news once the label is decontaminated?

The realized per-pair reward bakes in the exit rule, so a genuinely good article
whose gain was given back is labelled bad. That is a candidate explanation for
why the 13 decision-time features could not rank pairs even in-sample while a
hindsight oracle found +245 to +357 bp. This refits the selector on a label that
does not involve the exit at all: the best move available after the decision
minute.

Three guards, because the raw maximum is a trap:

1. **σ-normalised.** A maximum scales with volatility mechanically. On the
   September window every group -- including random tickers -- sat at 5.3-5.8
   sigma of maximum, so a selector trained on the raw maximum would learn which
   stock is volatile, not which news is good. The primary label is therefore
   `mfe / sigma`, with the raw label reported alongside to show the difference.
2. **Several horizons**, so the label is not an artifact of one window.
3. **Time-ordered monthly folds only.** Train strictly before the evaluation
   month; LODO is banned.

The label is NOT achievable and never appears as P&L: nobody sells at the
maximum. It ranks news. Anything that wins here still has to be verified on
realized KRW under an actual exit rule.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from exit_lab import Path as EntryPath

from daytrade_it.application.trading import bandit

HORIZONS = (30, 60, 120, "close")
CLOSE_MINUTE = 15 * 60
SIGMA_FLOOR = 0.0015
LAMS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
FAMILIES: dict[str, tuple[str, ...]] = {
    "all13": bandit.FEATURES,
    "judgment7": (
        "score",
        "surprise",
        "specificity",
        "sentiment_strength",
        "persistence_one_off",
        "persistence_multi_quarter",
        "subject_main",
    ),
    "price4": ("pre_entry_move", "ret_5m", "mkt_move", "minutes_since_open"),
}


def labels_for(p: EntryPath) -> dict[str, float] | None:
    """Post-decision maxima at each horizon, plus the pre-entry sigma that scales them."""
    by_min = {b.ts.hour * 60 + b.ts.minute: b for b in p.bars}
    after = sorted(m for m in by_min if m > p.buy_minute)
    if not after:
        return None
    entry = float(p.buy_price)
    if entry <= 0:
        return None
    prior = sorted(m for m in by_min if m < p.buy_minute and by_min[m].close > 0)[-31:]
    if len(prior) < 3:
        return None
    c = np.array([float(by_min[m].close) for m in prior])
    sigma = max(float((c[1:] / c[:-1] - 1).std()), SIGMA_FLOOR)

    out: dict[str, float] = {"sigma": sigma}
    for h in HORIZONS:
        end = CLOSE_MINUTE if h == "close" else p.buy_minute + int(h)
        window = [m for m in after if m <= end]
        mfe = (max(float(by_min[m].high) for m in window) / entry - 1) if window else 0.0
        out[f"mfe_{h}"] = mfe
        out[f"z_{h}"] = mfe / sigma
    peak = max((m for m in after if m <= CLOSE_MINUTE), key=lambda m: by_min[m].high, default=None)
    out["peak_min"] = float(peak - p.buy_minute) if peak is not None else 0.0
    return out


def fit(rows: list[dict], names: tuple[str, ...], lam: float, y: np.ndarray) -> bandit.BanditModel:
    X = np.array([[r["features"][f] for f in names] for r in rows], dtype=float)
    return bandit.fit(X, y, lam=lam, feature_names=names)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(set(x.tolist())) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float(rx @ ry / d) if d else float("nan")


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    raw = [json.loads(line) for line in args.paths.read_text().splitlines() if line]
    rows, labs = [], []
    for r in raw:
        lab = labels_for(EntryPath.from_row(r))
        if lab is not None:
            rows.append(r)
            labs.append(lab)
    days = np.array([r["day"] for r in rows])
    months = sorted({d[:7] for d in days.tolist()})
    realized = np.array([r["engine_reward"] for r in rows], dtype=float)
    print(
        f"paths={len(raw)}  labelled={len(rows)}  dropped={len(raw) - len(rows)} (no pre-entry bars)"
    )
    print(f"days={len(set(days.tolist()))} months={months}")

    L = {k: np.array([lab[k] for lab in labs]) for k in labs[0]}
    print("\nlabel scale (bp for mfe, sigmas for z):")
    for h in HORIZONS:
        print(
            f"  h={str(h):>5s}  mfe mean {L[f'mfe_{h}'].mean() * 1e4:>+6.0f} bp"
            f"  z mean {L[f'z_{h}'].mean():>5.2f}  z sd {L[f'z_{h}'].std():>5.2f}"
        )
    print(f"  sigma: median {np.median(L['sigma']) * 1e4:.0f} bp")
    print(
        f"  peak minute: p25 {np.quantile(L['peak_min'], 0.25):.0f}"
        f"  median {np.median(L['peak_min']):.0f}"
        f"  p75 {np.quantile(L['peak_min'], 0.75):.0f}"
        f"  share peaking in first 5 min {float((L['peak_min'] <= 5).mean()):.3f}"
    )
    print(f"  corr(realized, mfe_close) {np.corrcoef(realized, L['mfe_close'])[0, 1]:+.3f}")
    print(f"  corr(realized, z_close)   {np.corrcoef(realized, L['z_close'])[0, 1]:+.3f}")

    # Does the raw label just encode volatility?
    print(
        f"\n  corr(sigma, mfe_close) {np.corrcoef(L['sigma'], L['mfe_close'])[0, 1]:+.3f}"
        f"   corr(sigma, z_close) {np.corrcoef(L['sigma'], L['z_close'])[0, 1]:+.3f}"
    )

    # Univariate, month by month, on the sigma-normalised label.
    print("\n=== Spearman(feature, z_close) by month  [stab = share on the modal sign]")
    print(f"{'feature':26s} " + " ".join(f"{m[-2:]:>7s}" for m in months) + "    all   stab")
    uni = {}
    for f in bandit.FEATURES:
        per = []
        for m in months:
            sel = np.array([d[:7] == m for d in days.tolist()])
            x = np.array([r["features"][f] for r in np.array(rows, dtype=object)[sel]])
            per.append(spearman(x, L["z_close"][sel]))
        allx = np.array([r["features"][f] for r in rows])
        rho = spearman(allx, L["z_close"])
        signs = [1 if v > 0 else -1 for v in per if v == v]
        stab = max(signs.count(1), signs.count(-1)) / len(signs) if signs else float("nan")
        uni[f] = {"per_month": per, "all": rho, "stab": stab}
        print(
            f"{f:26s} "
            + " ".join(f"{v:+7.3f}" if v == v else "    n/a" for v in per)
            + f" {rho:+6.3f}  {stab:.2f}"
        )

    # Monthly time-ordered folds: rank by the model, measure the label of the top decile.
    print("\n=== time-ordered folds: mean z_close of the top-ranked 10% vs the month's mean")
    results: dict = defaultdict(dict)
    for label_key in ("z_close", "mfe_close"):
        y_all = L[label_key]
        print(f"\n-- label {label_key}")
        print(f"{'family':11s} " + " ".join(f"{m:>9s}" for m in months[1:]) + f" {'pooled':>9s}")
        for fam, names in FAMILIES.items():
            cells, pooled_top, pooled_base = [], [], []
            for m in months[1:]:
                tr = np.array([d[:7] < m for d in days.tolist()])
                ev = np.array([d[:7] == m for d in days.tolist()])
                tr_rows = [r for r, t in zip(rows, tr, strict=True) if t]
                ev_rows = [r for r, e in zip(rows, ev, strict=True) if e]
                # lam chosen on a time-ordered inner split of the training months
                tr_days = sorted({r["day"] for r in tr_rows})
                cut = tr_days[max(1, int(len(tr_days) * 0.75))]
                inner_fit = [r for r in tr_rows if r["day"] < cut]
                inner_val = [r for r in tr_rows if r["day"] >= cut]
                yf = np.array([y_all[i] for i, r in enumerate(rows) if tr[i] and r["day"] < cut])
                yv = np.array([y_all[i] for i, r in enumerate(rows) if tr[i] and r["day"] >= cut])
                best, best_lam = None, LAMS[0]
                for lam in LAMS:
                    mdl = fit(inner_fit, names, lam, yf)
                    sc = np.array(
                        [mdl.predict({f: r["features"][f] for f in names})[0] for r in inner_val]
                    )
                    k = max(1, len(sc) // 10)
                    top = yv[np.argsort(-sc)[:k]].mean()
                    if best is None or top > best:
                        best, best_lam = top, lam
                ytr = y_all[tr]
                mdl = fit(tr_rows, names, best_lam, ytr)
                sc = np.array(
                    [mdl.predict({f: r["features"][f] for f in names})[0] for r in ev_rows]
                )
                yev = y_all[ev]
                k = max(1, len(sc) // 10)
                top = yev[np.argsort(-sc)[:k]]
                cells.append(top.mean() - yev.mean())
                pooled_top.append(top)
                pooled_base.append(yev)
            allt = np.concatenate(pooled_top)
            allb = np.concatenate(pooled_base)
            se = float(np.sqrt(allt.var() / len(allt) + allb.var() / len(allb)))
            scale = 1.0 if label_key.startswith("z") else 1e4
            results[label_key][fam] = {
                "per_fold": [c * scale for c in cells],
                "pooled": float((allt.mean() - allb.mean()) * scale),
                "se": se * scale,
                "n_top": len(allt),
            }
            print(
                f"{fam:11s} "
                + " ".join(f"{c * scale:>+9.3f}" for c in cells)
                + f" {(allt.mean() - allb.mean()) * scale:>+9.3f}"
                f"  (se {se * scale:.3f}, n_top {len(allt)})"
            )

    # ---- the number that actually decides it: realized KRW, out-of-sample ----
    # Rank by a model TRAINED ON THE DECONTAMINATED LABEL, then spend the live
    # capital budget (5 entries/day) and score with the REALIZED reward. A
    # ranking that wins on the maximum but not here is not adopted.
    from bayes_judge import judge  # noqa: PLC0415

    NOTIONAL = 100_000.0
    TOP_K = 5

    def day_recs(idx_by_day: dict[str, list[int]]) -> list[dict]:
        out = []
        for day in sorted(idx_by_day):
            r = realized[np.array(idx_by_day[day])]
            out.append(
                {
                    "day": day,
                    "net": NOTIONAL * float(r.sum()),
                    "trades": len(r),
                    "losers": int((r < 0).sum()),
                    "worst_trade": NOTIONAL * float(r.min()),
                    "errors": 0,
                    "leftovers": 0,
                }
            )
        return out

    print("\n=== REALIZED KRW, out-of-sample: rank by the z_close model, take top 5/day")
    print(f"{'family':11s} " + " ".join(f"{m:>10s}" for m in months[1:]) + f" {'pooled':>12s}")
    order_by_day: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        order_by_day[r["day"]].append(i)
    for day in order_by_day:
        order_by_day[day].sort(key=lambda i: rows[i]["decide_at"])
    base_first = day_recs({d: v[:TOP_K] for d, v in order_by_day.items()})
    verdicts_krw = {}
    for fam, names in FAMILIES.items():
        cand_by_day: dict[str, list[int]] = {}
        cells = []
        for m in months[1:]:
            ev_idx = [i for i, r in enumerate(rows) if r["day"][:7] == m]
            tr_idx = [i for i, r in enumerate(rows) if r["day"][:7] < m]
            tr_rows = [rows[i] for i in tr_idx]
            mdl = fit(tr_rows, names, 10.0, L["z_close"][np.array(tr_idx)])
            sc = {i: mdl.predict({f: rows[i]["features"][f] for f in names})[0] for i in ev_idx}
            per_day: dict[str, list[int]] = defaultdict(list)
            for i in ev_idx:
                per_day[rows[i]["day"]].append(i)
            picked = {d: sorted(v, key=lambda i: -sc[i])[:TOP_K] for d, v in per_day.items()}
            cand_by_day.update(picked)
            net = sum(realized[np.array(v)].sum() for v in picked.values()) * NOTIONAL
            cells.append(net)
        cand = day_recs(cand_by_day)
        base = {d["day"]: d for d in base_first if d["day"] in {c["day"] for c in cand}}
        v = judge({c["day"]: c for c in cand}, base)
        verdicts_krw[fam] = v
        print(
            f"{fam:11s} "
            + " ".join(f"{c:>+10,.0f}" for c in cells)
            + f" {sum(c['net'] for c in cand):>+12,.0f}"
        )
    print(
        f"{'BASE first5':11s} "
        + " " * (11 * (len(months) - 1))
        + f" {sum(d['net'] for d in base_first if d['day'][:7] != months[0]):>+12,.0f}"
    )
    print()
    for fam, v in verdicts_krw.items():
        print(
            f"{fam:11s} vs live-order p(mu>0)={v['p_mu_gt_0']:.3f} mu={v['mu_mean']:+,.0f}"
            f" days>=0={v['days_ge_0']}/{v['n_days']}"
            f" worst {v['worst_day_candidate']:+,.0f} vs {v['worst_day_baseline']:+,.0f}"
            f" losers {v['loser_ratio_candidate']:.3f} vs {v['loser_ratio_baseline']:.3f}"
            f" adopt={v['adopt']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"univariate": uni, "folds": results, "realized_krw": verdicts_krw},
            indent=1,
            default=str,
        )
    )
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
