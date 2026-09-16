"""Does volume/turnover rank entries where the judgment features could not?

The user's strongest prior is that fill strength and volume are the most important
features ("체결력이 가장 중요한 피쳐야 물량이야 물량!"). The bandit's 13 features
contain nothing volume-related at all, so that prior has never actually been
tested on the entry side. This tests it on the full history.

Features, all computed from bars strictly BEFORE the fill minute, so the live
daemon has them at decision time (it already pulls the ticker's minute bars):

  tv_rel_1     last completed minute's turnover / median turnover of the 20 before it
  tv_rel_5     mean turnover of the last 5 minutes / that same median
  vol_trend    mean volume of the last 3 minutes / mean of the 10 before them
  tv_level     log10 of the median minute turnover -- the stock's liquidity level
  tv_since_open  turnover since 09:00 / (minutes elapsed x that median)

Turnover is `close x volume`. `minute_bars.trade_value` is exactly that (verified:
relative error <= 1e-5 on 304,183 bars, fully populated), i.e. a derived column
rather than exchange-reported 거래대금, so nothing is lost computing it here.

What this is NOT: aggressor side, 체결강도, or book pressure. Those live in
`ticks`/`quotes`, which cover 2026-08-24.. only and 21.4% of our actual dashboard
trades. This is minute volume and turnover, and the writeup says so.

Scored on realized KRW under the live capital budget (top-5/day), monthly
time-ordered folds, then applied FROZEN to the dashboard pool -- a different pool
and a later period.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rl_nonlinear import GBM, day_records

from daytrade_it.application.trading import bandit

NOTIONAL = 100_000.0
TOP_K = 5
VOL = ("tv_rel_1", "tv_rel_5", "vol_trend", "tv_level", "tv_since_open")
PRICE4 = ("pre_entry_move", "ret_5m", "mkt_move", "minutes_since_open")


def volume_features(row: dict) -> dict[str, float] | None:
    """Turnover/volume state from the bars completed before the fill minute."""
    buy = int(row["buy_minute"])
    prior = [b for b in row["bars"] if b[0] < buy]
    if len(prior) < 6:
        return None
    tv = np.array([b[4] * b[5] for b in prior], dtype=float)  # close x volume
    vol = np.array([float(b[5]) for b in prior])
    med = float(np.median(tv[-21:-1])) if len(tv) > 1 else 0.0
    if med <= 0:
        med = float(np.median(tv)) or 1.0
    last3 = vol[-3:].mean() if len(vol) >= 3 else vol.mean()
    prev10 = vol[-13:-3].mean() if len(vol) >= 13 else vol.mean()
    minutes = max(1.0, float(prior[-1][0] - 9 * 60 + 1))
    return {
        "tv_rel_1": float(tv[-1] / med),
        "tv_rel_5": float(tv[-5:].mean() / med),
        "vol_trend": float(last3 / prev10) if prev10 > 0 else 1.0,
        "tv_level": float(np.log10(max(med, 1.0))),
        "tv_since_open": float(tv.sum() / (minutes * med)),
    }


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    out = []
    for r in rows:
        if r.get("engine_reward") is None:
            continue
        vf = volume_features(r)
        if vf is None:
            continue
        r["features"] = {**r["features"], **vf}
        out.append(r)
    out.sort(key=lambda r: (r["day"], r["decide_at"]))
    return out


def rank_eval(
    rows: list[dict], y: np.ndarray, feats: tuple[str, ...], months: list[str]
) -> tuple[dict, list[float]]:
    """Monthly time-ordered folds; returns (picked top-K per day, per-fold KRW)."""
    picked: dict[str, list[int]] = {}
    cells = []
    for m in months[1:]:
        tr = [i for i, r in enumerate(rows) if r["day"][:7] < m]
        ev = [i for i, r in enumerate(rows) if r["day"][:7] == m]
        tr_days = sorted({rows[i]["day"] for i in tr})
        cut = tr_days[max(1, int(len(tr_days) * 0.75))]
        inner = [i for i in tr if rows[i]["day"] < cut]
        val = [i for i in tr if rows[i]["day"] >= cut]
        best_r, best_v = 30, None
        for rounds in (10, 30, 60):
            mdl = GBM(rounds, 0.05, 3, 40).fit(
                np.array([[rows[i]["features"][f] for f in feats] for i in inner]),
                y[np.array(inner)],
            )
            sc = mdl.predict(np.array([[rows[i]["features"][f] for f in feats] for i in val]))
            k = max(1, len(sc) // 10)
            v = float(y[np.array(val)][np.argsort(-sc)[:k]].mean())
            if best_v is None or v > best_v:
                best_r, best_v = rounds, v
        mdl = GBM(best_r, 0.05, 3, 40).fit(
            np.array([[rows[i]["features"][f] for f in feats] for i in tr]), y[np.array(tr)]
        )
        sc = mdl.predict(np.array([[rows[i]["features"][f] for f in feats] for i in ev]))
        per: dict[str, list] = defaultdict(list)
        for pos, i in enumerate(ev):
            per[rows[i]["day"]].append((sc[pos], i))
        chosen = {d: [i for _s, i in sorted(v, reverse=True)[:TOP_K]] for d, v in per.items()}
        picked.update(chosen)
        cells.append(sum(y[np.array(v)].sum() for v in chosen.values()) * NOTIONAL)
    return picked, cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=Path, required=True)
    ap.add_argument("--dash-paths", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from bayes_judge import judge

    rows = load(args.paths)
    y = np.array([r["engine_reward"] for r in rows])
    months = sorted({r["day"][:7] for r in rows})
    print(f"entries={len(rows)} days={len({r['day'] for r in rows})} months={months}")
    print(f"pool mean {y.mean() * 1e4:+.0f} bp")

    tv = np.array([r["features"]["tv_rel_1"] for r in rows])
    print(
        f"turnover vs own baseline at entry: median {np.median(tv):.2f},"
        f" p10 {np.quantile(tv, 0.1):.2f}, p90 {np.quantile(tv, 0.9):.2f}"
    )
    for f in VOL:
        x = np.array([r["features"][f] for r in rows])
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        rho = float(np.corrcoef(rx, ry)[0, 1])
        hi, lo = y[x >= np.quantile(x, 0.8)], y[x <= np.quantile(x, 0.2)]
        print(
            f"  {f:16s} spearman {rho:+.3f}   top-quintile {hi.mean() * 1e4:+.0f} bp"
            f" vs bottom {lo.mean() * 1e4:+.0f} bp"
        )

    families = {
        "volume5": VOL,
        "volume5+price4": VOL + PRICE4,
        "all13+volume5": bandit.FEATURES + VOL,
        "all13 (baseline)": bandit.FEATURES,
    }
    by_day: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_day[r["day"]].append(i)
    base_first = {d: v[:TOP_K] for d, v in by_day.items() if d[:7] != months[0]}

    print("\n=== realized KRW, top-5/day, monthly time-ordered folds")
    print(f"{'family':18s} " + " ".join(f"{m:>10s}" for m in months[1:]) + f" {'pooled':>12s}")
    verdicts = {}
    for name, feats in families.items():
        picked, cells = rank_eval(rows, y, feats, months)
        cand = day_records(picked, y)
        base = day_records({d: base_first[d] for d in picked}, y)
        v = judge({c["day"]: c for c in cand}, {b["day"]: b for b in base})
        verdicts[name] = v
        print(
            f"{name:18s} "
            + " ".join(f"{c:>+10,.0f}" for c in cells)
            + f" {sum(c['net'] for c in cand):>+12,.0f}"
        )
    print(
        f"{'BASE live order':18s} "
        + " " * (11 * (len(months) - 1))
        + f" {sum(b['net'] for b in day_records(base_first, y)):>+12,.0f}"
    )
    print()
    for name, v in verdicts.items():
        print(
            f"{name:18s} p(mu>0)={v['p_mu_gt_0']:.3f} mu={v['mu_mean']:+,.0f}"
            f" days>=0={v['days_ge_0']}/{v['n_days']}"
            f" worst {v['worst_day_candidate']:+,.0f} vs {v['worst_day_baseline']:+,.0f}"
            f" losers {v['loser_ratio_candidate']:.3f} vs {v['loser_ratio_baseline']:.3f}"
            f" adopt={v['adopt']}"
        )

    if args.dash_paths:
        print(
            "\n=== frozen on the dashboard pool (trained on every sweep month, applied unchanged)"
        )
        d_rows = load(args.dash_paths)
        dy = np.array([r["engine_reward"] for r in d_rows])
        d_by_day: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(d_rows):
            d_by_day[r["day"]].append(i)
        print(f"{'family':18s} {'top-5 bp':>10s} {'pool bp':>9s} {'first-5 bp':>11s}")
        pool_bp = dy.mean() * 1e4
        first5 = np.concatenate([dy[np.array(v[:TOP_K])] for v in d_by_day.values()])
        for name, feats in families.items():
            mdl = GBM(30, 0.05, 3, 40).fit(
                np.array([[r["features"][f] for f in feats] for r in rows]), y
            )
            sc = mdl.predict(np.array([[r["features"][f] for f in feats] for r in d_rows]))
            sel = []
            for _d, idx in d_by_day.items():
                order = sorted(idx, key=lambda i: -sc[i])[:TOP_K]
                sel += order
            print(
                f"{name:18s} {dy[np.array(sel)].mean() * 1e4:>+10.0f}"
                f" {pool_bp:>+9.0f} {first5.mean() * 1e4:>+11.0f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(verdicts, indent=1, default=str))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
