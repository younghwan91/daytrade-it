"""Does a nonlinear model over the free deterministic features rank entries?

The entry-ranking work so far used linear and monotone families only -- a
13-feature Bayesian linear model and a signed-standardised monotone score. Both
found essentially nothing. prompt-opt's permutation test says that is a family
problem rather than a data problem: shuffling the FREE deterministic attributes
collapses their gradient-boosted model (+34 -> -34 bp) while shuffling the LLM
judgment fields barely moves it (+34 -> +26). That is a statement about *where*
the signal is -- in interactions among the free attributes, which a linear or
monotone rule cannot express.

This tests that on the version that counts: the admitted set (pairs that cleared
the live gates and actually filled), scored on realized KRW, monthly
time-ordered folds. Their 9%-of-oracle-gap figure is held as a hypothesis, not a
baseline -- it leans on May, ignores capital contention, and is the sweep pool.

Gradient boosting is implemented here directly (depth-limited regression trees on
squared error) rather than pulling in a dependency the project does not have. It
is deliberately small: with ~4,300 rows the risk is variance, so the defaults are
shallow trees, a low learning rate and few rounds, and the number of rounds is
chosen inside each fold on a time-ordered inner split, never on the evaluation
month.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from daytrade_it.application.trading import bandit

NOTIONAL = 100_000.0
TOP_K = 5
#: The "free" attributes: known at decision time without an LLM call.
FREE = ("pre_entry_move", "ret_5m", "mkt_move", "minutes_since_open", "premarket")
JUDGMENT = (
    "score",
    "surprise",
    "specificity",
    "sentiment_strength",
    "persistence_one_off",
    "persistence_multi_quarter",
    "persistence_structural",
    "subject_main",
)


class Tree:
    """One depth-limited regression tree on squared error."""

    def __init__(self, depth: int, min_leaf: int) -> None:
        self.depth = depth
        self.min_leaf = min_leaf
        self.node: dict | None = None

    def fit(self, X: np.ndarray, g: np.ndarray) -> Tree:
        self.node = self._build(X, g, 0)
        return self

    def _build(self, X: np.ndarray, g: np.ndarray, d: int) -> dict:
        if d >= self.depth or len(g) < 2 * self.min_leaf:
            return {"leaf": float(g.mean())}
        best = None
        parent = float((g**2).sum() - g.sum() ** 2 / len(g))
        for j in range(X.shape[1]):
            col = X[:, j]
            qs = np.unique(np.quantile(col, [0.2, 0.4, 0.6, 0.8]))
            for t in qs:
                left = col <= t
                nl, nr = int(left.sum()), int((~left).sum())
                if nl < self.min_leaf or nr < self.min_leaf:
                    continue
                gl, gr = g[left], g[~left]
                sse = (gl**2).sum() - gl.sum() ** 2 / nl + (gr**2).sum() - gr.sum() ** 2 / nr
                if best is None or sse < best[0]:
                    best = (float(sse), j, float(t), left)
        if best is None or best[0] >= parent:
            return {"leaf": float(g.mean())}
        _sse, j, t, left = best
        return {
            "j": j,
            "t": t,
            "l": self._build(X[left], g[left], d + 1),
            "r": self._build(X[~left], g[~left], d + 1),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._one(self.node, x) for x in X])

    def _one(self, node: dict, x: np.ndarray) -> float:
        while "leaf" not in node:
            node = node["l"] if x[node["j"]] <= node["t"] else node["r"]
        return node["leaf"]


class GBM:
    def __init__(self, rounds: int, lr: float, depth: int, min_leaf: int) -> None:
        self.rounds, self.lr, self.depth, self.min_leaf = rounds, lr, depth, min_leaf
        self.base = 0.0
        self.trees: list[Tree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> GBM:
        self.base = float(y.mean())
        pred = np.full(len(y), self.base)
        for _ in range(self.rounds):
            tree = Tree(self.depth, self.min_leaf).fit(X, y - pred)
            pred = pred + self.lr * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.full(len(X), self.base)
        for t in self.trees:
            out = out + self.lr * t.predict(X)
        return out


def day_records(idx_by_day: dict[str, list[int]], y: np.ndarray) -> list[dict]:
    out = []
    for day in sorted(idx_by_day):
        r = y[np.array(idx_by_day[day])]
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from bayes_judge import judge

    rows = [json.loads(line) for line in args.paths.read_text().splitlines() if line]
    rows = [r for r in rows if r.get("engine_reward") is not None]
    rows.sort(key=lambda r: (r["day"], r["decide_at"]))
    days = np.array([r["day"] for r in rows])
    y = np.array([r["engine_reward"] for r in rows])
    months = sorted({d[:7] for d in days.tolist()})
    print(f"entries={len(rows)} days={len(set(days.tolist()))} months={months}")
    print(f"pool mean {y.mean() * 1e4:+.0f} bp")

    families = {
        "free5_gbm": FREE,
        "judgment8_gbm": JUDGMENT,
        "all13_gbm": bandit.FEATURES,
    }
    by_day: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_day[r["day"]].append(i)
    base_first = {d: v[:TOP_K] for d, v in by_day.items() if d[:7] != months[0]}

    print("\n=== realized KRW, top-5/day, monthly time-ordered folds")
    print(f"{'family':16s} " + " ".join(f"{m:>10s}" for m in months[1:]) + f" {'pooled':>12s}")
    results = {}
    for name, feats in families.items():
        picked: dict[str, list[int]] = {}
        cells = []
        for m in months[1:]:
            tr = [i for i, r in enumerate(rows) if r["day"][:7] < m]
            ev = [i for i, r in enumerate(rows) if r["day"][:7] == m]
            Xtr = np.array([[rows[i]["features"][f] for f in feats] for i in tr])
            Xev = np.array([[rows[i]["features"][f] for f in feats] for i in ev])
            # rounds chosen on a time-ordered inner split of the training months
            tr_days = sorted({rows[i]["day"] for i in tr})
            cut = tr_days[max(1, int(len(tr_days) * 0.75))]
            inner = [i for i in tr if rows[i]["day"] < cut]
            val = [i for i in tr if rows[i]["day"] >= cut]
            best_r, best_v = 10, None
            for rounds in (10, 30, 60, 120):
                mdl = GBM(rounds, 0.05, 3, 40).fit(
                    np.array([[rows[i]["features"][f] for f in feats] for i in inner]),
                    y[np.array(inner)],
                )
                sc = mdl.predict(np.array([[rows[i]["features"][f] for f in feats] for i in val]))
                k = max(1, len(sc) // 10)
                v = float(y[np.array(val)][np.argsort(-sc)[:k]].mean())
                if best_v is None or v > best_v:
                    best_r, best_v = rounds, v
            mdl = GBM(best_r, 0.05, 3, 40).fit(Xtr, y[np.array(tr)])
            sc = mdl.predict(Xev)
            per: dict[str, list[tuple[float, int]]] = defaultdict(list)
            for pos, i in enumerate(ev):
                per[rows[i]["day"]].append((sc[pos], i))
            chosen = {d: [i for _s, i in sorted(v, reverse=True)[:TOP_K]] for d, v in per.items()}
            picked.update(chosen)
            cells.append(sum(y[np.array(v)].sum() for v in chosen.values()) * NOTIONAL)
        cand = day_records(picked, y)
        base = day_records({d: base_first[d] for d in picked}, y)
        v = judge({c["day"]: c for c in cand}, {b["day"]: b for b in base})
        results[name] = v
        print(
            f"{name:16s} "
            + " ".join(f"{c:>+10,.0f}" for c in cells)
            + f" {sum(c['net'] for c in cand):>+12,.0f}"
        )
    print(
        f"{'BASE live order':16s} "
        + " " * (11 * (len(months) - 1))
        + f" {sum(b['net'] for b in day_records(base_first, y)):>+12,.0f}"
    )
    print()
    for name, v in results.items():
        print(
            f"{name:16s} p(mu>0)={v['p_mu_gt_0']:.3f} mu={v['mu_mean']:+,.0f}"
            f" days>=0={v['days_ge_0']}/{v['n_days']}"
            f" worst {v['worst_day_candidate']:+,.0f} vs {v['worst_day_baseline']:+,.0f}"
            f" losers {v['loser_ratio_candidate']:.3f} vs {v['loser_ratio_baseline']:.3f}"
            f" adopt={v['adopt']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, default=str))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
