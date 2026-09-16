"""Closing the nonlinear gap: gradient boosting over every decision-time attribute.

rl-builder's negative was explicitly about the linear/monotone family over 13
features. This widens it as far as the data allows: HistGradientBoosting over
the 10 v2 fields PLUS the six attributes no prompt is told, on 132,185 pairs
(29x rl-builder's 4,496), with honest time-ordered monthly folds.

If this cannot beat a random pick out of sample, the negative is not about model
class, feature count, or sample size.
"""

# Analysis script: the group/predicate lambdas share one signature on purpose.
# ruff: noqa: ARG005

from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "scripts/eval")
from prompt_ceiling import COST, D, title_tag

CAT = [
    "subject",
    "material_type",
    "persistence",
    "freshness",
    "scale_vs_size",
    "title_tag",
    "sector",
    "market",
]
NUM = [
    "specificity",
    "surprise",
    "sentiment_direction",
    "sentiment_strength",
    "reports_price_move",
    "score",
    "log_fanout",
    "act_min",
    "premarket",
    "pre_move",
    "abs_pre_move",
    "log_turnover",
    "title_len",
]


def build(rows, tags, sectors):
    codes = {c: {} for c in CAT}
    X = np.zeros((len(rows), len(CAT) + len(NUM)), dtype=np.float64)
    for i, (m, lab) in enumerate(rows):
        v2 = m["v2"]
        rn = float(m.get("related_n") or 1)
        vals = [
            v2.get("subject"),
            v2.get("material_type"),
            v2.get("persistence"),
            v2.get("freshness"),
            v2.get("scale_vs_size"),
            title_tag(m["title"]) if title_tag(m["title"]) in tags else "(rare)",
            str(m.get("sector")) if str(m.get("sector")) in sectors else "(rare)",
            m.get("market"),
        ]
        for j, v in enumerate(vals):
            d = codes[CAT[j]]
            if v not in d:
                d[v] = len(d)
            X[i, j] = d[v]
        X[i, len(CAT) :] = [
            float(v2.get("specificity") or 0),
            float(v2.get("surprise") or 0),
            float(v2.get("sentiment_direction") or 0),
            float(v2.get("sentiment_strength") or 0),
            1.0 if v2.get("reports_price_move") else 0.0,
            float(v2.get("score") or 0.0),
            math.log1p(rn),
            float(lab["act_min"]),
            1.0 if lab["act_min"] == 0 else 0.0,
            float(lab.get("pre_move") or 0.0),
            abs(float(lab.get("pre_move") or 0.0)),
            math.log1p(float(lab.get("turnover") or 0.0)),
            float(len(m["title"])),
        ]
    return X, codes


def main():
    master = {json.loads(ln)["key"]: json.loads(ln) for ln in (D / "master.jsonl").open()}
    rows = []
    y = []
    days = []
    months = []
    for line in (D / "labels.jsonl").open():
        lab = json.loads(line)
        m = master[lab["key"]]
        rows.append((m, lab))
        y.append(lab["ret_close"] - COST)
        days.append(m["day"])
        months.append(m["month"])
    tags = {t for t, n in Counter(title_tag(m["title"]) for m, _ in rows).items() if n >= 200}
    sectors = {s for s, n in Counter(str(m.get("sector")) for m, _ in rows).items() if n >= 500}
    X, _ = build(rows, tags, sectors)
    y = np.array(y)
    days = np.array(days)
    months = np.array(months)
    mons = sorted(set(months))
    cat_mask = [True] * len(CAT) + [False] * len(NUM)
    print(
        f"{len(y)} pairs, {len(set(days))} days, {X.shape[1]} features "
        f"({len(CAT)} categorical), pool {y.mean() * 1e4:+.0f} bp"
    )
    print(
        f"\n{'month':9s} {'train':>7s} {'top5 bp':>8s} {'rand5':>7s} {'gate n':>7s} {'gate bp':>8s} {'oracle':>7s}"
    )
    T5 = []
    R5 = []
    OR = []
    GA = []
    per = []
    for i, mo in enumerate(mons):
        if i == 0:
            continue
        tr = np.isin(months, mons[:i])
        te = months == mo
        if tr.sum() < 3000:
            continue
        g = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=200,
            l2_regularization=1.0,
            categorical_features=cat_mask,
            random_state=0,
            early_stopping=True,
            validation_fraction=0.15,
        )
        g.fit(X[tr], y[tr])
        p = g.predict(X[te])
        yt = y[te]
        dt = days[te]
        bd = defaultdict(list)
        for pp, yy, dd in zip(p, yt, dt, strict=False):
            bd[dd].append((pp, yy))
        t5 = [yy for ds in bd.values() for _, yy in sorted(ds, key=lambda z: -z[0])[:5]]
        r5 = [st.mean([yy for _, yy in ds]) for ds in bd.values()]
        orc = [yy for ds in bd.values() for _, yy in sorted(ds, key=lambda z: -z[1])[:5]]
        gate = yt[p > 0]
        T5 += t5
        R5 += r5
        OR += orc
        GA += list(gate)
        per.append(st.mean(t5))
        print(
            f"{mo:9s} {tr.sum():7d} {st.mean(t5) * 1e4:+8.0f} {st.mean(r5) * 1e4:+7.0f} "
            f"{len(gate):7d} {(gate.mean() * 1e4 if len(gate) else float('nan')):+8.0f} "
            f"{st.mean(orc) * 1e4:+7.0f}"
        )
    print("-" * 62)
    print(
        f"{'POOLED':9s} {'':7s} {st.mean(T5) * 1e4:+8.0f} {st.mean(R5) * 1e4:+7.0f} "
        f"{len(GA):7d} {st.mean(GA) * 1e4:+8.0f} {st.mean(OR) * 1e4:+7.0f}"
    )
    print(f"months with top5 > 0: {sum(1 for v in per if v > 0)}/{len(per)}")
    print(
        f"share of the oracle gap recovered: "
        f"{(st.mean(T5) - st.mean(R5)) / (st.mean(OR) - st.mean(R5)):.1%}"
    )

    # permutation importance of the LLM block vs the free block, on the last fold
    print("\n=== does the LLM block carry anything the free attributes do not? ===")
    tr = np.isin(months, mons[:-1])
    te = months == mons[-1]
    g = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=200,
        l2_regularization=1.0,
        categorical_features=cat_mask,
        random_state=0,
        early_stopping=True,
        validation_fraction=0.15,
    ).fit(X[tr], y[tr])
    llm_idx = [0, 1, 2, 3, 4] + [len(CAT) + k for k in range(6)]
    free_idx = [j for j in range(X.shape[1]) if j not in llm_idx]
    rng = np.random.default_rng(0)

    def top5_of(Xe):
        p = g.predict(Xe)
        bd = defaultdict(list)
        for pp, yy, dd in zip(p, y[te], days[te], strict=False):
            bd[dd].append((pp, yy))
        return st.mean([yy for ds in bd.values() for _, yy in sorted(ds, key=lambda z: -z[0])[:5]])

    base = top5_of(X[te])
    for name, idx in (("LLM fields shuffled", llm_idx), ("free attributes shuffled", free_idx)):
        vals = []
        for _ in range(5):
            Xs = X[te].copy()
            for j in idx:
                Xs[:, j] = rng.permutation(Xs[:, j])
            vals.append(top5_of(Xs))
        print(f"  {name:28s} top5 {st.mean(vals) * 1e4:+7.0f} bp  (intact {base * 1e4:+.0f})")


if __name__ == "__main__":
    main()
