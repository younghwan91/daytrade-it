"""An upper bound on what any prompt over the same text could extract.

A prompt is a lossy function of the text it is shown. The live judge sees the
title plus Toss's ~100-char summary and nothing else, so a model fitted
*directly* on that text bounds what any set of prompt questions over it could
recover -- with one honest exception noted at the end.

Character n-grams (2-5) are used rather than words: Korean headlines are
agglutinative and unsegmented, and character n-grams pick up the morphology a
word tokeniser would miss. The model is ridge on tf-idf, fitted per
time-ordered monthly fold, and it also gets the free deterministic attributes,
so it is strictly better informed than any prompt.

Reported against the same three references as everything else: the pool, a
random 5/day, and the day's realised best 5.
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

sys.path.insert(0, "scripts/eval")
from prompt_ceiling import COST, D  # noqa: E402


def main() -> None:
    master = {}
    with (D / "master.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            master[r["key"]] = r
    texts: list[str] = []
    extra: list[list[float]] = []
    y: list[float] = []
    days: list[str] = []
    months: list[str] = []
    mfe: list[float] = []
    with (D / "labels.jsonl").open() as f:
        for line in f:
            lb = json.loads(line)
            m = master[lb["key"]]
            texts.append(f"{m['title']} || {m.get('summary') or ''}")
            rn = float(m.get("related_n") or 1)
            extra.append(
                [
                    math.log1p(rn),
                    1.0 if lb["act_min"] == 0 else 0.0,
                    min(lb["act_min"], 380) / 380.0,
                    float(lb.get("pre_move") or 0.0),
                    abs(float(lb.get("pre_move") or 0.0)),
                    math.log1p(float(lb.get("turnover") or 0.0)) / 30.0,
                    1.0 if m.get("market") == "코스닥" else 0.0,
                ]
            )
            y.append(lb["ret_close"] - COST)
            mfe.append(lb["mfe"])
            days.append(m["day"])
            months.append(m["month"])
    y_a = np.array(y)
    mfe_a = np.array(mfe)
    days_a = np.array(days)
    months_a = np.array(months)
    extra_a = np.array(extra)
    mons = sorted(set(months))
    print(f"{len(y)} pairs, {len(set(days))} days, pool {y_a.mean() * 1e4:+.0f} bp")
    n_titles = len({t.split(" || ")[0] for t in texts})
    print(f"distinct titles: {n_titles}")

    def run(target: np.ndarray, name: str) -> None:
        print(f"\n=== target: {name} ===")
        print(
            f"{'month':9s} {'train':>7s} {'feats':>8s} {'top5 bp':>8s} {'rand5':>7s} {'oracle':>7s}"
        )
        T5: list[float] = []
        R5: list[float] = []
        OR: list[float] = []
        per: list[float] = []
        for i, mo in enumerate(mons):
            if i == 0:
                continue
            tr = np.isin(months_a, mons[:i])
            te = months_a == mo
            if tr.sum() < 3000:
                continue
            vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                min_df=20,
                max_features=300_000,
                sublinear_tf=True,
            )
            Xtr = sparse.hstack(
                [vec.fit_transform([texts[j] for j in np.flatnonzero(tr)]), extra_a[tr]],
                format="csr",
            )
            Xte = sparse.hstack(
                [vec.transform([texts[j] for j in np.flatnonzero(te)]), extra_a[te]],
                format="csr",
            )
            mdl = Ridge(alpha=50.0, solver="sparse_cg", max_iter=2000)
            mdl.fit(Xtr, target[tr])
            p = mdl.predict(Xte)
            yt = y_a[te]
            dt = days_a[te]
            bd: dict[str, list] = defaultdict(list)
            for pp, yy, dd in zip(p, yt, dt, strict=True):
                bd[dd].append((pp, yy))
            t5 = [yy for ds in bd.values() for _, yy in sorted(ds, key=lambda z: -z[0])[:5]]
            r5 = [st.mean([yy for _, yy in ds]) for ds in bd.values()]
            orc = [yy for ds in bd.values() for _, yy in sorted(ds, key=lambda z: -z[1])[:5]]
            T5 += t5
            R5 += r5
            OR += orc
            per.append(st.mean(t5))
            print(
                f"{mo:9s} {tr.sum():7d} {Xtr.shape[1]:8d} {st.mean(t5) * 1e4:+8.0f} "
                f"{st.mean(r5) * 1e4:+7.0f} {st.mean(orc) * 1e4:+7.0f}"
            )
        print("-" * 52)
        print(
            f"{'POOLED':9s} {'':7s} {'':8s} {st.mean(T5) * 1e4:+8.0f} "
            f"{st.mean(R5) * 1e4:+7.0f} {st.mean(OR) * 1e4:+7.0f}"
        )
        print(f"months with top5 > 0: {sum(1 for v in per if v > 0)}/{len(per)}")
        print(
            "share of the oracle gap recovered: "
            f"{(st.mean(T5) - st.mean(R5)) / (st.mean(OR) - st.mean(R5)):.1%}"
        )

    run(y_a, "net return (direction)")
    run(mfe_a, "max favourable excursion (travel distance)")


if __name__ == "__main__":
    main()
