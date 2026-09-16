"""The decisive Step-1 test: is there a prompt worth writing at all?

The brief's premise is that the prompt is the lever because the features come
from the prompt. That is only true if *some* decision-time description of a pair
separates winners from losers. This asks the question in the most generous form
available: fit an honest, time-ordered model over **every** decision-time
attribute -- the ten v2 fields the prompt asks for, plus six the prompt is not
even told (fan-out, title column tag, sector, market, session/acting minute,
pre-article move, that day's turnover) -- and see what it earns out of sample.

If a model that already knows more than any prompt could tell it cannot beat the
pool, then no rewrite of the prompt's questions can either, and the oracle gap is
not a locatable signal.

Baselines at the same trade count, as rl_folds.py uses:
  pool    every pair's mean
  rand5   5 x the day's mean (the expected value of a random pick)
  oracle  the day's realised best 5 (perfect hindsight)

Folds are by month: month m is scored by a model trained only on rows strictly
before it. No API, no DB.
"""

# Analysis script: the group/predicate lambdas share one signature on purpose.
# ruff: noqa: ARG005

from __future__ import annotations

import json
import math
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

D = Path("data/eval/prompt_v3")
COST = 0.00215
BRACKET = re.compile(r"^\s*\[([^\]]{1,20})\]")
DIGITS = re.compile(r"\d")

SUBJECTS = ("main", "partial", "mention")
MATERIALS = ("수주", "실적", "신약", "증설", "정책", "테마", "루머", "지분", "기타")
PERSIST = ("one_off", "multi_quarter", "structural")
FRESH = ("repeat", "follow_up", "new")
SCALE = ("unknown", "small", "medium", "large")


def title_tag(title: str) -> str:
    m = BRACKET.match(title)
    return f"[{DIGITS.sub('N', m.group(1))}]" if m else "(none)"


def featurise(m: dict, lab: dict, tags: list[str], sectors: list[str]) -> list[float]:
    v2 = m["v2"]
    f: list[float] = []
    # --- what the prompt asks today ---
    f += [1.0 if v2.get("subject") == s else 0.0 for s in SUBJECTS]
    f += [1.0 if v2.get("material_type") == s else 0.0 for s in MATERIALS]
    f += [1.0 if v2.get("persistence") == s else 0.0 for s in PERSIST]
    f += [1.0 if v2.get("freshness") == s else 0.0 for s in FRESH]
    f += [1.0 if v2.get("scale_vs_size") == s else 0.0 for s in SCALE]
    f += [
        float(v2.get("specificity") or 0),
        float(v2.get("surprise") or 0),
        float(v2.get("sentiment_direction") or 0),
        float(v2.get("sentiment_strength") or 0),
        1.0 if v2.get("reports_price_move") else 0.0,
        float(v2.get("score") or 0.0),
    ]
    # --- what the prompt is NOT told (all free, all deterministic) ---
    rn = float(m.get("related_n") or 1)
    f += [math.log1p(rn), 1.0 if rn == 1 else 0.0, 1.0 if rn > 15 else 0.0]
    f += [1.0 if title_tag(m["title"]) == t else 0.0 for t in tags]
    f += [1.0 if str(m.get("sector")) == s else 0.0 for s in sectors]
    f += [1.0 if m.get("market") == "코스닥" else 0.0]
    f += [
        1.0 if lab["act_min"] == 0 else 0.0,
        min(lab["act_min"], 380) / 380.0,
        float(lab.get("pre_move") or 0.0),
        abs(float(lab.get("pre_move") or 0.0)),
        math.log1p(float(lab.get("turnover") or 0.0)) / 30.0,
        float(len(m["title"])) / 60.0,
    ]
    return f


def ridge(X: list[list[float]], y: list[float], lam: float) -> list[float]:
    """Closed-form ridge with an intercept (features are few; n is large)."""
    p = len(X[0])
    xtx = [[0.0] * (p + 1) for _ in range(p + 1)]
    xty = [0.0] * (p + 1)
    for row, t in zip(X, y, strict=True):
        r = [*row, 1.0]
        for i in range(p + 1):
            ri = r[i]
            if ri == 0.0:
                continue
            xty[i] += ri * t
            xr = xtx[i]
            for j in range(p + 1):
                xr[j] += ri * r[j]
    for i in range(p):  # never penalise the intercept
        xtx[i][i] += lam
    # Gaussian elimination
    n = p + 1
    aug = [[*xtx[i], xty[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(aug[r][c]))
        if abs(aug[piv][c]) < 1e-12:
            continue
        aug[c], aug[piv] = aug[piv], aug[c]
        pv = aug[c][c]
        for j in range(c, n + 1):
            aug[c][j] /= pv
        for r in range(n):
            if r == c or aug[r][c] == 0.0:
                continue
            fac = aug[r][c]
            for j in range(c, n + 1):
                aug[r][j] -= fac * aug[c][j]
    return [aug[i][n] for i in range(n)]


def main() -> None:
    with (D / "master.jsonl").open() as f:
        master = {json.loads(ln)["key"]: json.loads(ln) for ln in f}
    rows = []
    for line in (D / "labels.jsonl").open():
        lab = json.loads(line)
        m = master[lab["key"]]
        rows.append((m, lab, lab["ret_close"] - COST))
    print(f"{len(rows)} pairs, {len({m['day'] for m, _, _ in rows})} days")

    tags = [t for t, n in Counter(title_tag(m["title"]) for m, _, _ in rows).items() if n >= 200]
    sectors = [s for s, n in Counter(str(m.get("sector")) for m, _, _ in rows).items() if n >= 500]
    print(f"title tags kept {len(tags)}, sectors kept {len(sectors)}")

    by_month: dict[str, list] = defaultdict(list)
    for r in rows:
        by_month[r[0]["month"]].append(r)
    months = sorted(by_month)

    print("\nper-month pool mean (bp) and the hindsight ceiling:")
    print(f"{'month':9s} {'n':>7s} {'pool':>7s} {'rand5':>7s} {'oracle5':>8s}")
    for mo in months:
        rs = by_month[mo]
        days = defaultdict(list)
        for r in rs:
            days[r[0]["day"]].append(r)
        orc = [x[2] for ds in days.values() for x in sorted(ds, key=lambda z: -z[2])[:5]]
        print(
            f"{mo:9s} {len(rs):7d} {st.mean([r[2] for r in rs]) * 1e4:+7.0f} "
            f"{st.mean([r[2] for r in rs]) * 1e4:+7.0f} {st.mean(orc) * 1e4:+8.0f}"
        )

    print("\n" + "=" * 74)
    print("HONEST TIME-ORDERED FOLDS -- train on every earlier month, score month m")
    print("=" * 74)
    print(
        f"{'month':9s} {'train n':>8s} {'gate n':>7s} {'gate bp':>8s} {'pool bp':>8s} "
        f"{'top5 bp':>8s} {'rand5':>7s} {'oracle':>7s}"
    )
    all_top5: list[float] = []
    all_rand5: list[float] = []
    all_oracle: list[float] = []
    all_gate: list[float] = []
    for i, mo in enumerate(months):
        if i == 0:
            continue
        train = [r for mm in months[:i] for r in by_month[mm]]
        test = by_month[mo]
        if len(train) < 3000:
            continue
        Xtr = [featurise(m, lab, tags, sectors) for m, lab, _ in train]
        ytr = [r[2] for r in train]
        w = ridge(Xtr, ytr, lam=len(train) * 1e-3)
        pred = []
        for m, lab, r in test:
            x = featurise(m, lab, tags, sectors)
            pred.append((sum(a * b for a, b in zip(x, w[:-1], strict=True)) + w[-1], r))
        pool = st.mean([p[1] for p in pred])
        gate = [r for p, r in pred if p > 0]
        days: dict[str, list] = defaultdict(list)
        for (m, _lab, _r), (p, _) in zip(test, pred, strict=True):
            days[m["day"]].append((p, r))
        top5 = [r for ds in days.values() for _, r in sorted(ds, key=lambda z: -z[0])[:5]]
        rand5 = [st.mean([r for _, r in ds]) for ds in days.values()]
        oracle = [r for ds in days.values() for _, r in sorted(ds, key=lambda z: -z[1])[:5]]
        all_top5 += top5
        all_rand5 += rand5
        all_oracle += oracle
        all_gate += gate
        print(
            f"{mo:9s} {len(train):8d} {len(gate):7d} "
            f"{(st.mean(gate) * 1e4 if gate else float('nan')):+8.0f} {pool * 1e4:+8.0f} "
            f"{st.mean(top5) * 1e4:+8.0f} {st.mean(rand5) * 1e4:+7.0f} {st.mean(oracle) * 1e4:+7.0f}"
        )

    print("-" * 74)
    print(
        f"{'POOLED':9s} {'':8s} {len(all_gate):7d} {st.mean(all_gate) * 1e4:+8.0f} "
        f"{st.mean([r[2] for r in rows]) * 1e4:+8.0f} "
        f"{st.mean(all_top5) * 1e4:+8.0f} {st.mean(all_rand5) * 1e4:+7.0f} "
        f"{st.mean(all_oracle) * 1e4:+7.0f}"
    )

    # ---- the same, but predicting MFE instead: is the tail reachable by an exit? ----
    print("\n" + "=" * 74)
    print("SAME MODEL, TARGET = MFE (can an exit rule harvest what entry cannot?)")
    print("=" * 74)
    mfe_rows = [(m, lab, lab["mfe"]) for m, lab, _ in rows]
    by_month_m: dict[str, list] = defaultdict(list)
    for r in mfe_rows:
        by_month_m[r[0]["month"]].append(r)
    print(f"{'month':9s} {'top5 mfe':>9s} {'pool mfe':>9s} {'top5 ret':>9s}")
    for i, mo in enumerate(months):
        if i == 0:
            continue
        train = [r for mm in months[:i] for r in by_month_m[mm]]
        test = by_month_m[mo]
        if len(train) < 3000:
            continue
        Xtr = [featurise(m, lab, tags, sectors) for m, lab, _ in train]
        w = ridge(Xtr, [r[2] for r in train], lam=len(train) * 1e-3)
        days: dict[str, list] = defaultdict(list)
        for m, lab, mf in test:
            x = featurise(m, lab, tags, sectors)
            p = sum(a * b for a, b in zip(x, w[:-1], strict=True)) + w[-1]
            days[m["day"]].append((p, mf, lab["ret_close"] - COST))
        t5 = [(mf, rt) for ds in days.values() for _, mf, rt in sorted(ds, key=lambda z: -z[0])[:5]]
        print(
            f"{mo:9s} {st.mean([a for a, _ in t5]) * 1e4:+9.0f} "
            f"{st.mean([r[2] for r in test]) * 1e4:+9.0f} "
            f"{st.mean([b for _, b in t5]) * 1e4:+9.0f}"
        )


if __name__ == "__main__":
    main()
