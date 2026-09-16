"""Mechanism check: the model predicts travel distance, not direction.

1. Which attributes drive the models -- the LLM's fields, or the free ones?
2. Does the direction model beat a deterministic-only model? (does the prompt add anything?)
3. If travel distance is predictable, can a target exit harvest it? (live-executable)
"""

# Analysis script: the group/predicate lambdas share one signature on purpose.
# ruff: noqa: ARG005

from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "scripts/eval")
from prompt_ceiling import (  # noqa: E402
    COST,
    FRESH,
    MATERIALS,
    PERSIST,
    SCALE,
    SUBJECTS,
    D,
    ridge,
    title_tag,
)


def blocks(
    m: dict, lb: dict, tags: list[str], sectors: list[str]
) -> tuple[list[float], list[float]]:
    """(what the LLM answered, what is free and deterministic)."""
    v2 = m["v2"]
    llm = (
        [1.0 if v2.get("subject") == s else 0.0 for s in SUBJECTS]
        + [1.0 if v2.get("material_type") == s else 0.0 for s in MATERIALS]
        + [1.0 if v2.get("persistence") == s else 0.0 for s in PERSIST]
        + [1.0 if v2.get("freshness") == s else 0.0 for s in FRESH]
        + [1.0 if v2.get("scale_vs_size") == s else 0.0 for s in SCALE]
        + [
            float(v2.get("specificity") or 0),
            float(v2.get("surprise") or 0),
            float(v2.get("sentiment_direction") or 0),
            float(v2.get("sentiment_strength") or 0),
            1.0 if v2.get("reports_price_move") else 0.0,
            float(v2.get("score") or 0.0),
        ]
    )
    rn = float(m.get("related_n") or 1)
    det = (
        [math.log1p(rn), 1.0 if rn == 1 else 0.0, 1.0 if rn > 15 else 0.0]
        + [1.0 if title_tag(m["title"]) == t else 0.0 for t in tags]
        + [1.0 if str(m.get("sector")) == s else 0.0 for s in sectors]
        + [
            1.0 if m.get("market") == "코스닥" else 0.0,
            1.0 if lb["act_min"] == 0 else 0.0,
            min(lb["act_min"], 380) / 380.0,
            float(lb.get("pre_move") or 0.0),
            abs(float(lb.get("pre_move") or 0.0)),
            math.log1p(float(lb.get("turnover") or 0.0)) / 30.0,
            float(len(m["title"])) / 60.0,
        ]
    )
    return llm, det


def main() -> None:
    master = {}
    with (D / "master.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            master[r["key"]] = r
    rows = []
    with (D / "labels.jsonl").open() as f:
        for line in f:
            lb = json.loads(line)
            rows.append((master[lb["key"]], lb))

    tags = [t for t, n in Counter(title_tag(m["title"]) for m, _ in rows).items() if n >= 200]
    sectors = [s for s, n in Counter(str(m.get("sector")) for m, _ in rows).items() if n >= 500]
    months = sorted({m["month"] for m, _ in rows})
    bym: dict[str, list] = defaultdict(list)
    for r in rows:
        bym[r[0]["month"]].append(r)

    def fold_top5(target, use_llm: bool, use_det: bool, label: str) -> None:
        def fx(m: dict, lb: dict) -> list[float]:
            llm, det = blocks(m, lb, tags, sectors)
            return (llm if use_llm else []) + (det if use_det else [])

        t5r: list[float] = []
        t5m: list[float] = []
        per: list[float] = []
        for i, mo in enumerate(months):
            if i == 0:
                continue
            train = [r for mm in months[:i] for r in bym[mm]]
            test = bym[mo]
            if len(train) < 3000:
                continue
            w = ridge(
                [fx(m, lb) for m, lb in train],
                [target(lb) for _, lb in train],
                lam=len(train) * 1e-3,
            )
            days: dict[str, list] = defaultdict(list)
            for m, lb in test:
                x = fx(m, lb)
                p = sum(a * b for a, b in zip(x, w[:-1], strict=True)) + w[-1]
                days[m["day"]].append((p, lb))
            sel = [lb for ds in days.values() for _, lb in sorted(ds, key=lambda z: -z[0])[:5]]
            per.append(st.mean([lb["ret_close"] - COST for lb in sel]))
            t5r += [lb["ret_close"] - COST for lb in sel]
            t5m += [lb["mfe"] for lb in sel]
        print(
            f"{label:34s} top5 ret {st.mean(t5r) * 1e4:+7.0f}  "
            f"top5 mfe {st.mean(t5m) * 1e4:+7.0f}  "
            f"months>0 {sum(1 for p in per if p > 0)}/{len(per)}"
        )

    print("=== 2. does the LLM add anything a free attribute does not? (target = return) ===")
    pool = st.mean([lb["ret_close"] - COST for _, lb in rows])
    print(f"{'pool':34s} top5 ret {pool * 1e4:+7.0f}")
    fold_top5(lambda lb: lb["ret_close"] - COST, True, True, "LLM + deterministic")
    fold_top5(lambda lb: lb["ret_close"] - COST, False, True, "deterministic only (no LLM)")
    fold_top5(lambda lb: lb["ret_close"] - COST, True, False, "LLM only")

    print("\n=== 1. the same for travel distance (target = MFE) ===")
    fold_top5(lambda lb: lb["mfe"], True, True, "LLM + deterministic")
    fold_top5(lambda lb: lb["mfe"], False, True, "deterministic only (no LLM)")
    fold_top5(lambda lb: lb["mfe"], True, False, "LLM only")

    print("\n=== 3. can a target exit harvest the predictable travel? ===")
    print("exit = the target if the day's high after entry reaches it, else the close.")
    print("(optimistic: it assumes the limit fills whenever the high touches it.)")
    for tgt in (0.02, 0.03, 0.05):

        def payoff(lb: dict, t: float = tgt) -> float:
            return (t if lb["mfe"] >= t else lb["ret_close"]) - COST

        hit = sum(1 for _, lb in rows if lb["mfe"] >= tgt) / len(rows)
        allp = st.mean([payoff(lb) for _, lb in rows])
        print(f"\n  target +{tgt:.0%}: whole pool {allp * 1e4:+.0f} bp (hit rate {hit:.1%})")
        t5: list[float] = []
        per: list[tuple[str, float]] = []
        for i, mo in enumerate(months):
            if i == 0:
                continue
            train = [r for mm in months[:i] for r in bym[mm]]
            test = bym[mo]
            if len(train) < 3000:
                continue

            def fx(m: dict, lb: dict) -> list[float]:
                llm, det = blocks(m, lb, tags, sectors)
                return llm + det

            w = ridge(
                [fx(m, lb) for m, lb in train],
                [payoff(lb) for _, lb in train],
                lam=len(train) * 1e-3,
            )
            days: dict[str, list] = defaultdict(list)
            for m, lb in test:
                x = fx(m, lb)
                p = sum(a * b for a, b in zip(x, w[:-1], strict=True)) + w[-1]
                days[m["day"]].append((p, lb))
            sel = [lb for ds in days.values() for _, lb in sorted(ds, key=lambda z: -z[0])[:5]]
            per.append((mo, st.mean([payoff(lb) for lb in sel])))
            t5 += [payoff(lb) for lb in sel]
        print(
            "    top5/day out of sample: " + "  ".join(f"{mo[-2:]} {v * 1e4:+.0f}" for mo, v in per)
        )
        print(
            f"    pooled {st.mean(t5) * 1e4:+.0f} bp  "
            f"months>0 {sum(1 for _, v in per if v > 0)}/{len(per)}"
        )


if __name__ == "__main__":
    main()
