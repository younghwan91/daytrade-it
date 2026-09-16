"""Does v2's score separate winners from losers anywhere?

Separation, not calibration: spearman(score, realised net return) and decile
lift, over the whole pool and inside every subpopulation a prompt could be
scoped to. Also prices what the live exit rules cost against simply holding.
"""

# Analysis script: the predicate lambdas share one signature on purpose.
# ruff: noqa: ARG005

from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "scripts/eval")
from prompt_ceiling import COST, D, title_tag  # noqa: E402
from prompt_oracle_mine import spearman  # noqa: E402

RL = Path("/home/young/git/daytrade-it-aplan/data/eval/bandit_a7prep/dataset.jsonl")
ERRORS = Path("data/eval/cache/universe_v2_errors.jsonl")


def decile_lift(pairs: list[tuple[float, float]]) -> float | None:
    """Top-decile mean minus bottom-decile mean, in bp, ordered by the score."""
    if len(pairs) < 100:
        return None
    s = sorted(pairs, key=lambda p: p[0])
    k = len(s) // 10
    return (st.mean([r for _, r in s[-k:]]) - st.mean([r for _, r in s[:k]])) * 1e4


def load() -> list[tuple[dict, dict, float]]:
    master = {}
    with (D / "master.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            master[r["key"]] = r
    rows = []
    with (D / "labels.jsonl").open() as f:
        for line in f:
            lb = json.loads(line)
            rows.append((master[lb["key"]], lb, lb["ret_close"] - COST))
    return rows


def exit_cost(rows: list[tuple[dict, dict, float]]) -> None:
    print("=== exit rules vs simply holding, on the rows both labels cover ===")
    rl = {}
    with RL.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("reward") is not None:
                rl[r["key"]] = r["reward"]
    held = {m["key"]: rt for m, _, rt in rows}
    ov = [(rl[k], held[k]) for k in rl if k in held]
    live = st.mean([a for a, _ in ov])
    hold = st.mean([b for _, b in ov])
    print(
        f"  n={len(ov)}   live exit rules {live * 1e4:+.0f} bp   "
        f"hold to close {hold * 1e4:+.0f} bp   cost of the exit {(live - hold) * 1e4:+.0f} bp"
    )
    print(
        f"  hold win rate {sum(1 for _, b in ov if b > 0) / len(ov):.1%}   "
        f"exit win rate {sum(1 for a, _ in ov if a > 0) / len(ov):.1%}"
    )


def main() -> None:
    rows = load()
    exit_cost(rows)

    print("\n=== v2 score separation, whole pool ===")
    sc = [(m["v2"].get("score") or 0.0, rt) for m, _, rt in rows]
    print(
        f"  n={len(sc)}  spearman {spearman([a for a, _ in sc], [b for _, b in sc]):+.4f}"
        f"  decile lift {decile_lift(sc):+.0f} bp"
    )
    nz = [p for p in sc if p[0] != 0]
    print(
        f"  non-zero only n={len(nz)}  "
        f"spearman {spearman([a for a, _ in nz], [b for _, b in nz]):+.4f}"
        f"  decile lift {decile_lift(nz):+.0f} bp"
    )

    print("\n=== score buckets ===")
    pool = st.mean([rt for *_, rt in rows])
    edges = [
        (-10.0, -0.15, "<= -0.15"),
        (-0.15, -1e-9, "-0.15..0"),
        (-1e-9, 1e-9, "exactly 0"),
        (1e-9, 0.15, "0..0.15"),
        (0.15, 0.3, "0.15..0.3"),
        (0.3, 0.5, "0.3..0.5"),
        (0.5, 10.0, "> 0.5"),
    ]
    for lo, hi, name in edges:
        v = [rt for m, _, rt in rows if lo < (m["v2"].get("score") or 0.0) <= hi]
        if len(v) < 50:
            continue
        print(
            f"  {name:12s} n={len(v):6d}  {st.mean(v) * 1e4:+6.0f} bp  "
            f"vs pool {(st.mean(v) - pool) * 1e4:+6.0f}  "
            f"win {sum(1 for x in v if x > 0) / len(v):5.1%}"
        )

    print("\n=== separation INSIDE each subpopulation (where could a prompt still work?) ===")
    print(f"{'subpopulation':34s} {'n':>7s} {'rho':>8s} {'decile bp':>10s} {'mean bp':>8s}")
    subs = {
        "fan-out == 1": lambda m, lb: (m.get("related_n") or 1) == 1,
        "fan-out <= 5": lambda m, lb: (m.get("related_n") or 1) <= 5,
        "fan-out > 15": lambda m, lb: (m.get("related_n") or 1) > 15,
        "subject == main": lambda m, lb: m["v2"].get("subject") == "main",
        "main & fan-out <= 5": lambda m, lb: (
            m["v2"].get("subject") == "main" and (m.get("related_n") or 1) <= 5
        ),
        "no bracket tag": lambda m, lb: title_tag(m["title"]) == "(none)",
        "premarket": lambda m, lb: lb["act_min"] == 0,
        "intraday": lambda m, lb: lb["act_min"] > 0,
        "freshness == new": lambda m, lb: m["v2"].get("freshness") == "new",
        "not a price-move report": lambda m, lb: not m["v2"].get("reports_price_move"),
        "pre_move <= +2%": lambda m, lb: (lb.get("pre_move") or 0) <= 0.02,
        "거래소": lambda m, lb: m.get("market") == "거래소",
        "코스닥": lambda m, lb: m.get("market") == "코스닥",
        "material 실적/수주/신약": lambda m, lb: (
            m["v2"].get("material_type") in ("실적", "수주", "신약")
        ),
        "material 테마": lambda m, lb: m["v2"].get("material_type") == "테마",
        "turnover > 1e11": lambda m, lb: (lb.get("turnover") or 0) > 1e11,
    }
    for name, pred in subs.items():
        v = [(m["v2"].get("score") or 0.0, rt) for m, lb, rt in rows if pred(m, lb)]
        if len(v) < 300:
            continue
        dl = decile_lift(v) or 0.0
        print(
            f"  {name:32s} {len(v):7d} "
            f"{spearman([a for a, _ in v], [b for _, b in v]):+8.4f}"
            f" {dl:+10.0f} {st.mean([b for _, b in v]) * 1e4:+8.0f}"
        )

    print("\n=== per-month separation (is it ever stable?) ===")
    bym: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for m, _, rt in rows:
        bym[m["month"]].append((m["v2"].get("score") or 0.0, rt))
    for mo in sorted(bym):
        v = bym[mo]
        print(
            f"  {mo}  n={len(v):6d}  "
            f"rho {spearman([a for a, _ in v], [b for _, b in v]):+.4f}"
            f"  decile {decile_lift(v):+6.0f} bp"
        )

    print("\n=== schema rejections: what is the judge refusing to answer? ===")
    reasons: Counter[str] = Counter()
    details: Counter[str] = Counter()
    with ERRORS.open() as f:
        for line in f:
            r = json.loads(line)
            reasons[str(r.get("reason"))] += 1
            if r.get("reason") == "schema":
                details[str(r.get("detail"))] += 1
    print(f"  {reasons.most_common()}")
    for k, v in details.most_common(14):
        print(f"    {v:6d}  {k}")


if __name__ == "__main__":
    main()
