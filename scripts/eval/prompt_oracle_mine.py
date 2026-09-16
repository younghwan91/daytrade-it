"""Step 1: what separates the oracle's picks from the rest.

Two label spaces, deliberately:

A) the **policy reward** on rl-builder's 4,496-row dataset -- the exact space the
   +245..+357 bp oracle gap was measured in. Small, but it is the gap in question.
B) the **wide label** on all 132,185 judged pairs (hold from the decision minute to
   the close, market-excess). The prompt runs on all of them, so this is the space a
   prompt change actually moves. (B) is only usable if it tracks (A); the first
   section measures that on the overlap.

No API, no DB, no look-ahead: every attribute below is known at `decide_at`.
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
RL = Path("/home/young/git/daytrade-it-aplan/data/eval/bandit_a7prep/dataset.jsonl")
COST = 0.00215  # 0.20% sell tax + 0.015% commission each side


def spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")

    def rank(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


def load() -> tuple[dict[str, dict], dict[str, dict]]:
    master = {}
    with (D / "master.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            master[r["key"]] = r
    labels = {}
    with (D / "labels.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            labels[r["key"]] = r
    return master, labels


BRACKET = re.compile(r"^\s*\[([^\]]{1,20})\]")
DIGITS = re.compile(r"\d")


def title_shape(title: str) -> str:
    m = BRACKET.match(title)
    if m:
        tag = DIGITS.sub("N", m.group(1))
        return f"[{tag}]"
    return "(none)"


def bucket(v: float | None, edges: list[float], names: list[str]) -> str:
    if v is None:
        return "na"
    for e, n in zip(edges, names, strict=False):
        if v < e:
            return n
    return names[-1]


def table(title: str, groups: dict[str, list[float]], base: float, min_n: int = 30) -> None:
    print(f"\n### {title}   (pool mean {base * 1e4:+.0f} bp)")
    print(f"{'group':28s} {'n':>7s} {'mean bp':>9s} {'vs pool':>9s} {'win%':>6s} {'p90 bp':>8s}")
    rows = [(k, v) for k, v in groups.items() if len(v) >= min_n]
    for k, v in sorted(rows, key=lambda kv: -st.mean(kv[1])):
        m = st.mean(v)
        win = sum(1 for x in v if x > 0) / len(v)
        p90 = sorted(v)[int(0.9 * len(v))]
        print(
            f"{k:28s} {len(v):7d} {m * 1e4:+9.0f} {(m - base) * 1e4:+9.0f} {win:6.1%} {p90 * 1e4:+8.0f}"
        )


def main() -> None:
    master, labels = load()

    # ---------------- A) bridge: wide label vs policy reward ----------------
    rl = {}
    with RL.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("reward") is not None:
                rl[r["key"]] = r
    both = [(labels[k], rl[k]) for k in rl if k in labels]
    print("=" * 78)
    print("A) BRIDGE -- does the wide label track the policy reward?")
    print("=" * 78)
    print(f"rl trainable rows {len(rl)}, of which labelled here {len(both)}")
    wide = [b[0]["ret_close"] - COST for b in both]
    pol = [b[1]["reward"] for b in both]
    print(f"spearman(ret_close_net, policy reward) = {spearman(wide, pol):+.3f}")
    print(f"  wide mean {st.mean(wide) * 1e4:+.0f} bp   policy mean {st.mean(pol) * 1e4:+.0f} bp")
    exc = [b[0]["exc_close"] for b in both if b[0]["exc_close"] is not None]
    pol2 = [b[1]["reward"] for b in both if b[0]["exc_close"] is not None]
    print(f"spearman(exc_close, policy reward)     = {spearman(exc, pol2):+.3f}")
    mfe = [b[0]["mfe"] for b in both]
    print(f"spearman(mfe, policy reward)           = {spearman(mfe, pol):+.3f}")

    # ---------------- B) the wide pool ----------------
    rows = []
    for k, lab in labels.items():
        m = master[k]
        r = lab["ret_close"] - COST
        rows.append((k, m, lab, r))
    base = st.mean([r for _, _, _, r in rows])
    print()
    print("=" * 78)
    print(
        f"B) WIDE POOL -- {len(rows)} judged pairs, {len({m['day'] for _, m, _, _ in rows})} days"
    )
    print("=" * 78)
    print(
        f"net hold-to-close mean {base * 1e4:+.0f} bp   "
        f"win {sum(1 for _, _, _, r in rows if r > 0) / len(rows):.1%}   "
        f"median {st.median([r for *_, r in rows]) * 1e4:+.0f} bp"
    )
    exc_all = [lab["exc_close"] for _, _, lab, _ in rows if lab["exc_close"] is not None]
    print(
        f"market-excess mean {st.mean(exc_all) * 1e4:+.0f} bp  (the 'news baseline is not zero' figure)"
    )

    # oracle: per day, the top-5 by realised reward
    by_day: dict[str, list] = defaultdict(list)
    for row in rows:
        by_day[row[1]["day"]].append(row)
    oracle_keys = set()
    for _day, rs in by_day.items():
        for row in sorted(rs, key=lambda x: -x[3])[:5]:
            oracle_keys.add(row[0])
    orc = [r for r in rows if r[0] in oracle_keys]
    print(
        f"\noracle top-5/day: n={len(orc)}  mean {st.mean([r[3] for r in orc]) * 1e4:+.0f} bp  "
        f"win {sum(1 for r in orc if r[3] > 0) / len(orc):.1%}"
    )

    # ---------------- attribute tables ----------------
    def groups_by(fn) -> dict[str, list[float]]:
        g: dict[str, list[float]] = defaultdict(list)
        for _, m, lab, r in rows:
            g[fn(m, lab)].append(r)
        return g

    def oracle_share(fn) -> None:
        pool: Counter[str] = Counter()
        orcc: Counter[str] = Counter()
        for k, m, lab, _ in rows:
            v = fn(m, lab)
            pool[v] += 1
            if k in oracle_keys:
                orcc[v] += 1
        print(f"{'group':28s} {'pool%':>7s} {'oracle%':>8s} {'lift':>6s}")
        for v, n in pool.most_common(14):
            ps, os_ = n / len(rows), orcc[v] / len(orc)
            print(f"{v:28s} {ps:7.2%} {os_:8.2%} {os_ / ps if ps else 0:6.2f}x")

    print("\n" + "-" * 78)
    print("1. FAN-OUT (how many companies Toss tagged on the article)")
    print("-" * 78)
    fo = lambda m, lab: bucket(  # noqa: E731
        m.get("related_n"),
        [2, 3, 6, 11, 16, 21, 41],
        ["1", "2", "3-5", "6-10", "11-15", "16-20", "21-40", "41+"],
    )
    table("net return by fan-out", groups_by(fo), base)
    oracle_share(fo)

    print("\n" + "-" * 78)
    print("2. TITLE SHAPE (bracketed column tag, digits masked)")
    print("-" * 78)
    ts = lambda m, lab: title_shape(m["title"])[:28]  # noqa: E731
    table("net return by title tag", groups_by(ts), base, min_n=80)
    oracle_share(ts)

    print("\n" + "-" * 78)
    print("3. v2 FIELDS the prompt already asks")
    print("-" * 78)
    for fld in (
        "subject",
        "material_type",
        "persistence",
        "freshness",
        "scale_vs_size",
        "reports_price_move",
    ):
        table(fld, groups_by(lambda m, lab, f=fld: str(m["v2"].get(f))), base)

    print("\n" + "-" * 78)
    print("4. SESSION / TIMING")
    print("-" * 78)
    table("session", groups_by(lambda m, lab: m["session"]), base)
    table(
        "acting minute",
        groups_by(
            lambda m, lab: bucket(
                lab["act_min"],
                [1, 30, 60, 120, 240],
                ["open", "09:01-09:30", "09:30-10:00", "10:00-11:00", "11:00-13:00", "13:00+"],
            )
        ),
        base,
    )

    print("\n" + "-" * 78)
    print("5. PRE-ARTICLE MOVE (entry vs prev close) -- known at decision time")
    print("-" * 78)
    table(
        "pre_move",
        groups_by(
            lambda m, lab: bucket(
                lab.get("pre_move"),
                [-0.03, -0.003, 0.003, 0.02, 0.05, 0.10],
                ["<-3%", "-3..-0.3%", "-0.3..+0.3%", "+0.3..2%", "+2..5%", "+5..10%", ">+10%"],
            )
        ),
        base,
    )

    print("\n" + "-" * 78)
    print("6. LIQUIDITY (that day's traded value)")
    print("-" * 78)
    table(
        "turnover",
        groups_by(
            lambda m, lab: bucket(
                lab.get("turnover"),
                [1e9, 5e9, 2e10, 1e11, 5e11],
                ["<1e9", "1-5e9", "5e9-2e10", "2e10-1e11", "1e11-5e11", ">5e11"],
            )
        ),
        base,
    )

    print("\n" + "-" * 78)
    print("7. SECTOR")
    print("-" * 78)
    table("sector", groups_by(lambda m, lab: str(m.get("sector"))[:28]), base, min_n=300)

    print("\n" + "-" * 78)
    print("8. MARKET")
    print("-" * 78)
    table("market", groups_by(lambda m, lab: str(m.get("market"))), base)

    # ---- separation of the current score ----
    print("\n" + "-" * 78)
    print("9. WHAT v2's SCORE ACTUALLY BUYS")
    print("-" * 78)
    sc = [m["v2"].get("score") or 0.0 for _, m, _, _ in rows]
    rr = [r for *_, r in rows]
    print(f"spearman(v2 score, net return) over all {len(rows)} pairs = {spearman(sc, rr):+.4f}")
    nz = [(s, r) for s, r in zip(sc, rr, strict=True) if s != 0]
    print(
        f"  restricted to the {len(nz)} non-zero scores       = "
        f"{spearman([a for a, _ in nz], [b for _, b in nz]):+.4f}"
    )
    table(
        "v2 score bucket",
        groups_by(
            lambda m, lab: bucket(
                m["v2"].get("score") or 0.0,
                [-0.15, 0.001, 0.15, 0.3, 0.5],
                ["<=-0.15", "-0.15..0", "0", "0..0.15", "0.15..0.3", "0.3..0.5", ">0.5"],
            )
        ),
        base,
    )

    # ---- how much of the oracle gap is even reachable at decision time ----
    print("\n" + "-" * 78)
    print("10. CEILING OF EACH DECISION-TIME ATTRIBUTE (top-5/day within the bucket)")
    print("-" * 78)
    print("If the prompt could perfectly identify one attribute value, what would")
    print("the day's best-5 inside it earn? (an upper bound on that attribute alone)")
    print(f"{'restriction':34s} {'days':>5s} {'n/day':>6s} {'top5 bp':>9s}")

    def restricted_top5(pred, name: str) -> None:
        tot, days = [], 0
        for _day, rs in by_day.items():
            sub = [r for r in rs if pred(r[1], r[2])]
            if len(sub) < 5:
                continue
            days += 1
            tot.extend(r[3] for r in sorted(sub, key=lambda x: -x[3])[:5])
        if days >= 20:
            per = sum(len([r for r in rs if pred(r[1], r[2])]) for rs in by_day.values()) / len(
                by_day
            )
            print(f"{name:34s} {days:5d} {per:6.0f} {st.mean(tot) * 1e4:+9.0f}")

    restricted_top5(lambda m, lab: True, "(all pairs -- the oracle)")
    restricted_top5(lambda m, lab: (m.get("related_n") or 99) <= 2, "fan-out <= 2")
    restricted_top5(lambda m, lab: (m.get("related_n") or 99) <= 5, "fan-out <= 5")
    restricted_top5(lambda m, lab: m["v2"].get("subject") == "main", "subject == main")
    restricted_top5(lambda m, lab: (m["v2"].get("score") or 0) > 0.15, "v2 BUY")
    restricted_top5(
        lambda m, lab: (m["v2"].get("score") or 0) > 0.15 and (m.get("related_n") or 99) <= 5,
        "v2 BUY & fan-out <= 5",
    )
    restricted_top5(lambda m, lab: (lab.get("pre_move") or 0) <= 0.02, "pre_move <= +2%")
    restricted_top5(lambda m, lab: lab["act_min"] == 0, "premarket (enters at open)")
    restricted_top5(lambda m, lab: lab["act_min"] > 0, "intraday")


if __name__ == "__main__":
    main()
