"""The latency ladder and the shape of the move. Reads path_sweep.jsonl."""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict

BP = 10_000
COST = 0.00367
LAGS = ["0", "1", "2", "3", "5", "10", "15", "22", "30", "45", "60", "90"]


def day_stats(by_day, reps=2000, seed=7):
    days = sorted(by_day)
    per = [st.fmean(by_day[d]) for d in days]
    n = len(days)
    if n < 2:
        return None
    rng = random.Random(seed)
    b = sorted(st.fmean([per[rng.randrange(n)] for _ in range(n)]) for _ in range(reps))
    return {
        "n": sum(len(v) for v in by_day.values()),
        "days": n,
        "mean": st.fmean(per),
        "se": st.stdev(per) / n**0.5,
        "lo": b[int(0.025 * reps)],
        "hi": b[int(0.975 * reps)],
        "ge0": sum(v >= 0 for v in per),
    }


def show(label, by_day):
    s = day_stats(by_day)
    if not s:
        print(f"{label:34s} n=0")
        return
    print(
        f"{label:34s} n={s['n']:6d} d={s['days']:3d} {s['mean'] * BP:+8.1f} bp "
        f"se {s['se'] * BP:6.1f} [{s['lo'] * BP:+7.1f},{s['hi'] * BP:+7.1f}] "
        f"days>=0 {s['ge0']}/{s['days']}"
    )


def ladder(rows, title, keep):
    print(f"\n=== {title} ===")
    print("  entry at pub+L, hold to close; alpha = own - matched null at that same minute")
    for L in LAGS:
        a, netd = defaultdict(list), defaultdict(list)
        for r in rows:
            if not keep(r):
                continue
            f = (r.get("fwd") or {}).get(L)
            if not f or f.get("null") is None:
                continue
            a[r["day"]].append(f["own"] - f["null"])
            netd[r["day"]].append(f["own"] - COST)
        s, sn = day_stats(a), day_stats(netd)
        if not s:
            print(f"  L=+{L:>2} min  n=0")
            continue
        print(
            f"  L=+{L:>2} min  n={s['n']:6d}  alpha {s['mean'] * BP:+7.1f} bp "
            f"(se {s['se'] * BP:5.1f}) [{s['lo'] * BP:+7.1f},{s['hi'] * BP:+7.1f}]  "
            f"net {sn['mean'] * BP:+8.1f} bp  days>=0 {s['ge0']}/{s['days']}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="data/eval/latency/path_sweep.jsonl")
    a = ap.parse_args()
    with open(a.rows) as f:
        rows = [json.loads(line) for line in f]
    print(f"rows {len(rows)}")

    ladder(rows, "H1. Whole pool", lambda _r: True)
    ladder(
        rows,
        "H2. Intraday-published only (pub_min > 0)",
        lambda r: not r["premarket"] and r["pub_min"] > 0,
    )
    ladder(rows, "H3. BUY subset (v2 score > 0.15)", lambda r: (r.get("score") or 0) > 0.15)
    ladder(
        rows,
        "H4. Intraday-published BUY subset",
        lambda r: not r["premarket"] and r["pub_min"] > 0 and (r.get("score") or 0) > 0.15,
    )

    # ---- shape of the move, on events that actually moved
    print("\n=== G. Shape of the move, conditioned on the event having moved ===")
    print("  share of the pub->pub+120min excess move already realized by pub+k.")
    print("  Benchmark: for a driftless walk conditioned on its endpoint the share is")
    print("  exactly k/120, so anything above that line is genuine front-loading.")
    for thresh in (0.01, 0.02, 0.05):
        sel = []
        for r in rows:
            nc = r.get("null_cell") or {}
            o = r["own"]
            if o.get("120") is None or nc.get("120") is None:
                continue
            tot = o["120"] - nc["120"]
            if abs(tot) < thresh:
                continue
            sel.append((r, tot))
        if len(sel) < 50:
            print(f"  |excess move at +120min| >= {thresh:.0%}: n={len(sel)} too few")
            continue
        print(f"\n  |excess move at +120min| >= {thresh:.0%}   n={len(sel)}")
        for k in ("1", "2", "3", "5", "10", "15", "22", "30", "60", "120"):
            fr = []
            for r, tot in sel:
                nc = r["null_cell"]
                o = r["own"].get(k)
                if o is None or nc.get(k) is None:
                    continue
                fr.append((o - nc[k]) / tot)
            if fr:
                fr.sort()
                print(
                    f"    by pub+{k:>3} min: mean {st.fmean(fr) * 100:6.1f}%  "
                    f"median {fr[len(fr) // 2] * 100:6.1f}%   "
                    f"(driftless benchmark {int(k) / 120 * 100:5.1f}%)"
                )


if __name__ == "__main__":
    main()
