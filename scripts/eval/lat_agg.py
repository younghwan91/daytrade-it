"""Aggregate the publication-anchored price paths.

Two questions:
  1. How much of the move is already consumed by the time we act?
  2. What alpha would a T+1-minute entry on the same events have earned,
     against the same day x minute x sigma-quartile x value-quartile null?

Inference is clustered on the day (the unit that repeats), bootstrap over days.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict

BP = 10_000
COST = 0.00367  # pool-weighted true round trip, per edge-hunt 2.5
HORIZONS = ["1", "2", "3", "5", "10", "15", "22", "30", "60", "120", "close"]


def day_stats(by_day, reps=2000, seed=7):
    days = sorted(by_day)
    per_day = [st.fmean(by_day[d]) for d in days]
    n = len(days)
    if n < 2:
        return None
    rng = random.Random(seed)
    boots = sorted(st.fmean([per_day[rng.randrange(n)] for _ in range(n)]) for _ in range(reps))
    return {
        "n": sum(len(v) for v in by_day.values()),
        "days": n,
        "mean": st.fmean(per_day),
        "se": st.stdev(per_day) / n**0.5,
        "lo": boots[int(0.025 * reps)],
        "hi": boots[int(0.975 * reps)],
        "days_ge0": sum(v >= 0 for v in per_day),
    }


def show(label, by_day):
    s = day_stats(by_day)
    if not s:
        print(f"{label:38s}  n=0")
        return
    print(
        f"{label:38s} n={s['n']:6d} d={s['days']:3d}  "
        f"{s['mean'] * BP:+8.1f} bp  se {s['se'] * BP:6.1f}  "
        f"[{s['lo'] * BP:+7.1f}, {s['hi'] * BP:+7.1f}]  days>=0 {s['days_ge0']}/{s['days']}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="data/eval/latency/path_sweep.jsonl")
    ap.add_argument("--exclude-premarket", action="store_true")
    args = ap.parse_args()

    rows = []
    with open(args.rows) as f:
        for line in f:
            r = json.loads(line)
            if args.exclude_premarket and r["premarket"]:
                continue
            rows.append(r)
    print(f"rows {len(rows)}   premarket-anchored {sum(1 for r in rows if r['premarket'])}")

    lags = [r["lag"] for r in rows]
    lags.sort()
    print(
        f"decision lag (act_min - pub_min), minutes: med {lags[len(lags) // 2]}  "
        f"p25 {lags[len(lags) // 4]}  p75 {lags[3 * len(lags) // 4]}  "
        f"p90 {lags[int(0.9 * len(lags))]}  mean {st.fmean(lags):.1f}"
    )

    # ---- 1. raw and matched-null-differenced drift from the publication minute
    print("\n=== A. From the PUBLICATION minute: pool vs matched null (alpha, bp) ===")
    print("    (own return over k minutes, minus the same-cell cross-section's)")
    for k in HORIZONS:
        by_day = defaultdict(list)
        for r in rows:
            nc = r.get("null_cell")
            if not nc or k not in nc:
                continue
            o = r["own"].get(k)
            if o is None:
                continue
            by_day[r["day"]].append(o - nc[k])
        show(f"  pub -> pub+{k}", by_day)

    print("\n=== B. Raw pool drift from the publication minute (bp, no null) ===")
    for k in HORIZONS:
        by_day = defaultdict(list)
        for r in rows:
            o = r["own"].get(k)
            if o is not None:
                by_day[r["day"]].append(o)
        show(f"  pub -> pub+{k}", by_day)

    # ---- 2. what fraction is consumed before we act
    print("\n=== C. Consumed before our decision minute ===")
    a_by_day, c_by_day, tot_by_day = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        pre = r["own"].get("act")
        tot = r["own"].get("close")
        post = r.get("act_own_close")
        if pre is None or tot is None or post is None:
            continue
        a_by_day[r["day"]].append(pre)
        c_by_day[r["day"]].append(post)
        tot_by_day[r["day"]].append(tot)
    show("  pub -> our entry (missed)", a_by_day)
    show("  our entry -> close (captured)", c_by_day)
    show("  pub -> close (whole window)", tot_by_day)

    # ---- 3. alpha of entering at pub+1 vs at our actual minute
    print("\n=== D. Hold-to-close alpha, entry at pub+1 vs at our actual minute ===")
    fast, slow, net_fast, net_slow = (defaultdict(list) for _ in range(4))
    for r in rows:
        nc, na = r.get("null_cell"), r.get("act_null_cell")
        if not nc or not na or "close" not in nc or "close" not in na:
            continue
        # entry at pub+1: own path from pub to pub+1, then pub+1 to close.
        o1 = r["own"].get("1")
        oc = r["own"].get("close")
        if o1 is None or oc is None or (1 + o1) <= 0:
            continue
        r_fast = (1 + oc) / (1 + o1) - 1
        n1 = nc.get("1")
        ncl = nc.get("close")
        if n1 is None or ncl is None or (1 + n1) <= 0:
            continue
        nullfast = (1 + ncl) / (1 + n1) - 1
        fast[r["day"]].append(r_fast - nullfast)
        net_fast[r["day"]].append(r_fast - COST)
        ra = r.get("act_own_close")
        if ra is None:
            continue
        slow[r["day"]].append(ra - na["close"])
        net_slow[r["day"]].append(ra - COST)
    show("  alpha, entry at pub+1min", fast)
    show("  alpha, entry at our minute", slow)
    show("  NET, entry at pub+1min", net_fast)
    show("  NET, entry at our minute", net_slow)

    # ---- 4. BUY subset only (score > 0.15)
    print("\n=== E. Same, restricted to score > 0.15 (what live would buy) ===")
    fb, sb = defaultdict(list), defaultdict(list)
    for r in rows:
        if (r.get("score") or 0) <= 0.15:
            continue
        nc, na = r.get("null_cell"), r.get("act_null_cell")
        if not nc or not na:
            continue
        o1, oc = r["own"].get("1"), r["own"].get("close")
        n1, ncl = nc.get("1"), nc.get("close")
        if None in (o1, oc, n1, ncl):
            continue
        fb[r["day"]].append(((1 + oc) / (1 + o1) - 1) - ((1 + ncl) / (1 + n1) - 1))
        ra = r.get("act_own_close")
        if ra is not None and "close" in na:
            sb[r["day"]].append(ra - na["close"])
    show("  BUY alpha, entry at pub+1min", fb)
    show("  BUY alpha, entry at our minute", sb)

    # ---- 5. intraday-published only (the ones latency can actually help)
    print("\n=== F. Intraday-published articles only (pub_min > 0) ===")
    fi, si = defaultdict(list), defaultdict(list)
    for r in rows:
        if r["premarket"] or r["pub_min"] == 0:
            continue
        nc, na = r.get("null_cell"), r.get("act_null_cell")
        if not nc or not na:
            continue
        o1, oc = r["own"].get("1"), r["own"].get("close")
        n1, ncl = nc.get("1"), nc.get("close")
        if None in (o1, oc, n1, ncl):
            continue
        fi[r["day"]].append(((1 + oc) / (1 + o1) - 1) - ((1 + ncl) / (1 + n1) - 1))
        ra = r.get("act_own_close")
        if ra is not None and "close" in na:
            si[r["day"]].append(ra - na["close"])
    show("  alpha, entry at pub+1min", fi)
    show("  alpha, entry at our minute", si)


if __name__ == "__main__":
    main()
