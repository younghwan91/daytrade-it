"""Does holding longer change the sign?

Cost is charged per round trip, not per day: one buy, one sell, ~36.7 bp,
whatever the holding period. So a hold of h days pays the same toll spread over
h days of exposure. That is the one lever that attacks the cost term instead of
trying to out-earn it, and the data to test it already exists.

For every pair, entry is unchanged -- the same acting minute, the same
SimBroker quote rule -- and the exit moves to the close of day d+h. The null
moves with it: the same universe, the same minute, the same horizon, so
`alpha` stays a like-for-like difference at every h and the day, minute and
horizon are all differenced out.

The return is composed from two legs rather than read off one price, because
`minute_bars` is raw and `daily_bars_adjusted` is adjusted -- dividing one by
the other breaks on any split or dividend in between:

    1 + ret_h = (1 + same-day raw return from minute bars)
              * (adjusted close at d+h / adjusted close at d)

Each leg is internally consistent, so the product is split-safe.

Reported per horizon:
  null    what a universe name earned over the same window   (drift)
  alpha   what our pairs earned above that                   (selection)
  net     alpha + null - COST, one round trip                (the trade)
  /day    net divided by h trading days of capital           (the comparison)

A longer hold that merely rides more drift is not an improvement; the question
is whether `net` crosses zero and whether `alpha` grows with h, which it would
if news took longer than a session to be priced.

READ-ONLY on kr_quant.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_db import connect  # noqa: E402
from edge_null import fetch, trading_days, universe_for  # noqa: E402

BP = 10_000
#: the round trip this system actually pays -- see edge_cost.py
COST = 0.00367
HORIZONS = (0, 1, 2, 5, 10)


def daily_closes(cur, lo: dt.date, hi: dt.date) -> dict[dt.date, dict[str, float]]:
    cur.execute(
        "SELECT date, code, close FROM daily_bars_adjusted WHERE date >= %s AND date <= %s",
        (lo, hi),
    )
    out: dict[dt.date, dict[str, float]] = defaultdict(dict)
    for date, code, c in cur.fetchall():
        if c and float(c) > 0:
            out[date][code] = float(c)
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/eval/prompt_v3/labels.jsonl")
    ap.add_argument("--size", type=int, default=300)
    args = ap.parse_args()

    pairs: dict[dt.date, list[dict]] = defaultdict(list)
    with Path(args.labels).open() as f:
        for line in f:
            r = json.loads(line)
            r["code"] = r.get("code") or r["key"].split("|")[1]
            pairs[dt.date.fromisoformat(r["day"])].append(r)
    days = sorted(pairs)

    conn = connect()
    cur = conn.cursor()
    cal = trading_days(cur, days[0] - dt.timedelta(days=30), days[-1] + dt.timedelta(days=40))
    idx = {d: i for i, d in enumerate(cal)}
    prev_of = {d: cal[i - 1] for i, d in enumerate(cal) if i}
    closes = daily_closes(cur, days[0], cal[-1])

    # per (horizon, day): our pairs' returns, and the universe's
    ours: dict[int, dict[str, list[float]]] = {h: defaultdict(list) for h in HORIZONS}
    null: dict[int, dict[str, list[float]]] = {h: defaultdict(list) for h in HORIZONS}

    for day in days:
        prev = prev_of.get(day)
        if prev is None:
            continue
        univ = universe_for(cur, prev, args.size)
        rows = pairs[day]
        bars = fetch(cur, day, sorted({r["code"] for r in rows} | set(univ)))
        i = idx[day]
        exits = {h: (cal[i + h] if i + h < len(cal) else None) for h in HORIZONS}

        # carry[h][code] = adjusted close at d+h / adjusted close at d
        today_adj = closes.get(day, {})
        carry: dict[int, dict[str, float]] = {}
        for h in HORIZONS:
            ex = exits[h]
            px = closes.get(ex, {}) if ex else {}
            carry[h] = {c: px[c] / today_adj[c] for c in px if c in today_adj and today_adj[c] > 0}

        # The universe null, weighted by how many of OUR pairs used each minute.
        # Weighting minutes equally would compare a pool that piles 42% of its
        # pairs on the 09:00 open against an average over the whole session,
        # and the difference would land in `alpha` as if it were selection.
        weight: dict[int, int] = defaultdict(int)
        for r in rows:
            weight[r["act_min"]] += 1
        same_day = {c: bars[c] for c in univ if c in bars}
        for h in HORIZONS:
            k = carry[h]
            if not k:
                continue
            num = den = 0.0
            for m, w in weight.items():
                vals = [
                    (1 + r0) * k[c] - 1
                    for c, d in same_day.items()
                    if c in k and (r0 := d.ret(m)) is not None
                ]
                if vals:
                    num += w * statistics.fmean(vals)
                    den += w
            if den:
                null[h][str(day)].append(num / den)

        for r in rows:
            d = bars.get(r["code"])
            if d is None:
                continue
            r0 = d.ret(r["act_min"])
            if r0 is None:
                continue
            for h in HORIZONS:
                k = carry[h].get(r["code"])
                if k is None:
                    continue
                ours[h][str(day)].append((1 + r0) * k - 1)
        print(f"{day}  pairs={len(rows)}", flush=True)
    conn.close()

    print(f"\ncost per round trip {COST * BP:.1f} bp, charged once at every horizon\n")
    print(
        f"{'hold':<10}{'pairs':>9}{'days':>6}{'null':>10}{'se':>7}{'alpha':>9}{'se':>7}"
        f"{'net':>10}{'net/day':>9}"
    )
    for h in HORIZONS:
        od, nd = ours[h], null[h]
        common = sorted(set(od) & set(nd))
        if not common:
            continue
        o = [statistics.fmean(od[d]) for d in common]
        n = [statistics.fmean(nd[d]) for d in common]
        a = [x - y for x, y in zip(o, n, strict=True)]
        npairs = sum(len(od[d]) for d in common)
        se = lambda v: statistics.stdev(v) / (len(v) ** 0.5)  # noqa: E731
        net = statistics.fmean(o) - COST
        per = net / max(h, 1)
        print(
            f"{('same day' if h == 0 else f'+{h}d'):<10}{npairs:>9,}{len(common):>6}"
            f"{statistics.fmean(n) * BP:>+10.1f}{se(n) * BP:>7.1f}"
            f"{statistics.fmean(a) * BP:>+9.1f}{se(a) * BP:>7.1f}"
            f"{net * BP:>+10.1f}{per * BP:>+9.1f}"
        )
    print()


if __name__ == "__main__":
    main()
