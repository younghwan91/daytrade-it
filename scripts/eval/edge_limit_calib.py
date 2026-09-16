"""Is the minute-bar fill model honest? Calibrate it against ticks and the book.

`edge_limit.py` fills a resting buy when a later minute's low touches the
limit. That is the standard SimBroker rule and it is optimistic in the
direction that flatters the policy: a minute's low only says *some* volume
traded at that price, not that our order was at the front of the queue.

The minute-bar low is itself derived from trades, so ticks cannot disagree
about *whether* a trade happened at our price -- they would be answering the
same question twice. What ticks and quotes can answer, and minute bars cannot,
is **how much volume traded there, against how much was already ahead of us**:

    depth ahead   `bidqty1` at the decision minute, if we join the best bid
    volume at     summed tick volume at price <= limit after our timestamp
    queue fill    volume_at > depth_ahead  -- the order in front cleared first

Reported against the naive rule so the gap is explicit. Both `ticks` and
`quotes` begin 2026-08-24, so this covers the last ~15 trading days of the
pool; the calibration is a check on the model, not a replacement for it.

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

BP = 10_000
OPEN = dt.time(9, 0)
WINDOWS = (5, 30, 390)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fills", default="data/eval/edge/limit_fills.jsonl")
    ap.add_argument("--start", default="2026-08-24")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.fills).read_text().splitlines()]
    rows = [r for r in rows if r["day"] >= args.start]
    by_day: dict[dt.date, list[dict]] = defaultdict(list)
    for r in rows:
        by_day[dt.date.fromisoformat(r["day"])].append(r)
    if not by_day:
        print("no pairs in the tick window")
        return

    conn = connect()
    cur = conn.cursor()
    stats: dict[str, list[int]] = defaultdict(list)
    depth_ahead: list[float] = []
    n_quote = 0

    for day in sorted(by_day):
        rs = by_day[day]
        codes = sorted({r["code"] for r in rs})
        base = dt.datetime.combine(day, OPEN)

        # depth at the touch, at each pair's decision minute
        cur.execute(
            "SELECT DISTINCT ON (code, date_trunc('minute', ts)) code, "
            "date_trunc('minute', ts) AS m, bid1, bidqty1 FROM quotes "
            "WHERE code = ANY(%s) AND ts >= %s AND ts < %s "
            "ORDER BY code, m, ts DESC",
            (codes, day, day + dt.timedelta(days=1)),
        )
        book: dict[tuple[str, dt.datetime], tuple[float, float]] = {
            (c, m): (float(b or 0), float(q or 0)) for c, m, b, q in cur.fetchall()
        }

        cur.execute(
            "SELECT code, ts, price, volume FROM ticks WHERE code = ANY(%s) "
            "AND ts >= %s AND ts < %s ORDER BY code, ts",
            (codes, day, day + dt.timedelta(days=1)),
        )
        trades: dict[str, list[tuple]] = defaultdict(list)
        for code, ts, price, vol in cur.fetchall():
            trades[code].append((ts, float(price), float(vol or 0)))

        for r in rs:
            start = base + dt.timedelta(minutes=r["act_min"])
            bid, qty = book.get((r["code"], start), (0.0, 0.0))
            if bid <= 0:
                continue
            n_quote += 1
            ahead = qty * bid  # KRW ahead of us at the touch
            depth_ahead.append(ahead)
            tr = trades.get(r["code"], [])
            for w in WINDOWS:
                stop = start + dt.timedelta(minutes=w)
                krw = sum(p * v for ts, p, v in tr if start <= ts <= stop and p <= r["limit"])
                naive = bool(r["fills"].get(str(w)))
                stats[f"{w}|naive"].append(naive)
                stats[f"{w}|queue"].append(krw > ahead)
                stats[f"{w}|both"].append(naive and krw > ahead)
        print(f"{day}  pairs={len(rs)}", flush=True)
    conn.close()

    print(f"\n{n_quote:,} pairs with a quote at the decision minute, from {args.start}")
    if depth_ahead:
        d = sorted(depth_ahead)
        print(
            f"depth ahead of us at the touch: median {d[len(d) // 2]:,.0f} KRW, "
            f"p90 {d[9 * len(d) // 10]:,.0f} KRW   (our order is 100,000 KRW)\n"
        )
    print(f"{'window':<10}{'naive fill':>12}{'queue fill':>12}{'both':>10}{'naive kept':>12}")
    for w in WINDOWS:
        naive = stats[f"{w}|naive"]
        queue = stats[f"{w}|queue"]
        both = stats[f"{w}|both"]
        if not naive:
            continue
        kept = sum(both) / sum(naive) if sum(naive) else float("nan")
        print(
            f"{str(w) + ' min':<10}{statistics.fmean(naive):>12.1%}"
            f"{statistics.fmean(queue):>12.1%}{statistics.fmean(both):>10.1%}{kept:>12.1%}"
        )
    print(
        "\n`naive kept` is the share of minute-bar fills that survive the queue "
        "test. If it is\nhigh, the optimistic model is close enough to use; if it "
        "is low, the assume-fill\nnumbers are unusable and should be reported as "
        "an upper bound only.\n"
    )


if __name__ == "__main__":
    main()
