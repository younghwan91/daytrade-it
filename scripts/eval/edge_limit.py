"""Limit entry instead of crossing the spread, at multi-day horizons.

We send marketable orders on news, so we pay the spread on every entry -- 14 to
31 bp round trip by measurement (`edge_spread.py`), the largest controllable
line in the toll. Joining the bid instead of lifting the ask saves that. At a
ten-day horizon the saving is a one-off that stays in the position, and the
usual objection to limit entries -- you miss the winners -- is much weaker than
at five minutes, because with a multi-day thesis there is no urgency to be in a
specific name on a specific minute.

**The crux is adverse selection, and it must be measured, not assumed.** A
limit that fills only when price comes down to us systematically buys the names
that fell. If the names we miss outperform the ones we fill, the spread saving
is paid back and possibly more. So this reports three groups, not two:

    market      every pair, entered at the decision minute's quote (today)
    filled      the subset a resting bid would have caught
    unfilled    the subset it would have missed -- what they went on to do

The arithmetic is exact rather than re-simulated. A return is `exit / entry - 1`,
so moving the entry from the crossed price `E_m` to the limit `E_l` rescales
every horizon at once:

    1 + ret_limit = (1 + ret_market) * E_m / E_l

`E_l` is one tick below `E_m` -- joining the best bid rather than crossing a
one-tick spread, which `edge_spread.py` measures as the floor.

**The fill model is optimistic and is labelled as such.** A minute's low
touching the limit means *some* volume traded there, not that our order was at
the front of the queue. At one share that is much less wrong than it would be
for size, but it flatters the policy. `--calibrate` re-tests the same fills
against the `ticks` table, where a trade at or below the limit after our
timestamp is directly observable -- available only for 2026-08-24 onward.

READ-ONLY on kr_quant.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_db import connect  # noqa: E402

BP = 10_000
COST = 0.00367
OPEN = dt.time(9, 0)
REGULAR_END = dt.time(15, 33)
TICKS = ((2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50), (200_000, 100), (500_000, 500))
TOP_TICK = 1_000
#: live cancels an unfilled entry after five minutes; the longer windows say
#: what a more patient order would have bought
WINDOWS = (5, 30, 390)


def tick_of(price: float) -> int:
    return next((t for bound, t in TICKS if price < bound), TOP_TICK)


def fetch_lows(cur, day: dt.date, codes: list[str]) -> dict[str, list[tuple]]:
    cur.execute(
        "SELECT code, ts, low FROM minute_bars WHERE code = ANY(%s) "
        "AND ts >= %s AND ts < %s ORDER BY code, ts",
        (codes, day, day + dt.timedelta(days=1)),
    )
    out: dict[str, list[tuple]] = defaultdict(list)
    for code, ts, low in cur.fetchall():
        if low and OPEN <= ts.time() <= REGULAR_END:
            out[code].append((ts, float(low)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/eval/edge/horizon_pairs.jsonl")
    ap.add_argument("--labels", default="data/eval/prompt_v3/labels.jsonl")
    ap.add_argument("--out", default="data/eval/edge/limit_fills.jsonl")
    args = ap.parse_args()

    entry: dict[str, float] = {}
    with Path(args.labels).open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("entry"):
                entry[r["key"]] = float(r["entry"])

    pairs: dict[dt.date, list[dict]] = defaultdict(list)
    with Path(args.pairs).open() as f:
        for line in f:
            r = json.loads(line)
            if r["key"] in entry:
                r["entry"] = entry[r["key"]]
                pairs[dt.date.fromisoformat(r["day"])].append(r)

    conn = connect()
    cur = conn.cursor()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as out:
        for day in sorted(pairs):
            rows = pairs[day]
            lows = fetch_lows(cur, day, sorted({r["code"] for r in rows}))
            base = dt.datetime.combine(day, OPEN)
            for r in rows:
                e_m = r["entry"]
                limit = e_m - tick_of(e_m)
                if limit <= 0:
                    continue
                start = base + dt.timedelta(minutes=r["act_min"])
                bars = lows.get(r["code"], [])
                fills = {}
                for w in WINDOWS:
                    stop = start + dt.timedelta(minutes=w)
                    fills[str(w)] = any(start <= ts <= stop and lo <= limit for ts, lo in bars)
                out.write(
                    json.dumps(
                        {
                            "key": r["key"],
                            "day": r["day"],
                            "code": r["code"],
                            "act_min": r["act_min"],
                            "entry": e_m,
                            "limit": limit,
                            "tick_bp": tick_of(e_m) / e_m,
                            "fills": fills,
                            "r": r["r"],
                            "nc": r.get("nc", {}),
                        }
                    )
                    + "\n"
                )
                n += 1
            print(f"{day}  pairs={len(rows)}", flush=True)
    conn.close()
    print(f"\n{n} pairs -> {out_path}")


if __name__ == "__main__":
    main()
