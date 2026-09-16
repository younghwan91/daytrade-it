"""Limit entry, redesigned: unfilled is two populations, and we get to retry.

Two flaws in the first pass, both from scalp-it, both material.

**Unfilled is not one thing.** It is (A) price ran away upward before we could
fill -- a genuinely missed winner -- and (B) price hovered near our limit but
the queue never cleared, where almost nothing was missed. Pooled, A dominates
the average and makes limit entry look far worse than it is. They are separated
here from the bar path.

A caveat that took a run to notice: under a naive `low <= limit` fill rule, B
is EMPTY BY CONSTRUCTION -- an unfilled day means price never reached our
price, so there is no "touched but did not fill" case without a queue. The
split below is therefore the nearest honest proxy on bar data: **A** if the
low stayed strictly above our original entry (it went straight up), **B** if
it dipped to or below the entry but never reached the limit one tick lower.
The true B -- touched the limit, queue never cleared -- needs the tick-level
model and exists only for 2026-08-24 onward.

**And "unfilled" is the wrong terminal state.** Live would not post once and
give up. At a ten-day horizon there is no urgency to be in a specific name on a
specific minute, so the order is re-posted the next day, and the next. The
success criterion is therefore **acquired by limit within N days**, and most of
A is recovered by retrying. This asymmetry is the whole reason the experiment
can work at our horizon and not at scalp-it's: at 300 seconds the opportunity
evaporates before a retry is possible.

**Through-fills get their own bucket.** A fill where price traded *through* the
limit rather than resting on it is the maximum-adverse-selection subset -- we
bought because it fell past us. Pooled with ordinary fills that signal is
buried, and whether it survives at +10d is the most direct available answer to
the question this experiment exists to ask.

Retry entry price is the limit on the day it fills, so a later fill is priced
at that day's quote minus a tick; the horizon is measured from the original
decision day, so a slow acquisition eats into its own holding period rather
than being handed a free later start.

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
from edge_null import trading_days  # noqa: E402

BP = 10_000
OPEN = dt.time(9, 0)
REGULAR_END = dt.time(15, 33)
TICKS = ((2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50), (200_000, 100), (500_000, 500))
TOP_TICK = 1_000
RETRY_DAYS = (0, 1, 3)


def tick_of(price: float) -> int:
    return next((t for bound, t in TICKS if price < bound), TOP_TICK)


def fetch(cur, lo: dt.date, hi: dt.date, codes: list[str]) -> dict:
    cur.execute(
        "SELECT code, ts, open, low, close FROM minute_bars WHERE code = ANY(%s) "
        "AND ts >= %s AND ts < %s ORDER BY code, ts",
        (codes, lo, hi + dt.timedelta(days=1)),
    )
    out: dict[tuple[str, dt.date], list[tuple]] = defaultdict(list)
    for code, ts, o, low, c in cur.fetchall():
        if low and OPEN <= ts.time() <= REGULAR_END:
            out[(code, ts.date())].append((ts, float(o), float(low), float(c)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/eval/edge/horizon_pairs.jsonl")
    ap.add_argument("--labels", default="data/eval/prompt_v3/labels.jsonl")
    ap.add_argument("--out", default="data/eval/edge/limit_retry.jsonl")
    ap.add_argument("--near", type=float, default=1.0, help="ticks above the limit that count as B")
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
    days = sorted(pairs)

    conn = connect()
    cur = conn.cursor()
    cal = trading_days(cur, days[0], days[-1] + dt.timedelta(days=20))
    idx = {d: i for i, d in enumerate(cal)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as out:
        for day in days:
            rows = pairs[day]
            i0 = idx[day]
            span = cal[i0 : i0 + max(RETRY_DAYS) + 1]
            bars = fetch(cur, span[0], span[-1], sorted({r["code"] for r in rows}))
            base = dt.datetime.combine(day, OPEN)

            for r in rows:
                e_m = r["entry"]
                tick = tick_of(e_m)
                limit0 = e_m - tick
                if limit0 <= 0:
                    continue
                start = base + dt.timedelta(minutes=r["act_min"])

                rec = {
                    "key": r["key"],
                    "day": r["day"],
                    "code": r["code"],
                    "act_min": r["act_min"],
                    "entry": e_m,
                    "tick_bp": tick / e_m,
                    "r": r["r"],
                    "nc": r.get("nc", {}),
                }

                # day 0: rest from the decision minute to the close
                d0 = [b for b in bars.get((r["code"], day), []) if b[0] >= start]
                low0 = min((b[2] for b in d0), default=None)
                filled_at: float | None = None
                fill_day = None
                through = False
                if low0 is not None and low0 <= limit0:
                    filled_at, fill_day = limit0, 0
                    through = low0 < limit0 - 1e-9

                # retries: re-post at that day's own quote minus a tick
                if filled_at is None:
                    for k in range(1, max(RETRY_DAYS) + 1):
                        if i0 + k >= len(cal):
                            break
                        dk = cal[i0 + k]
                        bk = bars.get((r["code"], dk), [])
                        if not bk:
                            continue
                        lim_k = bk[0][1] - tick_of(bk[0][1])
                        lowk = min(b[2] for b in bk)
                        if lowk <= lim_k:
                            filled_at, fill_day = lim_k, k
                            through = lowk < lim_k - 1e-9
                            break

                rec["fill_day"] = fill_day
                rec["fill_px"] = filled_at
                rec["through"] = through
                # A vs B, for the day-0 misses only
                if fill_day is None or fill_day > 0:
                    near = limit0 + args.near * tick
                    rec["miss_kind"] = (
                        "B_near" if (low0 is not None and low0 <= near) else "A_ranaway"
                    )
                else:
                    rec["miss_kind"] = None
                out.write(json.dumps(rec) + "\n")
                n += 1
            print(f"{day}  pairs={len(rows)}", flush=True)
    conn.close()
    print(f"\n{n} pairs -> {out_path}")


if __name__ == "__main__":
    main()
