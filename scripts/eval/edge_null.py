"""Matched null for every decision minute: the same day, the same minute, the
same universe, no news signal.

READ-ONLY on kr_quant. For each trading day the universe is
`select_universe(previous trading day's trade_value)` -- exactly the set the
live sweep could have picked from -- and for every distinct acting minute that
appears in the pool we price the WHOLE cross-section with `prompt_labels.py`'s
entry and exit rules. That is the null: what you get for holding a universe
name from that minute to the close without reading any news.

No sampling: the full cross-section is cheap because `ret_close` for a code is
just `day's last regular close / entry(minute) - 1`, and only `entry` moves
with the minute.

Output, one row per (day, acting minute):
    day, act_min, n_univ, null_mean, null_med, mkt_ret, n_pool
`mkt_ret` is 069500 over the identical window, so `null_mean - mkt_ret` is the
cross-section's excess over the index at that minute.
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

MARKET = "069500"
OPEN = dt.time(9, 0)
REGULAR_END = dt.time(15, 33)
UNIVERSE_SIZE = 300
#: minutes from 09:00 to REGULAR_END inclusive
SLOTS = 394
#: 15:20, the live signal expiry
EXPIRY_SLOT = 380


def _slot(ts: dt.datetime) -> int:
    return (ts.hour - 9) * 60 + ts.minute


class Day:
    """One code-day's bars indexed by minute slot, with both rules precomputed.

    `entry[i]`  -- SimBroker's quote at slot i: that minute's bar open, else the
                   last completed regular bar's close.
    `final`     -- close of the day's last regular bar, at slot `final_slot`.
                   `window(bars, minute)[-1]` of prompt_labels.py is that bar
                   for every minute at or before it, so `ret` needs no scan.
    `expiry`    -- the same at the 15:20 signal expiry (`ret_1520`).
    """

    __slots__ = ("entry", "final", "final_slot", "expiry", "expiry_slot")

    def __init__(self, bars: list[tuple]) -> None:
        opens: list[float | None] = [None] * SLOTS
        closes: list[float | None] = [None] * SLOTS
        for ts, o, _h, _l, c, _tv in bars:
            t = ts.time()
            if t < OPEN or t > REGULAR_END:
                continue
            i = _slot(ts)
            if 0 <= i < SLOTS:
                opens[i] = float(o)
                closes[i] = float(c)
        entry: list[float | None] = [None] * SLOTS
        prev_close: float | None = None
        for i in range(SLOTS):
            entry[i] = opens[i] if opens[i] is not None else prev_close
            if closes[i] is not None:
                prev_close = closes[i]
        self.entry = entry
        self.final, self.final_slot = self._last(closes, SLOTS - 1)
        self.expiry, self.expiry_slot = self._last(closes, EXPIRY_SLOT)

    @staticmethod
    def _last(closes: list[float | None], upto: int) -> tuple[float | None, int]:
        for i in range(min(upto, SLOTS - 1), -1, -1):
            if closes[i] is not None:
                return closes[i], i
        return None, -1

    def _ret(self, i: int, exit_px: float | None, exit_slot: int) -> float | None:
        if not (0 <= i <= exit_slot) or exit_px is None:
            return None
        e = self.entry[i]
        if not e or e <= 0:
            return None
        return exit_px / e - 1

    def ret(self, i: int) -> float | None:
        return self._ret(i, self.final, self.final_slot)

    def ret_1520(self, i: int) -> float | None:
        return self._ret(i, self.expiry, self.expiry_slot)


def trading_days(cur, lo: dt.date, hi: dt.date) -> list[dt.date]:
    cur.execute(
        "SELECT DISTINCT date FROM daily_bars_adjusted WHERE date >= %s AND date <= %s "
        "ORDER BY date",
        (lo, hi),
    )
    return [r[0] for r in cur.fetchall()]


def universe_for(cur, prev: dt.date, size: int) -> list[str]:
    cur.execute(
        "SELECT code, trade_value FROM daily_bars_adjusted WHERE date = %s",
        (prev,),
    )
    vals = {
        c: float(v) for c, v in cur.fetchall() if v and float(v) > 0 and len(c) == 6 and c.isdigit()
    }
    return sorted(vals, key=lambda c: (-vals[c], c))[:size]


def fetch(cur, day: dt.date, codes: list[str]) -> dict[str, Day]:
    cur.execute(
        "SELECT code, ts, open, high, low, close, trade_value FROM minute_bars "
        "WHERE code = ANY(%s) AND ts >= %s AND ts < %s ORDER BY code, ts",
        (codes, day, day + dt.timedelta(days=1)),
    )
    raw: dict[str, list[tuple]] = defaultdict(list)
    for code, ts, o, h, lo, c, tv in cur.fetchall():
        raw[code].append((ts, o, h, lo, c, tv))
    return {c: Day(bs) for c, bs in raw.items()}


def load_anchors(labels: Path) -> dict[dt.date, dict[int, int]]:
    """{day: {act_min: pool pairs at that minute}} from labels.jsonl."""
    out: dict[dt.date, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    with labels.open() as f:
        for line in f:
            r = json.loads(line)
            out[dt.date.fromisoformat(r["day"])][int(r["act_min"])] += 1
    return {d: dict(v) for d, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/eval/prompt_v3/labels.jsonl")
    ap.add_argument("--out", default="data/eval/edge/null_sweep.jsonl")
    ap.add_argument("--size", type=int, default=UNIVERSE_SIZE)
    args = ap.parse_args()

    anchors = load_anchors(Path(args.labels))
    days = sorted(anchors)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = connect()
    cur = conn.cursor()
    cal = trading_days(cur, days[0] - dt.timedelta(days=30), days[-1])
    prev_of = {d: cal[i - 1] for i, d in enumerate(cal) if i}

    with out_path.open("w") as f:
        for day in days:
            prev = prev_of.get(day)
            if prev is None:
                print(f"{day}  no previous trading day -- skipped", flush=True)
                continue
            univ = universe_for(cur, prev, args.size)
            bars = fetch(cur, day, sorted(set(univ) | {MARKET}))
            mkt = bars.get(MARKET)
            have = [bars[c] for c in univ if c in bars]
            for act_min, n_pool in sorted(anchors[day].items()):
                rets = [r for d in have if (r := d.ret(act_min)) is not None]
                if not rets:
                    continue
                exp = [r for d in have if (r := d.ret_1520(act_min)) is not None]
                f.write(
                    json.dumps(
                        {
                            "day": str(day),
                            "act_min": act_min,
                            "n_univ": len(rets),
                            "null_mean": statistics.fmean(rets),
                            "null_med": statistics.median(rets),
                            "null_mean_1520": statistics.fmean(exp) if exp else None,
                            "mkt_ret": mkt.ret(act_min) if mkt else None,
                            "mkt_ret_1520": mkt.ret_1520(act_min) if mkt else None,
                            "n_pool": n_pool,
                        }
                    )
                    + "\n"
                )
            print(
                f"{day}  prev={prev}  univ={len(univ)} bars={len(have)} "
                f"anchors={len(anchors[day])}",
                flush=True,
            )
    conn.close()
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
