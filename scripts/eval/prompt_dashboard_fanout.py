"""Does the fan-out money-tightening transfer to the pool live actually trades?

Everything in `2026-09-16-prompt-v3-oracle-mining.md` is measured on the SWEEP
pool, which loses money. Live trades the DASHBOARD pool, which does not. A rule
derived on the first does not license a change to the second, so this labels the
dashboard pool the same way and asks the same question there.

Sources, and their limits, stated up front because they bound every number below:

- Dashboard pairs: `data/eval/replay_events/*.jsonl` -- 7 days, 2026-09-07..09-15.
  Only 09-14 and 09-15 are true live-archive days (`meta.json: archive_days`);
  09-07..09-11 are reconstructed from kr_quant articles on the daemon's poll grid.
- Fan-out: live reads the detail response's `companyCodes`. That is archived only
  for 09-14..09-16 (`data/toss_detail/`, copied here as `live_fanout.json`). For
  the other days the only available proxy is the number of pairs sharing an
  `article_id` inside the dashboard pool -- which counts codes *after* universe
  narrowing and is therefore a LOWER bound on live's fan-out. A threshold applied
  to the proxy drops fewer pairs than the same threshold live, so the proxy
  understates the rule's reach.

READ-ONLY on kr_quant. No API, no Toss.
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import psycopg2

EVENTS = Path("data/eval/replay_events")
LIVE_FANOUT = Path("/tmp/live_fanout.json")
COST = 0.00215
MARKET = "069500"
OPEN = dt.time(9, 0)
REGULAR_END = dt.time(15, 33)


def load_pairs() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(EVENTS / "*.jsonl"))):
        day = os.path.basename(path)[:-6]
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                r["day"] = day
                rows.append(r)
    return rows


def label(rows: list[dict]) -> dict[str, dict]:
    """Same entry/exit rule as prompt_labels.py, on the dashboard pairs."""
    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r)
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5)
    conn.set_session(readonly=True, autocommit=True)
    out: dict[str, dict] = {}
    with conn.cursor() as cur:
        for day, rs in sorted(by_day.items()):
            d = dt.date.fromisoformat(day)
            codes = sorted({r["ticker"] for r in rs} | {MARKET})
            cur.execute(
                "SELECT code, ts, open, high, low, close FROM minute_bars "
                "WHERE code = ANY(%s) AND ts >= %s AND ts < %s ORDER BY code, ts",
                (codes, d, d + dt.timedelta(days=1)),
            )
            bars: dict[str, list] = defaultdict(list)
            for code, ts, o, h, lo, c in cur.fetchall():
                bars[code].append((ts, float(o), float(h), float(lo), float(c)))
            for r in rs:
                bs = bars.get(r["ticker"])
                if not bs:
                    continue
                dec = dt.datetime.fromisoformat(r["decide_at"]).replace(tzinfo=None)
                minute = dec.replace(second=0, microsecond=0)
                if minute.time() < OPEN:
                    minute = dt.datetime.combine(d, OPEN)
                entry = next((b[1] for b in bs if b[0] == minute), None)
                if entry is None:
                    done = [b for b in bs if b[0] < minute and b[0].time() >= OPEN]
                    entry = done[-1][4] if done else None
                if not entry or entry <= 0:
                    continue
                fwd = [b for b in bs if minute <= b[0] and OPEN <= b[0].time() <= REGULAR_END]
                if not fwd:
                    continue
                out[r["key"]] = {
                    "ret_close": fwd[-1][4] / entry - 1,
                    "mfe": max(b[2] for b in fwd) / entry - 1,
                }
    conn.close()
    return out


def main() -> None:
    rows = load_pairs()
    days = sorted({r["day"] for r in rows})
    print(f"dashboard pairs: {len(rows)} over {len(days)} days ({days[0]}..{days[-1]})")

    live_fo = json.loads(LIVE_FANOUT.read_text()) if LIVE_FANOUT.exists() else {}
    in_pool: Counter[str] = Counter(r["article_id"] for r in rows)
    n_live = sum(1 for r in rows if r["article_id"] in live_fo)
    print(
        f"pairs with a true live fan-out (companyCodes archived): {n_live} "
        f"({n_live / len(rows):.1%})"
    )

    # how good is the in-pool proxy where both exist?
    both = [
        (in_pool[r["article_id"]], live_fo[r["article_id"]]["company"])
        for r in rows
        if r["article_id"] in live_fo
    ]
    if both:
        agree = sum(1 for a, b in both if a == b)
        under = sum(1 for a, b in both if a < b)
        print(
            f"proxy vs live fan-out on {len(both)} pairs: equal {agree}, "
            f"proxy lower {under}, proxy higher {len(both) - agree - under}"
        )

    lab = label(rows)
    print(f"labelled: {len(lab)} / {len(rows)}")
    pool = [lab[r["key"]]["ret_close"] - COST for r in rows if r["key"] in lab]
    print(
        f"dashboard pool net mean {st.mean(pool) * 1e4:+.0f} bp   "
        f"win {sum(1 for x in pool if x > 0) / len(pool):.1%}   n={len(pool)}"
    )

    buys = [r for r in rows if r["key"] in lab and (r["features"].get("score") or 0) > 0.15]
    br = [lab[r["key"]]["ret_close"] - COST for r in buys]
    print(
        f"dashboard BUYs (score>0.15): n={len(br)}  mean {st.mean(br) * 1e4:+.0f} bp  "
        f"win {sum(1 for x in br if x > 0) / len(br):.1%}"
    )

    def fanout_of(r: dict) -> int:
        e = live_fo.get(r["article_id"])
        return e["company"] if e else in_pool[r["article_id"]]

    print("\n=== fan-out tightening on the DASHBOARD pool ===")
    print(
        f"{'threshold':14s} {'kept':>6s} {'kept %':>7s} {'kept mean':>10s} "
        f"{'dropped':>8s} {'dropped mean':>13s}"
    )
    for n in (None, 15, 11, 8, 5, 3, 2, 1):
        kept = [x for x in buys if n is None or fanout_of(x) <= n]
        drop = [x for x in buys if not (n is None or fanout_of(x) <= n)]
        kr = [lab[x["key"]]["ret_close"] - COST for x in kept]
        dr = [lab[x["key"]]["ret_close"] - COST for x in drop]
        name = "no filter" if n is None else f"fan-out <= {n}"
        print(
            f"{name:14s} {len(kr):6d} {len(kr) / len(buys):7.1%} "
            f"{(st.mean(kr) * 1e4 if kr else float('nan')):+10.0f} {len(dr):8d} "
            f"{(st.mean(dr) * 1e4 if dr else float('nan')):+13.0f}"
        )

    print("\n=== the same, restricted to the 2 true live-archive days ===")
    arch = {"2026-09-14", "2026-09-15"}
    abuys = [r for r in buys if r["day"] in arch]
    print(f"BUYs on those days: {len(abuys)}")
    for n in (None, 11, 5, 2, 1):
        kept = [x for x in abuys if n is None or fanout_of(x) <= n]
        kr = [lab[x["key"]]["ret_close"] - COST for x in kept]
        name = "no filter" if n is None else f"fan-out <= {n}"
        print(
            f"  {name:14s} kept {len(kr):4d}  "
            f"mean {(st.mean(kr) * 1e4 if kr else float('nan')):+8.0f} bp"
        )


if __name__ == "__main__":
    main()
