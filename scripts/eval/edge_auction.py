"""What does the closing auction cost, and can it absorb us?

A close entry crosses the closing auction, not the continuous session, so the
14.0 / 19.2 bp afternoon figures from `edge_spread.py` do not apply to it. With
an overnight alpha of about +7 bp, an auction spread of 10 bp would eat the
conclusion whole -- so this number has to travel with any positive close-entry
result rather than be assumed.

KRX runs a call auction over the last ten minutes (15:20-15:30) with no
continuous trading, then prints at 15:30. During the call the book still
publishes, so `quotes` shows the indicative spread and depth: the question is
what the touch looks like as the auction converges, and whether the size ahead
of us is anything like the continuous-session wall (median 66 million KRW).

Reported for the last continuous minutes and for the auction window itself, by
market, so the comparison is like for like. `vi_flag` rows are counted
separately -- a volatility interruption is a different animal and should not be
averaged into a normal auction.

READ-ONLY on kr_quant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_db import connect  # noqa: E402

BP = 10_000
NOTIONAL = 100_000.0

WINDOW = """
  CASE
    WHEN ts::time <  time '15:00' THEN '3 continuous 13:00-14:59'
    WHEN ts::time <  time '15:20' THEN '4 continuous 15:00-15:19'
    WHEN ts::time <  time '15:25' THEN '5 AUCTION 15:20-15:24'
    WHEN ts::time <= time '15:30' THEN '6 AUCTION 15:25-15:30'
    ELSE '7 after'
  END
"""

SQL = f"""
SELECT {WINDOW} AS win,
       coalesce(vi_flag::text, 'none')                                   AS vi,
       count(*)                                                          AS n,
       avg((ask1 - bid1)::float8 / ((ask1 + bid1)::float8 / 2))           AS spread,
       percentile_cont(0.5) WITHIN GROUP (
           ORDER BY (ask1 - bid1)::float8 / ((ask1 + bid1)::float8 / 2))  AS med,
       avg(bid1::float8 * bidqty1)                                       AS bid_krw,
       avg(ask1::float8 * askqty1)                                       AS ask_krw
FROM quotes
WHERE ts >= %s AND ts < %s
  AND bid1 > 0 AND ask1 > bid1
  AND ts::time >= time '13:00' AND ts::time <= time '15:30'
  AND code = ANY(%s)
GROUP BY 1, 2 ORDER BY 1, 2
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="data/eval/prompt_v3/master.jsonl")
    ap.add_argument("--start", default="2026-08-24")
    ap.add_argument("--end", default="2026-09-17")
    args = ap.parse_args()

    market: dict[str, str] = {}
    with Path(args.master).open() as f:
        for line in f:
            m = json.loads(line)
            code = m.get("code") or m["key"].split("|")[1]
            market.setdefault(code, m.get("market") or "거래소")

    conn = connect()
    cur = conn.cursor()
    for label in ("거래소", "코스닥"):
        codes = sorted(c for c, mk in market.items() if mk == label)
        cur.execute(SQL, (args.start, args.end, codes))
        rows = cur.fetchall()
        if not rows:
            continue
        print(f"\n=== {label}   {args.start}..{args.end}")
        print(
            f"{'window':<26}{'vi':<6}{'quotes':>11}{'spread bp':>11}{'median':>9}"
            f"{'bid1 KRW':>14}{'fits 100k?':>12}"
        )
        for win, vi, n, spread, med, bid_krw, ask_krw in rows:
            if win.startswith("7"):
                continue
            room = min(bid_krw or 0, ask_krw or 0)
            print(
                f"{win[2:]:<26}{vi:<6}{n:>11,}{spread * BP:>11.1f}{med * BP:>9.1f}"
                f"{bid_krw:>14,.0f}{('yes' if room >= NOTIONAL else 'NO'):>12}"
            )
    conn.close()
    print(
        "\nKRX runs a call auction 15:20-15:30 and prints at the close, so the "
        "'AUCTION' rows\nare indicative quotes during the call, not executable "
        "continuous prices. A close\nentry pays the printed auction price; the "
        "spread here bounds what that costs.\n"
    )


if __name__ == "__main__":
    main()
