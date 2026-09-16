"""What does the book actually cost us, and how much of it is the 09:00 open?

The cost model assumes 0.05% one-way slippage on KOSPI and 0.15% on KOSDAQ.
Those are configuration values nobody has checked, and they are the largest
controllable line in the toll. The `quotes` table now holds 18 trading days of
full 10-level depth, so the spread can be measured instead of assumed.

Two questions, both about execution rather than alpha:

  1. How wide is the touch, by time of day and by market? We send market
     orders into the 09:00 open, which is where the book should be thinnest.
     Crossing a spread of `s` costs `s` per round trip (buy the ask, sell the
     bid), so the measured spread IS the slippage line.
  2. Is there room at the touch for our order? A 100,000 KRW order that
     exhausts bid1/ask1 walks the book and pays more than the touch. Depth at
     the touch is reported in KRW so the comparison is direct.

Aggregated in SQL so the 8.9M rows never come back over the wire.

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

BUCKET = """
  CASE
    WHEN ts::time <  time '09:05' THEN '09:00-09:04'
    WHEN ts::time <  time '09:30' THEN '09:05-09:29'
    WHEN ts::time <  time '10:00' THEN '09:30-09:59'
    WHEN ts::time <  time '11:00' THEN '10:00-10:59'
    WHEN ts::time <  time '13:00' THEN '11:00-12:59'
    WHEN ts::time <= time '15:20' THEN '13:00-15:20'
    ELSE 'after'
  END
"""

SQL = f"""
SELECT {BUCKET} AS bucket,
       count(*)                                                        AS n,
       avg((ask1 - bid1)::float8 / ((ask1 + bid1)::float8 / 2))        AS spread,
       percentile_cont(0.5) WITHIN GROUP (
           ORDER BY (ask1 - bid1)::float8 / ((ask1 + bid1)::float8 / 2)) AS med,
       avg(bid1::float8 * bidqty1)                                     AS bid_krw,
       avg(ask1::float8 * askqty1)                                     AS ask_krw
FROM quotes
WHERE ts >= %s AND ts < %s
  AND bid1 > 0 AND ask1 > bid1
  AND ts::time >= time '09:00' AND ts::time <= time '15:20'
  AND code = ANY(%s)
GROUP BY 1 ORDER BY 1
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
        print(f"\n=== {label}   ({len(codes)} codes ever judged)   {args.start}..{args.end}")
        print(
            f"{'time':<14}{'quotes':>12}{'spread bp':>11}{'median':>9}"
            f"{'bid1 KRW':>12}{'ask1 KRW':>12}{'fits 100k?':>12}"
        )
        for bucket, n, spread, med, bid_krw, ask_krw in rows:
            room = min(bid_krw or 0, ask_krw or 0)
            print(
                f"{bucket:<14}{n:>12,}{spread * BP:>11.1f}{med * BP:>9.1f}"
                f"{bid_krw:>12,.0f}{ask_krw:>12,.0f}"
                f"{('yes' if room >= NOTIONAL else 'NO'):>12}"
            )
    conn.close()
    print(
        "\ncrossing the touch costs one spread per round trip; the cost model "
        "assumes 10.0 bp (거래소) and 30.0 bp (코스닥)\n"
    )


if __name__ == "__main__":
    main()
