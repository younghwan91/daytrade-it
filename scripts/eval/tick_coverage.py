"""Do we have tick and order-book data for the trades we actually made?

`ticks` and `quotes` cover 2026-08-24..09-16 for ~850 codes, which is not the
whole market -- so the question that bounds every tick-based feature is whether
they cover the names the live policy bought on the days it bought them. This
checks the 154 dashboard-pool entries directly rather than reasoning from row
counts.

kr_quant read-only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import psycopg2

SQL = """
WITH entries AS (SELECT * FROM (VALUES {vals}) AS t(code, d))
SELECT e.d::text AS day,
       count(*) AS entries,
       count(*) FILTER (WHERE tk.n > 0) AS with_ticks,
       count(*) FILTER (WHERE qn.n > 0) AS with_quotes,
       coalesce(round(avg(tk.n) FILTER (WHERE tk.n > 0)), 0) AS avg_ticks,
       coalesce(round(avg(qn.n) FILTER (WHERE qn.n > 0)), 0) AS avg_quotes
FROM entries e
LEFT JOIN LATERAL (
    SELECT count(*) AS n FROM ticks t
    WHERE t.code = e.code AND t.ts >= e.d::timestamp AND t.ts < e.d::timestamp + interval '1 day'
) tk ON true
LEFT JOIN LATERAL (
    SELECT count(*) AS n FROM quotes s
    WHERE s.code = e.code AND s.ts >= e.d::timestamp AND s.ts < e.d::timestamp + interval '1 day'
) qn ON true
GROUP BY e.d
ORDER BY e.d
"""


def main() -> None:
    rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line]
    pairs = sorted({(r["ticker"], r["day"]) for r in rows})
    vals = ", ".join(["(%s, %s::date)"] * len(pairs))
    params = [x for p in pairs for x in p]
    with psycopg2.connect(os.environ["KR_QUANT_DB"]) as c, c.cursor() as cur:
        cur.execute("SET default_transaction_read_only = on")
        cur.execute(SQL.format(vals=vals), params)
        out = cur.fetchall()
    print(f"{'day':12s} {'entries':>8s} {'ticks':>7s} {'quotes':>7s} {'avg tk':>9s} {'avg q':>9s}")
    tot_n = tot_t = tot_q = 0
    for day, n, t, q, at, aq in out:
        print(f"{day:12s} {n:>8d} {t:>7d} {q:>7d} {int(at):>9,d} {int(aq):>9,d}")
        tot_n, tot_t, tot_q = tot_n + n, tot_t + t, tot_q + q
    print(
        f"{'TOTAL':12s} {tot_n:>8d} {tot_t:>7d} {tot_q:>7d}"
        f"   ticks {tot_t / tot_n:.1%}  quotes {tot_q / tot_n:.1%}"
    )


if __name__ == "__main__":
    main()
