import collections
import datetime as dt
import json
import sys

sys.path.insert(0, "scripts/eval")
import edge_db

OUT = "/home/young/git/daytrade-it-rlnews/data/eval/rlnews"
with open(f"{OUT}/feats2_judged.jsonl", encoding="utf-8") as _fh:
    rows = [json.loads(line) for line in _fh]
by_day = collections.defaultdict(list)
for r in rows:
    by_day[r["day"]].append(r)

c = edge_db.connect()
cur = c.cursor()
cur.execute("select ts::date, code, min(ts::time), max(ts::time), count(*) from ticks group by 1,2")
tick = collections.defaultdict(dict)
for d, code, lo, hi, n in cur.fetchall():
    tick[d.isoformat()][code] = (lo, hi, n)
cur.execute("select ts::date, code, count(*) from quotes group by 1,2")
quo = collections.defaultdict(dict)
for d, code, n in cur.fetchall():
    quo[d.isoformat()][code] = n

print(
    f"{'day':<12} {'pairs':>6} {'codes':>6} {'tk_cod':>7} "
    f"{'pair_cv':>7} {'min_cv':>7} {'qt_cod':>7}"
)
tot = cov = mincov = 0
for d in sorted(by_day):
    if d not in tick:
        continue
    rs = by_day[d]
    codes = {r["code"] for r in rs}
    tk = tick[d]
    c_hit = sum(1 for r in rs if r["code"] in tk)
    m_hit = 0
    for r in rs:
        t = tk.get(r["code"])
        if not t:
            continue
        am = int(r["features"]["act_min"])
        hhmm = dt.time(9 + am // 60, am % 60) if am < 390 else dt.time(15, 29)
        if t[0] <= hhmm <= t[1]:
            m_hit += 1
    tot += len(rs)
    cov += c_hit
    mincov += m_hit
    print(
        f"{d:<12} {len(rs):>6d} {len(codes):>6d} {len(tk):>7d} "
        f"{100 * c_hit / len(rs):>6.1f}% {100 * m_hit / len(rs):>6.1f}% "
        f"{len(quo.get(d, {})):>7d}"
    )
print(
    f"overall pairs {tot}  code-day cov {100 * cov / tot:.1f}%  "
    f"acting-minute cov {100 * mincov / tot:.1f}%"
)

# accrual rate over the last 10 collected days
days = sorted(tick)
recent = days[-10:]
print("collected trading days:", len(days), days[0], "..", days[-1])
print("mean codes/day last10:", sum(len(tick[d]) for d in recent) / len(recent))
