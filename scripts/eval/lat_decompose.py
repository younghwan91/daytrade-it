"""Decompose publication -> order latency on the live archive (2026-09-14..16).

Sources, all live archive, nothing reconstructed:
  ~/latency_archive/news_judgments/YYYY-MM-DD.jsonl  published_at, judged_at per pair
  ~/latency_archive/toss_detail/YYYY-MM-DD/<id>.json  mtime = when our loop reached the article
  ~/latency_archive/news_signal_daemon.log            poll-complete / signal / order events

Segments per article:
  A  published_at -> seen_at      upstream (Toss dashboard appearance + our poll grid)
  B  seen_at      -> queued_at    our own in-poll queue (article order within the batch)
  C  queued_at    -> judged_at    detail fetch + Claude v2 (+ v4 shadow) call
  D  judged_at    -> signal_at    signal generation + entry filter + DB persist
  E  signal_at    -> order_at     AutoTrader submit
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import statistics as st
from pathlib import Path

ARCH = Path.home() / "latency_archive"
KST = dt.timezone(dt.timedelta(hours=9))
DAYS = ["2026-09-14", "2026-09-15", "2026-09-16"]


def q(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, int(round(p * (len(xs) - 1))))
    return xs[i]


def summarize(name, xs, unit="min"):
    if not xs:
        print(f"{name:34s} n=0")
        return
    d = 60.0 if unit == "min" else 1.0
    print(
        f"{name:34s} n={len(xs):5d}  med {q(xs, 0.5) / d:8.2f}  mean {st.mean(xs) / d:8.2f}  "
        f"p25 {q(xs, 0.25) / d:8.2f}  p75 {q(xs, 0.75) / d:8.2f}  p90 {q(xs, 0.90) / d:8.2f}"
    )


# ---- log events -------------------------------------------------------------
polls: dict[str, list[dt.datetime]] = {d: [] for d in DAYS}
sig_by_title: list[tuple[dt.datetime, str, str]] = []  # (ts, ticker, title)
ord_by_ticker: list[tuple[dt.datetime, str]] = []

for line in (ARCH / "news_signal_daemon.log").open():
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    ev = r.get("event", "")
    ts = r.get("timestamp")
    if not ts:
        continue
    t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(KST)
    day = t.date().isoformat()
    if ev == "Toss news poll complete" and day in polls:
        polls[day].append(t)
    elif ev == "news_signal_poller: ACTIONABLE SIGNAL PERSISTED":
        sig_by_title.append((t, r.get("ticker", ""), r.get("article_title", "")))
    elif ev == "auto_trader: ENTRY SUBMITTED":
        ord_by_ticker.append((t, r.get("ticker", "")))

for d in DAYS:
    polls[d].sort()
print("poll-complete events per day:", {d: len(polls[d]) for d in DAYS})
for d in DAYS:
    ps = polls[d]
    gaps = [(ps[i + 1] - ps[i]).total_seconds() for i in range(len(ps) - 1)]
    if gaps:
        print(
            f"  {d}  first {ps[0].time()}  last {ps[-1].time()}  "
            f"gap med {q(gaps, 0.5):.0f}s p90 {q(gaps, 0.9):.0f}s"
        )

# ---- detail mtimes ----------------------------------------------------------
queued: dict[str, dt.datetime] = {}
for d in DAYS:
    for f in glob.glob(str(ARCH / "toss_detail" / d / "*.json")):
        aid = os.path.basename(f)[:-5]
        queued[aid] = dt.datetime.fromtimestamp(os.path.getmtime(f), KST)

# ---- judgments --------------------------------------------------------------
rows = []
for d in DAYS:
    p = ARCH / "news_judgments" / f"{d}.jsonl"
    if not p.exists():
        continue
    for line in p.open():
        r = json.loads(line)
        r["day"] = d
        r["pub"] = dt.datetime.fromisoformat(r["published_at"]).astimezone(KST)
        r["judged"] = dt.datetime.fromisoformat(r["judged_at"]).astimezone(KST)
        rows.append(r)

print(
    f"\npairs in judgment log: {len(rows)}  "
    f"(prefilter-skipped {sum(1 for r in rows if r.get('prefilter_skipped'))})"
)

# first judged per article (proxy for when the article entered scoring)
first_judged: dict[str, dt.datetime] = {}
for r in rows:
    a = r["article_id"]
    if a not in first_judged or r["judged"] < first_judged[a]:
        first_judged[a] = r["judged"]

# article-level table
arts = {}
for r in rows:
    a = r["article_id"]
    if a not in arts:
        arts[a] = {"day": r["day"], "pub": r["pub"], "title": r["title"]}

print(
    f"distinct articles: {len(arts)}   with archived detail: {sum(1 for a in arts if a in queued)}"
)


def seen_at(day, t):
    """Last poll-complete at or before t (the poll that emitted this article)."""
    ps = polls[day]
    prev = None
    for p in ps:
        if p <= t:
            prev = p
        else:
            break
    return prev


def prev_poll(day, t):
    """The poll before the one that emitted it: the article was NOT on the
    dashboard at this instant (or we would have taken it then)."""
    ps = polls[day]
    prev, prev2 = None, None
    for p in ps:
        if p <= t:
            prev2, prev = prev, p
        else:
            break
    return prev2


A, B, C, tot = [], [], [], []
A_intraday, tot_intraday = [], []
toss_lag_lo, toss_lag_hi = [], []
premarket = 0
for aid, a in arts.items():
    qt = queued.get(aid) or first_judged.get(aid)
    if qt is None:
        continue
    s = seen_at(a["day"], qt)
    if s is None:
        continue
    j = first_judged[aid]
    pub = a["pub"]
    open_t = dt.datetime.combine(dt.date.fromisoformat(a["day"]), dt.time(9, 0, 30), tzinfo=KST)
    is_pre = pub < open_t
    if is_pre:
        premarket += 1
    A.append((s - pub).total_seconds())
    B.append((qt - s).total_seconds())
    C.append((j - qt).total_seconds())
    tot.append((j - pub).total_seconds())
    if not is_pre:
        A_intraday.append((s - pub).total_seconds())
        tot_intraday.append((j - pub).total_seconds())
        p2 = prev_poll(a["day"], qt)
        if p2 is not None and p2 > pub:
            # article was published before p2 yet not emitted until s:
            # the whole gap is upstream (Toss had not shown it yet at p2).
            toss_lag_lo.append((p2 - pub).total_seconds())
            toss_lag_hi.append((s - pub).total_seconds())
        elif p2 is not None:
            toss_lag_lo.append(0.0)
            toss_lag_hi.append((s - pub).total_seconds())

print(
    f"\narticles usable: {len(A)}   pre-09:00 {premarket} "
    f"({100 * premarket / max(1, len(A)):.0f}%)   intraday {len(A_intraday)}"
)
print("\n--- ALL ARTICLES (minutes) ---")
summarize("A published -> we saw (poll)", A)
summarize("B we saw -> our loop reached it", B)
summarize("C reached -> judged", C)
summarize("TOTAL published -> judged", tot)
print("\n--- INTRADAY ARTICLES ONLY (published after 09:00:30) ---")
summarize("A published -> we saw (poll)", A_intraday)
summarize("TOTAL published -> judged", tot_intraday)
print("\n  upstream (Toss dashboard appearance) bound, intraday:")
summarize("  lower bound", toss_lag_lo)
summarize("  upper bound", toss_lag_hi)

# ---- pair-level: judged -> signal -> order ---------------------------------
sig_idx = {}
for t, tk, title in sig_by_title:
    sig_idx.setdefault((tk, title[:80]), []).append(t)
D, E = [], []
for r in rows:
    if r.get("prefilter_skipped"):
        continue
    k = (r["ticker"], (r["title"] or "")[:80])
    for t in sig_idx.get(k, []):
        if 0 <= (t - r["judged"]).total_seconds() < 120:
            D.append((t - r["judged"]).total_seconds())
            for ot, otk in ord_by_ticker:
                dtsec = (ot - t).total_seconds()
                if otk == r["ticker"] and 0 <= dtsec < 300:
                    E.append(dtsec)
                    break
            break
print("\n--- SIGNAL / ORDER TAIL (seconds) ---")
summarize("D judged -> signal persisted", D, unit="s")
summarize("E signal -> order submitted", E, unit="s")

# ---- pair-level publish->judge, matching the team-lead's 21.8 min ----------
pair_tot = [(r["judged"] - r["pub"]).total_seconds() for r in rows]
print("\n--- PAIR-LEVEL published -> judged (minutes) ---")
summarize("all pairs", pair_tot)
pj = [
    (r["judged"] - r["pub"]).total_seconds()
    for r in rows
    if r["pub"]
    >= dt.datetime.combine(dt.date.fromisoformat(r["day"]), dt.time(9, 0, 30), tzinfo=KST)
]
summarize("intraday pairs", pj)
