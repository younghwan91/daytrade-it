"""Is the Toss dashboard lag uniform, or does some news arrive fast?

If a subset (breaking, 특징주, a particular wire) appears on the dashboard in
two minutes rather than eighteen, that subset is where a low-latency strategy
would live. Measured on the live archive only.
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ARCH = Path.home() / "latency_archive"
KST = dt.timezone(dt.timedelta(hours=9))
DAYS = ["2026-09-14", "2026-09-15", "2026-09-16"]
TAG = re.compile(r"^\s*\[([^\]]{1,16})\]")

polls = defaultdict(list)
for line in (ARCH / "news_signal_daemon.log").open():
    if not line.startswith("{"):
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get("event") != "Toss news poll complete":
        continue
    t = dt.datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).astimezone(KST)
    polls[t.date().isoformat()].append(t)
for d in polls:
    polls[d].sort()

queued = {}
src = {}
for d in DAYS:
    for f in glob.glob(str(ARCH / "toss_detail" / d / "*.json")):
        aid = os.path.basename(f)[:-5]
        queued[aid] = dt.datetime.fromtimestamp(os.path.getmtime(f), KST)
        try:
            with open(f) as fh:
                j = json.load(fh)
            nid = j.get("news_id") or ""
            src[aid] = nid.split("_")[0] if "_" in nid else "na"
        except Exception:
            src[aid] = "na"

arts = {}
for d in DAYS:
    p = ARCH / "news_judgments" / f"{d}.jsonl"
    if not p.exists():
        continue
    for line in p.open():
        r = json.loads(line)
        a = r["article_id"]
        if a in arts:
            continue
        arts[a] = {
            "day": d,
            "pub": dt.datetime.fromisoformat(r["published_at"]).astimezone(KST),
            "title": r["title"] or "",
            "cat": r.get("category") or "na",
        }


def seen_at(day, t):
    prev = None
    for p in polls[day]:
        if p <= t:
            prev = p
        else:
            break
    return prev


groups = defaultdict(list)
for aid, a in arts.items():
    qt = queued.get(aid)
    if qt is None:
        continue
    s = seen_at(a["day"], qt)
    if s is None:
        continue
    openp = dt.datetime.combine(dt.date.fromisoformat(a["day"]), dt.time(9, 0, 30), tzinfo=KST)
    if a["pub"] < openp:
        continue  # pre-market: the lag is the market being shut, not Toss
    lag = (s - a["pub"]).total_seconds() / 60
    m = TAG.match(a["title"])
    groups[("tag", m.group(1) if m else "none")].append(lag)
    groups[("category", a["cat"])].append(lag)
    groups[("source", src.get(aid, "na"))].append(lag)
    groups[("all", "all")].append(lag)

for kind in ("all", "category", "tag", "source"):
    print(f"\n--- lag by {kind} (minutes, intraday articles only) ---")
    rows = [(k[1], v) for k, v in groups.items() if k[0] == kind and len(v) >= 4]
    for name, v in sorted(rows, key=lambda x: st.median(x[1])):
        v = sorted(v)
        print(
            f"  {name:16s} n={len(v):4d}  med {v[len(v) // 2]:6.1f}  "
            f"min {v[0]:6.1f}  p25 {v[len(v) // 4]:6.1f}  p75 {v[3 * len(v) // 4]:6.1f}"
        )
