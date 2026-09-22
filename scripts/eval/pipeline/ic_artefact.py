"""Diagnostic: does stage-1's direction IC survive moving the reference price one bar?"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing_core as tc

S1 = Path.home() / "git/daytrade-it-rlnews/data/eval/pipeline/stage1_oof.jsonl"
T = ("mfe", "ret_close", "mfe_n", "ret_n", "rr", "lrr")
sc = {}
for line in S1.open():
    r = json.loads(line)
    if r["split"] == "selection" and r["score"] is not None:
        sc[r["key"]] = [r[f"score_{t}"] for t in T]
bars = tc.load_bars(tc.DEFAULT_PATHS, 2)
rows = []
for i in range(bars.n):
    k = str(bars.key[i])
    if k not in sc:
        continue
    mi, o, h, lo, c, v = bars.path(i)
    bm, bp = int(bars.buy_minute[i]), float(bars.buy_price[i])
    G = tc.exit_grid(mi, o, c, bm)
    if G is None or bm >= tc.ENTRY_END_MINUTE:
        continue
    flat = G[2][-1]
    nxt = np.flatnonzero(mi > bm)
    if nxt.size < 2:
        continue
    rows.append((str(bars.day[i]), bm == 540, flat / bp - 1, flat / o[nxt[0]] - 1, *sc[k]))
day = np.array([r[0] for r in rows])
op = np.array([r[1] for r in rows])
r0 = np.array([r[2] for r in rows])
r1 = np.array([r[3] for r in rows])
S = np.array([r[4:] for r in rows])


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def ic(y, s, mask):
    out = []
    for d in np.unique(day[mask]):
        m = mask & (day == d)
        if m.sum() >= 10:
            out.append(np.corrcoef(rank(y[m]), rank(s[m]))[0, 1])
    a = np.array(out)
    return f"{a.mean():+.3f}+-{a.std() / np.sqrt(a.size):.3f}"


for name, mask in (("all", np.ones(len(rows), bool)), ("0900", op), ("after0900", ~op)):
    print(name, int(mask.sum()))
    for t in range(len(T)):
        print(
            f"  {T[t]:10s} IC ref=news-minute open {ic(r0, S[:, t], mask)}   ref=next-bar open {ic(r1, S[:, t], mask)}"
        )
