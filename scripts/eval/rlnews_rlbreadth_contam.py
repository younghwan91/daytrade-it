"""Contamination audit for the ten v2 judgment fields.

Three checks per field:
  1. provenance  -- value is a function of text that exists before decide_at
  2. backward    -- association with the pre-entry move (benign if strong)
  3. forward-leak-- association with the FORWARD market-index return from the
                    acting minute to the close. No honest article-level field
                    can know that; a non-zero loading is the leak signature.
Day-clustered standard errors throughout (articles cluster hard by day).
"""

from __future__ import annotations

import json

import numpy as np

REPO = "/home/young/git/daytrade-it-rlnews"
MASTER = f"{REPO}/data/eval/prompt_v3/master.jsonl"
LABELS = f"{REPO}/data/eval/prompt_v3/labels.jsonl"

FIELDS = [
    "subject",
    "reports_price_move",
    "material_type",
    "persistence",
    "specificity",
    "freshness",
    "surprise",
    "sentiment_direction",
    "sentiment_strength",
    "scale_vs_size",
]


def jl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


lab = {r["key"]: r for r in jl(LABELS)}
rows = []
for m in jl(MASTER):
    r = lab.get(m["key"])
    if r is None or not m.get("v2"):
        continue
    rows.append((m, r))
print("pairs with label+v2:", len(rows))

days = np.array([r["day"] for _, r in rows])
pre = np.array([(r.get("pre_move") if r.get("pre_move") is not None else np.nan) for _, r in rows])
fwd = np.array([r["ret_close"] for _, r in rows])
mkt_fwd = np.array(
    [(r["mkt_close"] if r.get("mkt_close") is not None else np.nan) for _, r in rows]
)
exc = np.array([(r["exc_close"] if r.get("exc_close") is not None else np.nan) for _, r in rows])
act = np.array([r["act_min"] for _, r in rows], dtype=float)

# The index's forward return depends mechanically on how long we hold, so a
# field correlated with act_min would show a spurious loading. Remove the
# act_min-bucket mean first; what is left is "does this field know which DAY
# the market goes up", which no honest article-level field can know.
bucket = np.minimum((act // 30).astype(int), 12)
mkt_resid = mkt_fwd.copy()
for b in np.unique(bucket):
    sel = (bucket == b) & ~np.isnan(mkt_fwd)
    mkt_resid[sel] = mkt_fwd[sel] - mkt_fwd[sel].mean()


def cluster_mean_diff(mask, y):
    """Mean of y on mask minus mean off mask, with day-clustered se (block bootstrap)."""
    ok = ~np.isnan(y)
    m, y2, d2 = mask[ok], y[ok], days[ok]
    if m.sum() < 30 or (~m).sum() < 30:
        return None
    est = y2[m].mean() - y2[~m].mean()
    uds = np.unique(d2)
    idx = {d: np.where(d2 == d)[0] for d in uds}
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(300):
        pick = rng.choice(len(uds), len(uds), replace=True)
        ii = np.concatenate([idx[uds[p]] for p in pick])
        mm = m[ii]
        if mm.sum() < 5 or (~mm).sum() < 5:
            continue
        bs.append(y2[ii][mm].mean() - y2[ii][~mm].mean())
    return est, float(np.std(bs))


out = {}
for f in FIELDS:
    vals = [m["v2"].get(f) for m, _ in rows]
    uniq = sorted({str(v) for v in vals})
    ent = {}
    for u in uniq:
        mask = np.array([str(v) == u for v in vals])
        if mask.sum() < 200:
            continue
        e = {}
        for nm, y in (
            ("pre_move", pre),
            ("fwd_ret", fwd),
            ("fwd_exc", exc),
            ("LEAK_mkt_resid", mkt_resid),
        ):
            r = cluster_mean_diff(mask, y)
            if r:
                e[nm] = [round(r[0] * 1e4, 2), round(r[1] * 1e4, 2)]
        e["n"] = int(mask.sum())
        ent[u] = e
    out[f] = ent

print(json.dumps(out, ensure_ascii=False, indent=1))
