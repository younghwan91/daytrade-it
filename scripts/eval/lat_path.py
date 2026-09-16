"""Where does the move happen -- before we act, or after?

For every pair in the 125-day sweep pool, reprice the same stock from the
ARTICLE'S PUBLICATION MINUTE rather than from our decision minute, and walk
it forward. Against the same matched null `edge_null.py` / `edge_strata.py`
built (day x minute x volatility quartile x liquidity quartile), so the day,
the minute and the name's own character are differenced out at every horizon.

Entry/exit rules are `prompt_labels.py`'s, copied verbatim from
`edge_null.py`'s `Day` class -- SimBroker's quote at a minute is that minute's
bar open, else the last completed regular bar's close.

READ-ONLY on kr_quant.

Output: one JSON row per pair with the pool return and the matched-null return
from the publication minute at each horizon, so alpha(k) can be aggregated
day-clustered downstream.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_db import connect  # noqa: E402

OPEN = dt.time(9, 0)
REGULAR_END = dt.time(15, 33)
SLOTS = 394
UNIVERSE_SIZE = 300
#: horizons in minutes after the anchor; None means "to the day's last close"
HORIZONS = (1, 2, 3, 5, 10, 15, 22, 30, 60, 120)
#: entry lags after publication, for the "how fast would we have to be" ladder
LAGS = (0, 1, 2, 3, 5, 10, 15, 22, 30, 45, 60, 90)


def _slot(ts: dt.datetime) -> int:
    return (ts.hour - 9) * 60 + ts.minute


def build_entry(bars: list[tuple]) -> tuple[np.ndarray, float]:
    """(entry price per slot, last regular close). NaN where unquotable."""
    opens = np.full(SLOTS, np.nan)
    closes = np.full(SLOTS, np.nan)
    for ts, o, c in bars:
        t = ts.time()
        if t < OPEN or t > REGULAR_END:
            continue
        i = _slot(ts)
        if 0 <= i < SLOTS:
            opens[i] = float(o)
            closes[i] = float(c)
    entry = np.full(SLOTS, np.nan)
    prev = np.nan
    for i in range(SLOTS):
        entry[i] = opens[i] if not np.isnan(opens[i]) else prev
        if not np.isnan(closes[i]):
            prev = closes[i]
    fin = np.nan
    for i in range(SLOTS - 1, -1, -1):
        if not np.isnan(closes[i]):
            fin = closes[i]
            break
    return entry, fin


def trading_days(cur, lo, hi):
    cur.execute(
        "SELECT DISTINCT date FROM daily_bars_adjusted WHERE date >= %s AND date <= %s ORDER BY date",
        (lo, hi),
    )
    return [r[0] for r in cur.fetchall()]


def universe_for(cur, prev, size):
    cur.execute("SELECT code, trade_value FROM daily_bars_adjusted WHERE date = %s", (prev,))
    vals = {
        c: float(v) for c, v in cur.fetchall() if v and float(v) > 0 and len(c) == 6 and c.isdigit()
    }
    return sorted(vals, key=lambda c: (-vals[c], c))[:size]


def fetch(cur, day, codes):
    cur.execute(
        "SELECT code, ts, open, close FROM minute_bars "
        "WHERE code = ANY(%s) AND ts >= %s AND ts < %s ORDER BY code, ts",
        (list(codes), day, day + dt.timedelta(days=1)),
    )
    raw = defaultdict(list)
    for code, ts, o, c in cur.fetchall():
        raw[code].append((ts, o, c))
    return raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/eval/prompt_v3/labels.jsonl")
    ap.add_argument("--master", default="data/eval/prompt_v3/master.jsonl")
    ap.add_argument("--strata", default="data/eval/edge/strata_sweep.jsonl")
    ap.add_argument("--out", default="data/eval/latency/path_sweep.jsonl")
    ap.add_argument("--min-cell", type=int, default=5)
    ap.add_argument("--size", type=int, default=UNIVERSE_SIZE)
    ap.add_argument("--days", type=int, default=0, help="limit to the first N days (smoke test)")
    args = ap.parse_args()

    lab = {}
    with open(args.labels) as f:
        for line in f:
            r = json.loads(line)
            lab[r["key"]] = r

    pairs_by_day = defaultdict(list)
    with open(args.master) as f:
        for line in f:
            m = json.loads(line)
            label = lab.get(m["key"])
            if label is None or not m.get("published_at"):
                continue
            pub = dt.datetime.fromisoformat(m["published_at"])
            day = dt.date.fromisoformat(label["day"])
            pub_slot = _slot(pub) if pub.date() == day else -1
            # pre-market / prior-day publication: the earliest tradable anchor
            # is the open, which is exactly what live could have done.
            if pub.date() != day or pub_slot < 0:
                pub_slot = 0
                premarket = True
            else:
                premarket = pub_slot == 0
            if pub_slot >= SLOTS:
                continue
            v2 = m.get("v2") or {}
            pairs_by_day[day].append(
                {
                    "key": m["key"],
                    "code": m.get("code") or m["key"].split("|")[1],
                    "pub_min": int(pub_slot),
                    "act_min": int(label["act_min"]),
                    "premarket": bool(premarket),
                    "score": v2.get("score"),
                }
            )

    strat = defaultdict(dict)
    with open(args.strata) as f:
        for line in f:
            s = json.loads(line)
            strat[s["day"]][s["code"]] = (int(s["sig_q"]), int(s["val_q"]))

    days = sorted(pairs_by_day)
    if args.days:
        days = days[: args.days]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = connect()
    cur = conn.cursor()
    cal = trading_days(cur, days[0] - dt.timedelta(days=30), days[-1])

    written = 0
    with out_path.open("w") as out:
        for day in days:
            i = cal.index(day) if day in cal else -1
            if i <= 0:
                continue
            univ = universe_for(cur, cal[i - 1], args.size)
            pool = pairs_by_day[day]
            codes = sorted(set(univ) | {p["code"] for p in pool})
            raw = fetch(cur, day, codes)
            E, F = {}, {}
            for c in codes:
                if c in raw:
                    E[c], F[c] = build_entry(raw[c])

            ucodes = [c for c in univ if c in E]
            if len(ucodes) < 50:
                continue
            EU = np.vstack([E[c] for c in ucodes])  # (n, SLOTS)
            FU = np.array([F[c] for c in ucodes])  # (n,)
            sday = strat.get(day.isoformat(), {})
            cell = np.array([sday[c][0] * 4 + sday[c][1] if c in sday else -1 for c in ucodes])

            anchors = set()
            for p in pool:
                anchors.add(p["act_min"])
                for L in LAGS:
                    anchors.add(min(p["pub_min"] + L, SLOTS - 1))
            anchors = sorted(anchors)
            nulls: dict[int, dict] = {}
            for m in anchors:
                base = EU[:, m]
                with np.errstate(invalid="ignore", divide="ignore"):
                    rows = {}
                    for k in HORIZONS:
                        j = min(m + k, SLOTS - 1)
                        rows[k] = EU[:, j] / base - 1
                    rows["close"] = FU / base - 1
                per_cell = {}
                for cid in range(16):
                    sel = cell == cid
                    if sel.sum() == 0:
                        continue
                    e = {}
                    for k, v in rows.items():
                        vv = v[sel]
                        vv = vv[np.isfinite(vv)]
                        if len(vv) >= args.min_cell:
                            e[str(k)] = float(vv.mean())
                    if e:
                        e["n"] = int(np.isfinite(rows["close"][sel]).sum())
                        per_cell[cid] = e
                allc = {}
                for k, v in rows.items():
                    vv = v[np.isfinite(v)]
                    if len(vv) >= args.min_cell:
                        allc[str(k)] = float(vv.mean())
                nulls[m] = {"cells": per_cell, "all": allc}

            for p in pool:
                c = p["code"]
                if c not in E:
                    continue
                e = E[c]
                fin = F[c]
                pm, am = p["pub_min"], p["act_min"]
                if not np.isfinite(e[pm]) or not np.isfinite(fin):
                    continue
                cid = sday[c][0] * 4 + sday[c][1] if c in sday else -1
                own = {}
                for k in HORIZONS:
                    j = min(pm + k, SLOTS - 1)
                    own[str(k)] = float(e[j] / e[pm] - 1) if np.isfinite(e[j]) else None
                own["close"] = float(fin / e[pm] - 1)
                own["act"] = float(e[am] / e[pm] - 1) if am < SLOTS and np.isfinite(e[am]) else None
                # the latency ladder: enter at pub+L, hold to the close,
                # each against the null at that very minute and cell.
                fwd = {}
                for L in LAGS:
                    m = min(pm + L, SLOTS - 1)
                    if not np.isfinite(e[m]):
                        continue
                    nb_l = nulls.get(m, {})
                    nl = (nb_l.get("cells", {}).get(cid) or {}).get("close")
                    fwd[str(L)] = {
                        "own": float(fin / e[m] - 1),
                        "null": nl,
                        "null_all": nb_l.get("all", {}).get("close"),
                    }

                nb = nulls.get(pm, {})
                ncell = nb.get("cells", {}).get(cid)
                nall = nb.get("all", {})
                na = nulls.get(am, {})
                out.write(
                    json.dumps(
                        {
                            "key": p["key"],
                            "day": day.isoformat(),
                            "code": c,
                            "pub_min": pm,
                            "act_min": am,
                            "lag": am - pm,
                            "premarket": p["premarket"],
                            "score": p["score"],
                            "cell": int(cid),
                            "own": own,
                            "null_cell": ncell,
                            "null_all": nall,
                            "act_null_cell": na.get("cells", {}).get(cid),
                            "fwd": fwd,
                            "act_own_close": (
                                float(fin / e[am] - 1)
                                if am < SLOTS and np.isfinite(e[am])
                                else None
                            ),
                        }
                    )
                    + "\n"
                )
                written += 1
            print(f"{day} pool={len(pool)} univ={len(ucodes)} written={written}", flush=True)
    print("rows:", written)


if __name__ == "__main__":
    main()
