"""Daily universe-level series for the regime HMM, plus the frozen daily confirm window.

Every row is one trading day d with:
  - TARGETS (day d itself, never an input): night, intra, and the mover/calm
    versions of each.
  - OBSERVABLES (all computed from data at or before the CLOSE of the previous
    trading day): the lagged legs, dispersion, breadth, realised vol, traded
    value level and change.

The decision point is the close of day d-1 (that is when an overnight leg is
bought). So an observable is admissible iff it is a function of bars up to and
including d-1's close. Every field prefixed `o_` satisfies that by construction:
it is built from the row of day d-1 or earlier.

READ-ONLY on kr_quant.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/young/git/daytrade-it-rlnews")
sys.path.insert(0, str(REPO / "data" / "eval" / "rlnews"))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from daily_horizon import load_panel, rows_by_day  # noqa: E402
from edge_db import connect  # noqa: E402
from universe_coverage import select_universe  # noqa: E402

MARKET = "069500"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-09-09")
    ap.add_argument("--end", default="2026-09-21")
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--out", default=str(REPO / "data/eval/rlnews/hmm_daily_series.jsonl"))
    args = ap.parse_args()
    lo = dt.date.fromisoformat(args.start)
    hi = dt.date.fromisoformat(args.end)

    conn = connect()
    cur = conn.cursor()
    days, codes, di, ci, opn, C, V = load_panel(cur, lo - dt.timedelta(days=60), hi)
    conn.close()
    print(f"panel {days[0]}..{days[-1]} {len(days)}d {len(codes)} codes", flush=True)

    night = np.full_like(C, np.nan)
    night[:, 1:] = opn[:, 1:] / C[:, :-1] - 1.0
    intra = C / opn - 1.0
    cc = np.full_like(C, np.nan)
    cc[:, 1:] = C[:, 1:] / C[:, :-1] - 1.0

    rbd = rows_by_day(days, codes, V)

    # per-day universe-level aggregates, indexed by day position
    agg: dict[int, dict] = {}
    for j, d in enumerate(days):
        if j == 0:
            continue
        univ = [c for c in select_universe(rbd[days[j - 1]], size=args.size) if c != MARKET]
        idx = np.array([ci[c] for c in univ if c in ci])
        if idx.size < 40:
            continue
        n = night[idx, j]
        it = intra[idx, j]
        c2c = cc[idx, j]
        prev_abs = np.abs(cc[idx, j - 1])
        fin = np.isfinite(n) & np.isfinite(it)
        if fin.sum() < 40:
            continue
        pa = prev_abs[fin]
        pfin = np.isfinite(pa)
        med = float(np.median(pa[pfin])) if pfin.sum() > 10 else np.nan
        mover = pfin & (pa > med) if np.isfinite(med) else np.zeros(fin.sum(), bool)
        calm = pfin & ~(pa > med) if np.isfinite(med) else np.zeros(fin.sum(), bool)
        nf, itf = n[fin], it[fin]
        tv = float(V[idx, j].sum())
        agg[j] = {
            "day": d.isoformat(),
            "n_names": int(fin.sum()),
            "night": float(nf.mean()),
            "intra": float(itf.mean()),
            "cc": float(np.nanmean(c2c)),
            "disp": float(np.nanstd(c2c[np.isfinite(c2c)])),
            "breadth": float(np.mean(c2c[np.isfinite(c2c)] > 0)),
            "logtv": float(np.log(max(tv, 1.0))),
            "night_mover": float(nf[mover].mean()) if mover.sum() > 5 else None,
            "night_calm": float(nf[calm].mean()) if calm.sum() > 5 else None,
            "intra_mover": float(itf[mover].mean()) if mover.sum() > 5 else None,
            "intra_calm": float(itf[calm].mean()) if calm.sum() > 5 else None,
        }

    js = sorted(agg)
    pos = {j: k for k, j in enumerate(js)}
    out = []
    for j in js:
        k = pos[j]
        if k < 22:  # need 21 prior aggregate rows for the vol/tv windows
            continue
        a = agg[j]
        d = dt.date.fromisoformat(a["day"])
        if not (lo <= d <= hi):
            continue
        prev = agg[js[k - 1]]
        hist = [agg[js[k - i]] for i in range(1, 22)]  # d-1 .. d-21, all <= d-1 close
        cc5 = [h["cc"] for h in hist[:5]]
        cc21 = [h["cc"] for h in hist]
        tv5 = float(np.mean([h["logtv"] for h in hist[:5]]))
        tv21 = float(np.mean([h["logtv"] for h in hist]))
        row = {
            "day": a["day"],
            "n_names": a["n_names"],
            # --- observables (all from day d-1 close or earlier) ---
            "o_night": prev["night"],
            "o_intra": prev["intra"],
            "o_disp": prev["disp"],
            "o_breadth": prev["breadth"],
            "o_rv5": float(np.std(cc5)),
            "o_rv21": float(np.std(cc21)),
            "o_logtv": prev["logtv"],
            "o_dlogtv": tv5 - tv21,
            "o_night5": float(np.mean([h["night"] for h in hist[:5]])),
            "o_intra5": float(np.mean([h["intra"] for h in hist[:5]])),
            # --- targets (day d) ---
            "t_night": a["night"],
            "t_intra": a["intra"],
            "t_cc": a["cc"],
            "t_night_mover": a["night_mover"],
            "t_night_calm": a["night_calm"],
            "t_intra_mover": a["intra_mover"],
            "t_intra_calm": a["intra_calm"],
        }
        out.append(row)

    p = Path(args.out)
    p.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"wrote {len(out)} rows {out[0]['day']}..{out[-1]['day']} -> {p}")

    # ---- frozen daily confirm window: most recent 15% of the daily range ----
    wpath = REPO / "data/eval/rlnews/DAILY_CONFIRM_WINDOW.json"
    if not wpath.exists():
        ncut = int(round(len(out) * 0.85))
        rec = {
            "frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
            "note": (
                "Fixed BEFORE any HMM search. CONFIRM_WINDOWS.json (f9376cf) has no "
                "daily-resolution entry; this is the daily one, the most recent 15% "
                "of the daily-bar range, held out from every fit and every search."
            ),
            "track_c_daily": {
                "all_days": len(out),
                "search": [out[0]["day"], out[ncut - 1]["day"]],
                "confirm": [out[ncut]["day"], out[-1]["day"]],
                "n_search": ncut,
                "n_confirm": len(out) - ncut,
            },
        }
        wpath.write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps(rec["track_c_daily"], indent=1))
    else:
        print("confirm window already frozen:", wpath)


if __name__ == "__main__":
    main()
