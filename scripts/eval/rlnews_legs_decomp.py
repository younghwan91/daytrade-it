"""The overnight / intraday split, by year, at BOTH resolutions.

Daily: `daily_bars_adjusted`, 2016-09-09.. -- overnight = prev close -> open,
intraday = open -> close, averaged across the top-300 universe each day and
then across days in a year.

Hourly: the Yahoo 60-minute dataset, 2023-09-19..2026-09-21 -- overnight =
09:00 bar open / previous session's 15:00 bar close, intraday = 15:00 bar
close / 09:00 bar open. `irregular_session` and unverified-identity codes are
dropped; high/low are never touched (NaN by design in the reconstructed 15:00
bar).

The two series are NOT the same measurement: Yahoo back-adjusts for dividends
and our daily bars do not, so on an ex-date the daily overnight leg carries a
~2.4% drop that the hourly leg does not. That is a prediction, not a nuisance
-- it should show up in January / June / September and nowhere else, and the
run reports the legs with and without those months so the agreement can be
judged on the clean part.

Mover / calm split uses the PREVIOUS session's absolute close-to-close move,
median inside that day's own universe. Nothing from day t enters the split for
day t.

READ-ONLY on kr_quant. Reads /home/young/git/daytrade-it/data/yahoo_hourly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "data" / "eval" / "rlnews"))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from daily_horizon import load_panel, rows_by_day  # noqa: E402
from edge_db import connect  # noqa: E402
from universe_coverage import select_universe  # noqa: E402

BP = 10_000
MARKET = "069500"
YAHOO = Path("/home/young/git/daytrade-it/data/yahoo_hourly/bars")
EXDIV_MONTHS = (1, 6, 9)


def universe_map(days, codes, V, size, lo, hi):
    rbd = rows_by_day(days, codes, V)
    out = {}
    for j, d in enumerate(days):
        if j == 0 or not (lo <= d <= hi):
            continue
        out[d] = [c for c in select_universe(rbd[days[j - 1]], size=size) if c != MARKET]
    return out


def show(title, per, keys):
    years = sorted({y for k in keys for y in per.get(k, {})})
    print(f"\n--- {title}  (bp per day)")
    print(f"{'leg':<22}" + "".join(f"{y:>8}" for y in years) + f"{'pooled':>10}{'se':>7}{'n':>8}")
    for k in keys:
        v = per.get(k, {})
        flat = [x for y in years for x in v.get(y, [])]
        if not flat:
            continue
        cells = "".join(
            f"{statistics.fmean(v[y]) * BP:>8.1f}" if v.get(y) else f"{'.':>8}" for y in years
        )
        se = statistics.stdev(flat) / (len(flat) ** 0.5) if len(flat) > 1 else float("nan")
        print(f"{k:<22}{cells}{statistics.fmean(flat) * BP:>+10.1f}{se * BP:>7.1f}{len(flat):>8}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-09-12")
    ap.add_argument("--end", default="2026-09-21")
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--out", default="data/eval/rlnews/legs_decomp.json")
    args = ap.parse_args()
    lo, hi = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)

    conn = connect()
    cur = conn.cursor()
    days, codes, di, ci, opn, C, V = load_panel(cur, lo - dt.timedelta(days=30), hi)
    conn.close()
    print(f"daily panel {days[0]}..{days[-1]}  {len(days)} days  {len(codes)} codes", flush=True)

    night = np.full_like(C, np.nan)
    night[:, 1:] = opn[:, 1:] / C[:, :-1] - 1.0
    intra = C / opn - 1.0
    prev_abs = np.full_like(C, np.nan)
    prev_abs[:, 1:] = np.abs(C[:, 1:] / C[:, :-1] - 1.0)

    umap = universe_map(days, codes, V, args.size, lo, hi)
    per: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    univ_by_day = {}
    mov_by_day = {}
    for d, univ in umap.items():
        j = di[d]
        idx = np.array([ci[c] for c in univ if c in ci])
        if idx.size < 40:
            continue
        univ_by_day[d] = [c for c in univ if c in ci]
        pa = prev_abs[idx, j - 1]
        fin = np.isfinite(pa)
        med = float(np.median(pa[fin])) if fin.sum() > 10 else None
        mov = {}
        if med is not None:
            for c, v in zip(univ_by_day[d], pa, strict=True):
                if np.isfinite(v):
                    mov[c] = bool(v > med)
        mov_by_day[d] = mov
        y = d.year
        for leg, arr in (("overnight", night), ("intraday", intra)):
            v = arr[idx, j]
            v = v[np.isfinite(v)]
            if v.size:
                per[f"daily {leg}"][y].append(float(v.mean()))
            for name, want in (("mover", True), ("calm", False)):
                sel = np.array([ci[c] for c in univ_by_day[d] if mov.get(c) is want], dtype=int)
                if sel.size >= 20:
                    w = arr[sel, j]
                    w = w[np.isfinite(w)]
                    if w.size:
                        per[f"daily {leg} {name}"][y].append(float(w.mean()))
        if d.month not in EXDIV_MONTHS:
            for leg, arr in (("overnight", night), ("intraday", intra)):
                v = arr[idx, j]
                v = v[np.isfinite(v)]
                if v.size:
                    per[f"daily {leg} ex-div-free"][y].append(float(v.mean()))

    show(
        "Q3 daily resolution, top-300 universe, full history",
        per,
        [
            "daily overnight",
            "daily intraday",
            "daily overnight ex-div-free",
            "daily intraday ex-div-free",
        ],
    )
    show(
        "Q4 daily resolution, split by PREVIOUS day's move",
        per,
        [
            "daily overnight mover",
            "daily overnight calm",
            "daily intraday mover",
            "daily intraday calm",
        ],
    )

    # ---------------------------------------------------------------- hourly
    print("\nloading yahoo hourly ...", flush=True)
    df = pd.read_parquet(
        YAHOO,
        columns=[
            "code",
            "ts",
            "open",
            "close",
            "source",
            "irregular_session",
            "identity_verified",
            "adjustment_step",
            "recovery_ok",
        ],
    )
    # `recovery_ok` is required, not optional. This decomposition reads the
    # reconstructed 15:00 bar on BOTH legs (overnight = 09:00 open / previous
    # 15:00 close, intraday = 15:00 close / 09:00 open), and that bar's close was
    # rebuilt on 2026-09-22 after it was found to sit on a different price basis
    # from the Yahoo bars around it. Filtering on identity and session alone let
    # the bad bars through, because the affected codes had verified identity.
    df = df[(~df.irregular_session) & (df.identity_verified) & (df.recovery_ok)]
    df["day"] = df.ts.dt.floor("D")
    df["hour"] = df.ts.dt.hour
    df = df.sort_values(["code", "ts"])
    sess = (
        df.groupby(["code", "day"], sort=True)
        .agg(
            first_open=("open", "first"),
            last_close=("close", "last"),
            n=("open", "size"),
            maxhour=("hour", "max"),
            adj=("adjustment_step", "first"),
        )
        .reset_index()
    )
    sess = sess[(sess.n >= 6) & (sess.maxhour == 15)]
    sess["date"] = sess["day"].dt.date
    sess = sess[["code", "date", "first_open", "last_close", "n", "maxhour", "adj"]]
    print(f"hourly sessions: {len(sess):,} code-days, {sess.code.nunique()} codes", flush=True)

    per_h: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_code: dict[str, dict[dt.date, tuple[float, float, bool]]] = defaultdict(dict)
    for code, date, fo, lc, _n, _hc, adj in sess.itertuples(index=False):
        if fo and lc and fo > 0 and lc > 0:
            by_code[code][date] = (float(fo), float(lc), bool(adj))

    hdays = sorted(set(umap))
    hpos = {d: i for i, d in enumerate(hdays)}
    for d in hdays:
        i = hpos[d]
        if i == 0:
            continue
        pd_ = hdays[i - 1]
        univ = univ_by_day.get(d)
        if not univ:
            continue
        mov = mov_by_day.get(d, {})
        y = d.year
        on, intr = [], []
        on_m, on_c, in_m, in_c = [], [], [], []
        on_clean = []
        for code in univ:
            rec = by_code.get(code)
            if not rec:
                continue
            a, b = rec.get(pd_), rec.get(d)
            if not a or not b:
                continue
            o = b[0] / a[1] - 1.0
            t = b[1] / b[0] - 1.0
            if not (np.isfinite(o) and np.isfinite(t) and abs(o) < 0.5):
                continue
            on.append(o)
            intr.append(t)
            if not b[2]:
                on_clean.append(o)
            if code in mov:
                (on_m if mov[code] else on_c).append(o)
                (in_m if mov[code] else in_c).append(t)
        if len(on) >= 30:
            per_h["hourly overnight"][y].append(statistics.fmean(on))
            per_h["hourly intraday"][y].append(statistics.fmean(intr))
            if d.month not in EXDIV_MONTHS:
                per_h["hourly overnight ex-div-free"][y].append(statistics.fmean(on))
                per_h["hourly intraday ex-div-free"][y].append(statistics.fmean(intr))
            if len(on_clean) >= 30:
                per_h["hourly overnight no-adj-step"][y].append(statistics.fmean(on_clean))
        if len(on_m) >= 15 and len(on_c) >= 15:
            per_h["hourly overnight mover"][y].append(statistics.fmean(on_m))
            per_h["hourly overnight calm"][y].append(statistics.fmean(on_c))
            per_h["hourly intraday mover"][y].append(statistics.fmean(in_m))
            per_h["hourly intraday calm"][y].append(statistics.fmean(in_c))

    show(
        "Q3 hourly resolution, same top-300 universe",
        per_h,
        [
            "hourly overnight",
            "hourly intraday",
            "hourly overnight ex-div-free",
            "hourly intraday ex-div-free",
            "hourly overnight no-adj-step",
        ],
    )
    show(
        "Q4 hourly resolution, split by PREVIOUS day's move",
        per_h,
        [
            "hourly overnight mover",
            "hourly overnight calm",
            "hourly intraday mover",
            "hourly intraday calm",
        ],
    )

    both = {**dict(per.items()), **dict(per_h.items())}
    Path(args.out).write_text(
        json.dumps(
            {
                k: {str(y): [statistics.fmean(v), len(v)] for y, v in ys.items() if v}
                for k, ys in both.items()
            },
            indent=1,
        )
    )
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
