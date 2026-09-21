"""Hourly-resolution news labels + stratum-matched null over the Yahoo 60m bars.

The minute-resolution twin of this is `scripts/eval/prompt_labels.py` (labels),
`scripts/eval/edge_null.py` (the matched null) and `scripts/eval/edge_strata.py`
(the 4x4 sigma x value cells). Those all read kr_quant `minute_bars`, which
stops at 2025-08-01. This file reproduces the same three artefacts on
`/home/young/git/daytrade-it/data/yahoo_hourly/bars`, which reaches back to
2023-09-19, and writes files edge_ladder.py can consume unchanged.

Bars are filtered to identity_verified AND NOT irregular_session:
9,313,689 rows / 2,050 codes / 722 trading days / 2023-09-19..2026-09-21.

THE ACTING BAR.  A judgment lands inside an hour, not on a bar boundary. The
entry is the open of the FIRST bar whose start is at or after `decide_at`
(premarket rows -- published_at before 09:00 -- clamp to the 09:00 bar, exactly
as prompt_labels.label_day clamps them to the 09:00 minute).

    Why not the literal transplant of SimBroker's rule ("this bar's open, else
    the last completed bar's close")?  At minute resolution both of those are
    at most 60 seconds stale, so the rule is harmless. At hourly resolution
    both of them are, for an article arriving mid-bar, a price struck up to 59
    minutes BEFORE the article -- i.e. before the news moved the tape. Paying
    the pre-news price for a post-news decision is look-ahead in the only
    direction that matters here, and it flatters the news pool specifically
    (the null cross-section has no event to precede). The first bar starting at
    or after decide_at is the earliest price we could actually have been filled
    at, so it is the honest hourly analogue.

THE EXIT.  Close of the 14:00 bar -- the 15:00 price, and the last bar that
comes from Yahoo itself. NOT the reconstructed 15:00 bar. This is a forced
deviation from the minute-resolution `ret_close`:

    The 15:00 bar's close is lifted from kr_quant `daily_bars_adjusted`, which
    is NOT dividend-back-adjusted, while every 09:00-14:00 bar is Yahoo's
    dividend-back-adjusted series. For a code carrying any accumulated
    adjustment the two sit on different price scales, so close(15:00) over an
    intraday open is a scale ratio, not a return. Measured over all 1,327,025
    clean code-days: close(15)/close(14) departs from 1 by more than 3% on
    6.1% of them and by more than 50% on 1.4%; 104 codes -- every one
    adjustment_step=True -- are wrong on over half their days (002070 reads
    about +2,000% every single day). The mean of close(15)/close(14)-1 is
    +968 bp, which is the whole reason a first pass of this file reported the
    top-300 earning +208 bp and the broad cross-section +1,078 bp per day.

    Rescaling that bar would be imputation; dropping only the code-days whose
    ratio looks wrong would condition on the outcome. Exiting one bar earlier
    keeps one price scale end to end and reads no reconstructed value at all.
    The cost is that the 15:00-15:30 closing auction sits outside the hold.

The reconstructed 15:00 bar is therefore never read here -- not its close, and
not its NaN high/low, so nothing is imputed anywhere. Entry at it is likewise
impossible, so an article whose decide_at falls after 14:00 has no acting bar
and is dropped (counted as `pairs_no_slot`).

`act_min` is the acting bar's start expressed as minutes from 09:00
(0/60/.../300), keeping edge_null.py's unit convention so edge_ladder.py joins
on (day, act_min) unchanged.

MARKET RUNG.  069500 is NOT in the Yahoo pull (it holds no ETFs) and has no
daily_bars_adjusted rows either, so R0 cannot be the index ETF here. `mkt_ret`
is instead the equal-weight mean of the WHOLE clean hourly cross-section
(~2,050 names) over the identical window -- a broad-market proxy. D1 therefore
reads "top-300 minus the whole listed cross-section", not "minus KOSPI200".

Dividend adjustment: entry and exit are now both Yahoo bars of the same day, so
they carry the identical back-adjustment factor and the ratio is unaffected.
Only a cross-day hold would be wrong, and there are none.

Survivorship: delisted names are absent from the Yahoo pull and cannot be
recovered. `univ_missing_bars` is the size of that hole per day.

READ-ONLY on kr_quant (calendar, trade_value, daily closes only).
"""
# ruff: noqa: B023  -- the closures below are invoked inside the same loop
# iteration that binds the names they read, never stored for later.

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

_EVAL = Path("/home/young/git/daytrade-it-rlnews/scripts/eval")
sys.path.insert(0, str(_EVAL))
import edge_db  # noqa: E402

BARS = "/home/young/git/daytrade-it/data/yahoo_hourly/bars"
#: 15:00 is deliberately absent -- see THE EXIT.
HOURS = [9, 10, 11, 12, 13, 14]
SLOT_MIN = [(h - 9) * 60 for h in HOURS]  # 0,60,...,300
NSLOT = len(HOURS)
OPEN = dt.time(9, 0)
LOOKBACK = 20
NQ = 4
UNIVERSE_SIZE = 300
MIN_CELL = 5


def acting_slot(published: dt.datetime, decide: dt.datetime, rule: str = "next") -> int | None:
    """Index into HOURS of the acting bar.

    rule="next" -- the one this file argues for: the first bar whose start is
    at or after `decide`, i.e. the earliest price we could actually have been
    filled at.

    rule="stale" -- the literal transplant of SimBroker's minute rule: the bar
    the decision falls INSIDE, entered at that bar's open, i.e. a price struck
    up to 59 minutes BEFORE the article. Kept only so the size of that
    look-ahead can be measured rather than asserted. Never the headline.

    Premarket (published before 09:00) clamps to the 09:00 bar under both,
    matching prompt_labels.label_day. None when there is no usable bar.
    """
    if published.time() < OPEN:
        return 0
    if rule == "stale":
        i = decide.hour - 9
        return i if 0 <= i < NSLOT else None
    for i, h in enumerate(HOURS):
        if decide <= decide.replace(hour=h, minute=0, second=0, microsecond=0):
            return i
    return None


# ---------------------------------------------------------------- kr_quant


def load_daily(lo: dt.date, hi: dt.date):
    conn = edge_db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT code, date, close, trade_value FROM daily_bars_adjusted "
                "WHERE date >= %s AND date <= %s ORDER BY code, date",
                (lo, hi),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    hist: dict[str, list[tuple[dt.date, float]]] = defaultdict(list)
    value_on: dict[dt.date, dict[str, float]] = defaultdict(dict)
    days: set[dt.date] = set()
    for code, date, close, value in rows:
        days.add(date)
        if not (len(code) == 6 and code.isdigit()):
            continue
        if close and float(close) > 0:
            hist[code].append((date, float(close)))
        if value and float(value) > 0:
            value_on[date][code] = float(value)
    sig_index = {c: ([d for d, _ in r], [x for _, x in r]) for c, r in hist.items()}
    return sig_index, value_on, sorted(days)


def sigma_of(index, code: str, day: dt.date) -> float | None:
    """sd of the trailing 20 close-to-close returns ENDING AT THE PREVIOUS session."""
    got = index.get(code)
    if not got:
        return None
    dates, closes = got
    i = bisect.bisect_left(dates, day)  # strictly before `day`
    w = closes[max(0, i - LOOKBACK - 1) : i]
    if len(w) < 11:
        return None
    rets = [w[k] / w[k - 1] - 1 for k in range(1, len(w))]
    return statistics.pstdev(rets) if len(rets) >= 10 else None


def quantile_edges(values: list[float], n: int) -> list[float]:
    s = sorted(values)
    return [s[int(len(s) * k / n)] for k in range(1, n)]


def bucket_of(value, edges) -> int:
    return bisect.bisect_right(edges, value) if value is not None else -1


# ---------------------------------------------------------------- bars


def day_matrix(g: pd.DataFrame):
    """(codes, ci, open[n,6], exit_px[n], exit_slot[n]) for one day's clean bars."""
    codes = pd.Index(sorted(g["code"].unique()))
    ci = {c: i for i, c in enumerate(codes)}
    n = len(codes)
    op = np.full((n, NSLOT), np.nan)
    cl = np.full((n, NSLOT), np.nan)
    for code, slot, o, c in zip(
        g["code"].values, g["slot"].values, g["open"].values, g["close"].values, strict=False
    ):
        i = ci[code]
        op[i, slot] = o
        cl[i, slot] = c
    exit_slot = np.full(n, -1, dtype=int)
    exit_px = np.full(n, np.nan)
    for s in range(NSLOT):
        ok = ~np.isnan(cl[:, s])
        exit_slot[ok] = s
        exit_px[ok] = cl[ok, s]
    return codes, ci, op, exit_px, exit_slot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="data/eval/rlnews/master_hourly.jsonl")
    ap.add_argument("--out-dir", default="data/eval/rlnews")
    ap.add_argument("--labels", default="labels_hourly.jsonl")
    ap.add_argument("--null", default="null_hourly_strat.jsonl")
    ap.add_argument("--strata", default="strata_hourly.jsonl")
    ap.add_argument("--census", default="census_hourly.json")
    ap.add_argument("--entry-rule", choices=("next", "stale"), default="next")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- master pairs
    by_day: dict[dt.date, list[dict]] = defaultdict(list)
    with Path(args.master).open() as f:
        for line in f:
            r = json.loads(line)
            by_day[dt.date.fromisoformat(r["day"])].append(r)
    print(f"master: {sum(len(v) for v in by_day.values())} pairs / {len(by_day)} days", flush=True)

    # ---- clean hourly bars
    dset = ds.dataset(BARS, partitioning="hive")
    tbl = dset.to_table(
        columns=["code", "ts", "open", "close"],
        filter=(ds.field("identity_verified") == True)  # noqa: E712
        & (ds.field("irregular_session") == False),  # noqa: E712
    )
    bars = tbl.to_pandas()
    bars["day"] = bars["ts"].dt.date
    bars["slot"] = bars["ts"].dt.hour - 9
    n_all = len(bars)
    bars = bars[(bars["slot"] >= 0) & (bars["slot"] < NSLOT)]
    print(
        f"bars: {len(bars)} of {n_all} rows kept "
        f"({n_all - len(bars)} reconstructed 15:00 rows dropped unread) / "
        f"{bars['code'].nunique()} codes / {bars['day'].nunique()} days",
        flush=True,
    )
    hourly_days = sorted(bars["day"].unique())

    lo = min(hourly_days) - dt.timedelta(days=120)
    sig_index, value_on, cal = load_daily(lo, max(hourly_days))
    prev_of = {d: cal[i - 1] for i, d in enumerate(cal) if i}
    print(f"calendar: {len(cal)} trading days from daily_bars_adjusted", flush=True)

    census = {
        "hourly_days": len(hourly_days),
        "first_day": str(min(hourly_days)),
        "last_day": str(max(hourly_days)),
        "master_pairs": sum(len(v) for v in by_day.values()),
        "entry_rule": args.entry_rule,
        "days_run": 0,
        "days_skipped_no_prev": 0,
        "days_skipped_few_sigma": 0,
        "pairs_in": 0,
        "pairs_no_bars": 0,
        "pairs_no_slot": 0,
        "pairs_no_entry": 0,
        "pairs_exit_before_entry": 0,
        "pairs_out": 0,
        "pairs_not_in_universe": 0,
        "univ_slots": 0,
        "univ_missing_bars": 0,
        "exit_slot_hist": defaultdict(int),
        "act_slot_hist": defaultdict(int),
    }

    fl = (out / args.labels).open("w")
    fn = (out / args.null).open("w")
    fs = (out / args.strata).open("w")

    for day, g in bars.groupby("day", sort=True):
        prev = prev_of.get(day)
        if prev is None:
            census["days_skipped_no_prev"] += 1
            continue
        codes, ci, op, exit_px, exit_slot = day_matrix(g)

        vals = value_on.get(prev, {})
        univ = sorted(vals, key=lambda c: (-vals[c], c))[:UNIVERSE_SIZE]
        univ_set = set(univ)

        rows = by_day.get(day, [])
        census["pairs_in"] += len(rows)
        pool_codes = {r["code"] for r in rows}

        # ---- strata: universe breakpoints, pool codes placed against them
        sig = {c: sigma_of(sig_index, c, day) for c in univ_set | pool_codes}
        u_sig = [sig[c] for c in univ if sig.get(c) is not None]
        u_val = [vals[c] for c in univ]
        if len(u_sig) < NQ * 4:
            census["days_skipped_few_sigma"] += 1
            print(f"{day}  too few sigmas ({len(u_sig)}) -- skipped", flush=True)
            continue
        sig_edges = quantile_edges(u_sig, NQ)
        val_edges = quantile_edges(u_val, NQ)
        cell_of = {}
        for code in sorted(univ_set | pool_codes):
            sq = bucket_of(sig.get(code), sig_edges)
            vq = bucket_of(vals.get(code), val_edges)
            cell_of[code] = f"{sq}|{vq}"
            fs.write(
                json.dumps(
                    {
                        "day": str(day),
                        "code": code,
                        "sigma": sig.get(code),
                        "sig_q": sq,
                        "val_q": vq,
                        "in_universe": code in univ_set,
                    }
                )
                + "\n"
            )

        n_missing = sum(1 for c in univ if c not in ci)
        census["univ_slots"] += len(univ)
        census["univ_missing_bars"] += n_missing

        # ---- labels
        anchors: dict[int, int] = defaultdict(int)
        for r in rows:
            code = r["code"]
            if code not in univ_set:
                census["pairs_not_in_universe"] += 1
            i = ci.get(code)
            if i is None:
                census["pairs_no_bars"] += 1
                continue
            published = dt.datetime.fromisoformat(r["published_at"]).replace(tzinfo=None)
            decide = dt.datetime.fromisoformat(r["decide_at"]).replace(tzinfo=None)
            s = acting_slot(published, decide, args.entry_rule)
            if s is None:
                census["pairs_no_slot"] += 1
                continue
            e = op[i, s]
            if not np.isfinite(e) or e <= 0:
                census["pairs_no_entry"] += 1
                continue
            xs = int(exit_slot[i])
            if xs < s or not np.isfinite(exit_px[i]):
                census["pairs_exit_before_entry"] += 1
                continue
            fl.write(
                json.dumps(
                    {
                        "key": r["key"],
                        "code": code,
                        "day": str(day),
                        "act_min": SLOT_MIN[s],
                        "act_slot": s,
                        "exit_slot": xs,
                        "entry": float(e),
                        "ret_close": float(exit_px[i] / e - 1),
                        "premarket": published.time() < OPEN,
                        "in_universe": code in univ_set,
                    }
                )
                + "\n"
            )
            anchors[s] += 1
            census["pairs_out"] += 1
            census["act_slot_hist"][s] += 1
            census["exit_slot_hist"][xs] += 1

        census["days_run"] += 1
        if not anchors:
            continue

        # ---- null: universe cross-section, broad-market proxy, 4x4 cells
        have = np.array([ci[c] for c in univ if c in ci], dtype=int)
        members: dict[str, list[int]] = defaultdict(list)
        for c in univ:
            if c in ci:
                members[cell_of[c]].append(ci[c])
        allidx = np.arange(len(codes))

        def rets(idx, s):
            e = op[idx, s]
            x = exit_px[idx]
            ok = np.isfinite(e) & (e > 0) & np.isfinite(x) & (exit_slot[idx] >= s)
            return x[ok] / e[ok] - 1

        for s in sorted(anchors):
            ur = rets(have, s)
            if ur.size == 0:
                continue
            mr = rets(allidx, s)
            cells = {}
            for cell, idxs in members.items():
                cr = rets(np.array(idxs, dtype=int), s)
                if cr.size:
                    cells[cell] = {"n": int(cr.size), "mean": float(cr.mean())}
            fn.write(
                json.dumps(
                    {
                        "day": str(day),
                        "act_min": SLOT_MIN[s],
                        "n_univ": int(ur.size),
                        "null_mean": float(ur.mean()),
                        "null_med": float(np.median(ur)),
                        "mkt_ret": float(mr.mean()) if mr.size else None,
                        "n_mkt": int(mr.size),
                        "n_pool": anchors[s],
                        "n_univ_missing": n_missing,
                        "cells": cells,
                    }
                )
                + "\n"
            )
        if census["days_run"] % 100 == 0:
            print(f"{day}  days={census['days_run']} pairs={census['pairs_out']}", flush=True)

    fl.close()
    fn.close()
    fs.close()
    census["exit_slot_hist"] = dict(census["exit_slot_hist"])
    census["act_slot_hist"] = dict(census["act_slot_hist"])
    (out / args.census).write_text(json.dumps(census, indent=2, default=str))
    print(json.dumps(census, indent=2, default=str))


if __name__ == "__main__":
    main()
