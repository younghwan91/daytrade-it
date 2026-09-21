"""Multi-day news alpha at DAILY resolution, across years.

Kiwoom minute bars start 2025-08-01, so every intraday result in this project
lives in one regime. `daily_bars_adjusted` goes back to 2016 and the Toss
archive to 2023-03-16, so the multi-day question -- does a news-tagged name
beat a matched null over 1..40 sessions -- IS answerable across regimes, just
not at minute resolution.

Construction, live-parity:
  entry   the CLOSE of the article day. Every pair in the pool comes from the
          live sweep window [08:00, 15:20), so the close is reachable. No
          look-ahead: nothing after 15:20 enters the decision.
  exit    the close of d+h, h in {1,2,5,10,20,40}.
  pool    `news_master_prior.py` pairs -- top-300 universe by the PREVIOUS
          day's trade_value (`universe_coverage.select_universe`), articles
          from `data/toss_universe_42m`. Collapsed to unique (day, code):
          at daily resolution ten articles on one name is one return.
  null    `edge_strata.py`'s 4x4 stratum, rebuilt at each horizon -- trailing-20
          population sd of close-to-close ending the PREVIOUS session
          (`edge_overnight.sigma_of`) x previous day's traded value, quartiles
          taken inside that day's universe. Alpha is pair minus its own cell's
          mean at the same entry day and the same horizon.

Overlap: h-day windows started every day are h-fold overlapping, so the naive
day-level standard error is too small by ~sqrt(h). Both are printed; `se_ov`
is the one to read.

Dividends: `daily_bars_adjusted` is split-adjusted but NOT dividend-adjusted,
so a window spanning an ex-date is short by the dividend. Two methods are run
and compared (`--div-method all|split`). See the report.

READ-ONLY on kr_quant.
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

#: `data/` is a symlink into the main checkout, so __file__.resolve() walks
#: out of this worktree. cwd is the repo root by contract.
REPO = Path.cwd()
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from edge_db import connect  # noqa: E402
from edge_overnight import sigma_index, sigma_of  # noqa: E402
from edge_strata import NQ, quantile_edges  # noqa: E402
from universe_coverage import select_universe  # noqa: E402

BP = 10_000
HORIZONS = (1, 2, 5, 10, 20, 40)
MIN_CELL = 5
MARKET = "069500"
MANIFEST = Path("/home/young/git/daytrade-it/data/yahoo_hourly/_MANIFEST.json")


# ---------------------------------------------------------------- panel


def load_panel(cur, lo: dt.date, hi: dt.date):
    cur.execute(
        "SELECT date, code, open, close, trade_value FROM daily_bars_adjusted "
        "WHERE date >= %s AND date <= %s",
        (lo, hi),
    )
    rows = cur.fetchall()
    days = sorted({r[0] for r in rows})
    codes = sorted({r[1] for r in rows})
    di = {d: i for i, d in enumerate(days)}
    ci = {c: i for i, c in enumerate(codes)}
    n, m = len(codes), len(days)
    opn = np.full((n, m), np.nan)
    C = np.full((n, m), np.nan)
    V = np.zeros((n, m))
    for d, c, o, cl, tv in rows:
        i, j = ci[c], di[d]
        if o and cl and float(o) > 0 and float(cl) > 0:
            opn[i, j] = float(o)
            C[i, j] = float(cl)
        V[i, j] = float(tv or 0.0)
    return days, codes, di, ci, opn, C, V


def rows_by_day(days, codes, V):
    """[(code, trade_value)] per day -- the exact input select_universe takes."""
    out = {}
    for j, d in enumerate(days):
        col = V[:, j]
        nz = np.nonzero(col)[0]
        out[d] = [(codes[i], float(col[i])) for i in nz]
    return out


def win_sum(leg: np.ndarray):
    """(cumsum, count) with NaN treated as missing, for O(1) window sums."""
    ok = np.isfinite(leg)
    cs = np.concatenate([np.zeros((leg.shape[0], 1)), np.nan_to_num(leg).cumsum(axis=1)], axis=1)
    ct = np.concatenate([np.zeros((leg.shape[0], 1)), ok.cumsum(axis=1)], axis=1)
    return cs, ct


def window(cs, ct, a: int, b: int):
    """Sum of leg[:, a:b], NaN where any element inside is missing."""
    s = cs[:, b] - cs[:, a]
    c = ct[:, b] - ct[:, a]
    s = np.where(c == (b - a), s, np.nan)
    return s


# ---------------------------------------------------------------- dividends


def load_breaks() -> dict[str, list[dt.date]]:
    """Per-code ex-adjustment break dates, from the yahoo-vs-ours all-code sweep.

    A break between day X and day Y means the two series disagree from Y on;
    any return window spanning it is shifted by the step (median 2.4%, and the
    month histogram is January/June/September, i.e. Korean ex-dividend dates).
    Only codes and dates inside the yahoo window 2023-09..2026-09 are known.
    """
    m = json.loads(MANIFEST.read_text())
    out: dict[str, list[dt.date]] = {}
    for rec in m["adjustment_breaks"]:
        brk = rec.get("breaks")
        if not isinstance(brk, list):
            continue
        ds = sorted({dt.date.fromisoformat(b["and"]) for b in brk if b.get("and")})
        if ds:
            out[rec["code"]] = ds
    return out


def spans_year_turn(d0: dt.date, d1: dt.date) -> bool:
    """Does [d0, d1] touch the Korean annual ex-dividend turn?

    `spans_break` keys off the first date of the new plateau in the yahoo-vs-ours
    ratio, and for the annual dividend that is the first trading day of January
    -- days after the true ex-date in late December. It also only knows the 315
    flagged codes inside 2023-09..2026-09. This is the blunt, code-blind,
    full-history version: drop any window overlapping 20 Dec .. 10 Jan. It costs
    windows rather than risking a silent pool across an ex-date.
    """
    d = d0
    while d <= d1:
        if (d.month == 12 and d.day >= 20) or (d.month == 1 and d.day <= 10):
            return True
        d += dt.timedelta(days=1)
    return False


def spans_break(breaks: dict[str, list[dt.date]], code: str, d0: dt.date, d1: dt.date) -> bool:
    ds = breaks.get(code)
    if not ds:
        return False
    i = bisect.bisect_right(ds, d0)
    return i < len(ds) and ds[i] <= d1


# ---------------------------------------------------------------- core


def run(args) -> dict:
    conn = connect()
    cur = conn.cursor()
    lo = dt.date.fromisoformat(args.start) - dt.timedelta(days=120)
    hi = dt.date.fromisoformat(args.end) + dt.timedelta(days=90)
    days, codes, di, ci, opn, C, V = load_panel(cur, lo, hi)
    print(f"panel {days[0]}..{days[-1]}  {len(days)} days  {len(codes)} codes", flush=True)

    cur.execute("SELECT code, market FROM stocks")
    market = dict(cur.fetchall())
    cur.execute(
        "SELECT code, last_trade_date FROM delisted_stocks WHERE last_trade_date IS NOT NULL"
    )
    delisted = dict(cur.fetchall())
    conn.close()

    rbd = rows_by_day(days, codes, V)
    hist = {}
    for i, c in enumerate(codes):
        ok = np.isfinite(C[i])
        if ok.any():
            hist[c] = [(days[j], float(C[i, j]), float(V[i, j])) for j in np.nonzero(ok)[0]]
    sindex = sigma_index(hist)

    # legs
    night = np.full_like(C, np.nan)
    night[:, :-1] = opn[:, 1:] / C[:, :-1] - 1.0
    intra = C / opn - 1.0
    cs_n, ct_n = win_sum(night)
    cs_i, ct_i = win_sum(intra)
    prev_abs = np.full_like(C, np.nan)
    prev_abs[:, 1:] = np.abs(C[:, 1:] / C[:, :-1] - 1.0)

    breaks = load_breaks()

    # pool: unique (day, code)
    pool: dict[dt.date, set[str]] = defaultdict(set)
    npairs = 0
    with Path(args.pool).open() as f:
        for line in f:
            r = json.loads(line)
            pool[dt.date.fromisoformat(r["day"])].add(r["code"])
            npairs += 1
    pool_days = sorted(d for d in pool if args.start <= str(d) <= args.end)
    print(
        f"pool {npairs:,} pairs -> {sum(len(pool[d]) for d in pool_days):,} code-days "
        f"over {len(pool_days)} days",
        flush=True,
    )

    persist: dict[tuple[str, str], str] = {}
    if args.judged:
        with Path(args.judged).open() as f:
            for line in f:
                r = json.loads(line)
                v2 = r.get("v2") or {}
                p = v2.get("persistence")
                if p:
                    persist[(r["day"], r["code"])] = p

    # accumulators: key -> {year -> {h -> [daily means]}}
    acc: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    counts: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    legs: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    surv_seen = {
        "universe_slots": 0,
        "delisted_in_universe": 0,
        "pool_slots": 0,
        "delisted_in_pool": 0,
    }

    def add(tag, year, h, val, n=1):
        acc[tag][year][h].append(val)
        counts[tag][year][h] += n

    for d in pool_days:
        j = di.get(d)
        if j is None or j == 0:
            continue
        prev = days[j - 1]
        univ = [c for c in select_universe(rbd[prev], size=args.size) if c != MARKET]
        [ci[c] for c in univ if c in ci]
        univ = [c for c in univ if c in ci]
        if len(univ) < 40:
            continue

        sig = {c: sigma_of(sindex, c, d) for c in univ}
        u_sig = [sig[c] for c in univ if sig[c] is not None]
        if len(u_sig) < NQ * 4:
            continue
        se_ = quantile_edges(u_sig, NQ)
        pv = {c: float(V[ci[c], j - 1]) for c in univ}
        ve_ = quantile_edges([pv[c] for c in univ], NQ)

        def cell(c):
            s = sig.get(c)
            return (
                bisect.bisect_right(se_, s) if s is not None else -1,
                bisect.bisect_right(ve_, pv.get(c, 0.0)) if pv.get(c) else -1,
            )

        cell_of = {c: cell(c) for c in univ}
        # mover split, decision-time clean: |return| of day j-1, which closes
        # one full session before the entry close at j. Median inside the
        # day's own universe, so a market-wide quiet day moves nobody.
        pa = {c: float(prev_abs[ci[c], j - 1]) for c in univ}
        pa = {c: v for c, v in pa.items() if np.isfinite(v)}
        med = statistics.median(pa.values()) if pa else None
        mov_of = {c: (pa[c] > med) for c in pa} if med is not None else {}

        year = d.year
        codes_today = [c for c in pool[d] if c in cell_of]
        surv_seen["universe_slots"] += len(univ)
        surv_seen["delisted_in_universe"] += sum(1 for c in univ if c in delisted)
        surv_seen["pool_slots"] += len(codes_today)
        surv_seen["delisted_in_pool"] += sum(1 for c in codes_today if c in delisted)
        if not codes_today:
            continue

        for h in HORIZONS:
            k = j + h
            if k >= len(days):
                continue
            exd = days[k]
            r_all = C[:, k] / C[:, j] - 1.0
            on = window(cs_n, ct_n, j, j + h)  # nights j..j+h-1
            ind = window(cs_i, ct_i, j + 1, j + h + 1)  # sessions j+1..j+h

            turn = spans_year_turn(d, exd)
            for method in ("all", "split", "turn"):
                if method == "turn" and turn:
                    continue
                keep_u = {}
                for c in univ:
                    i = ci[c]
                    v = r_all[i]
                    if not np.isfinite(v):
                        continue
                    if method == "split" and spans_break(breaks, c, d, exd):
                        continue
                    keep_u[c] = (float(v), float(on[i]), float(ind[i]))
                if len(keep_u) < NQ * NQ * MIN_CELL:
                    continue

                # cell means (and cell x mover means)
                grp: dict = defaultdict(list)
                grpm: dict = defaultdict(list)
                for c, (v, o_, i_) in keep_u.items():
                    grp[cell_of[c]].append((v, o_, i_))
                    if c in mov_of:
                        grpm[(cell_of[c], mov_of[c])].append((v, o_, i_))

                def means(g):
                    out = {}
                    for kk, vs in g.items():
                        if len(vs) >= MIN_CELL:
                            fin = [x for x in vs if np.isfinite(x[1]) and np.isfinite(x[2])]
                            out[kk] = (
                                statistics.fmean(x[0] for x in vs),
                                statistics.fmean(x[1] for x in fin) if fin else None,
                                statistics.fmean(x[2] for x in fin) if fin else None,
                            )
                    return out

                cm, cmm = means(grp), means(grpm)

                buckets: dict[str, list[float]] = defaultdict(list)
                legbuf: dict[str, list[tuple[float, float]]] = defaultdict(list)
                for c in codes_today:
                    if c not in keep_u:
                        continue
                    v, o_, i_ = keep_u[c]
                    base = cm.get(cell_of[c])
                    if base is None:
                        continue
                    a = v - base[0]
                    buckets[f"pool/{method}"].append(a)
                    if method == "all":
                        if c not in delisted:
                            buckets["pool/surv"].append(a)
                        if base[1] is not None and np.isfinite(o_) and np.isfinite(i_):
                            legbuf["pool"].append((o_ - base[1], i_ - base[2]))
                        mb = cmm.get((cell_of[c], mov_of.get(c)))
                        if c in mov_of and mb is not None:
                            tag = "mover" if mov_of[c] else "calm"
                            buckets[f"pool/{tag}"].append(v - mb[0])
                            if mb[1] is not None and np.isfinite(o_) and np.isfinite(i_):
                                legbuf[tag].append((o_ - mb[1], i_ - mb[2]))
                        p = persist.get((str(d), c))
                        if p:
                            buckets[f"persist/{p}"].append(a)

                for tag, vals in buckets.items():
                    if vals:
                        add(tag, year, h, statistics.fmean(vals), len(vals))
                for tag, vals in legbuf.items():
                    legs[tag][year][h].append(
                        (statistics.fmean(v[0] for v in vals), statistics.fmean(v[1] for v in vals))
                    )
        if d.day == 1 or d == pool_days[-1]:
            print(f"  {d}", flush=True)

    # cost
    from costs import round_trip

    mk = [market.get(c, "거래소") for d in pool_days[:200] for c in list(pool[d])[:50]]
    kosdaq = sum(1 for x in mk if x == "코스닥") / max(1, len(mk))
    cost = (1 - kosdaq) * round_trip(pool_days[-1], "거래소") + kosdaq * round_trip(
        pool_days[-1], "코스닥"
    )

    return {
        "acc": acc,
        "counts": counts,
        "legs": legs,
        "surv": surv_seen,
        "cost": cost,
        "kosdaq_share": kosdaq,
        "n_days": len(pool_days),
        "span": [str(pool_days[0]), str(pool_days[-1])],
    }


# ---------------------------------------------------------------- report


def table(acc, counts, tag, title, cost):
    years = sorted(set(acc[tag]))
    print(f"\n--- {title}   (bp, gross alpha vs stratum-matched null)")
    print(
        f"{'h':>4}" + "".join(f"{y:>16}" for y in years) + f"{'pooled':>10}{'se_ov':>8}"
        f"{'net':>9}{'bp/day':>8}"
    )
    for h in HORIZONS:
        cells = ""
        flat = []
        for y in years:
            v = acc[tag][y].get(h, [])
            flat += v
            cells += f"{statistics.fmean(v) * BP:>+9.1f}({len(v):>4})" if v else f"{'.':>16}"
        if not flat:
            continue
        se = statistics.stdev(flat) / (len(flat) ** 0.5) if len(flat) > 1 else float("nan")
        se_ov = se * (h**0.5)
        g = statistics.fmean(flat)
        print(
            f"{h:>4}{cells}{g * BP:>+10.1f}{se_ov * BP:>8.1f}"
            f"{(g - cost) * BP:>+9.1f}{(g - cost) / h * BP:>+8.2f}"
        )
    n = {h: sum(counts[tag][y].get(h, 0) for y in years) for h in HORIZONS}
    print("     code-days: " + "  ".join(f"h{h}={n[h]:,}" for h in HORIZONS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/eval/rlnews/master_daily_full.jsonl")
    ap.add_argument("--judged", default="data/eval/prompt_v3/master.jsonl")
    ap.add_argument("--start", default="2023-03-16")
    ap.add_argument("--end", default="2026-09-15")
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--out", default="data/eval/rlnews/daily_horizon.json")
    args = ap.parse_args()

    res = run(args)
    acc, counts, cost = res["acc"], res["counts"], res["cost"]
    print(f"\n\n=== pool {res['span'][0]}..{res['span'][1]}, {res['n_days']} trading days")
    print(
        f"=== round trip {cost * BP:.1f} bp (costs.round_trip, KOSDAQ share "
        f"{res['kosdaq_share']:.2f}); charged ONCE per trade, so bp/day = net/h"
    )

    table(acc, counts, "pool/all", "Q1  news pool, every window (method A)", cost)
    table(
        acc, counts, "pool/split", "Q1  news pool, ex-adjustment windows dropped (method B)", cost
    )
    table(acc, counts, "pool/turn", "Q1  news pool, year-end ex-div turn dropped (method C)", cost)
    table(acc, counts, "pool/surv", "survivorship: delisted codes removed from the pool", cost)
    for p in ("structural", "multi_quarter", "one_off"):
        if f"persist/{p}" in acc:
            table(acc, counts, f"persist/{p}", f"Q2  persistence={p} (judged days only)", cost)
    table(acc, counts, "pool/mover", "Q4  prev-day movers (cell x mover null)", cost)
    table(acc, counts, "pool/calm", "Q4  prev-day calm (cell x mover null)", cost)

    print("\n--- Q3  pool alpha split into overnight / intraday legs (bp, daily bars)")
    for tag in ("pool", "mover", "calm"):
        if tag not in res["legs"]:
            continue
        years = sorted(res["legs"][tag])
        print(f"\n  {tag}")
        print(f"{'h':>4}" + "".join(f"{y:>18}" for y in years) + f"{'on':>9}{'intra':>9}")
        for h in HORIZONS:
            cells = ""
            fo, fi = [], []
            for y in years:
                v = res["legs"][tag][y].get(h, [])
                fo += [x[0] for x in v]
                fi += [x[1] for x in v]
                cells += (
                    f"{statistics.fmean(x[0] for x in v) * BP:>+9.1f}"
                    f"{statistics.fmean(x[1] for x in v) * BP:>+9.1f}"
                    if v
                    else f"{'.':>18}"
                )
            if fo:
                print(
                    f"{h:>4}{cells}{statistics.fmean(fo) * BP:>+9.1f}"
                    f"{statistics.fmean(fi) * BP:>+9.1f}"
                )

    print(f"\n--- survivorship census: {res['surv']}")

    out = {
        "span": res["span"],
        "n_days": res["n_days"],
        "cost": cost,
        "kosdaq_share": res["kosdaq_share"],
        "surv": res["surv"],
        "alpha": {
            t: {
                str(y): {
                    str(h): [statistics.fmean(v) if v else None, len(v)] for h, v in hs.items()
                }
                for y, hs in ys.items()
            }
            for t, ys in acc.items()
        },
        "counts": {
            t: {str(y): {str(h): n for h, n in hs.items()} for y, hs in ys.items()}
            for t, ys in counts.items()
        },
        "legs": {
            t: {
                str(y): {
                    str(h): [statistics.fmean(x[0] for x in v), statistics.fmean(x[1] for x in v)]
                    for h, v in hs.items()
                    if v
                }
                for y, hs in ys.items()
            }
            for t, ys in res["legs"].items()
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
