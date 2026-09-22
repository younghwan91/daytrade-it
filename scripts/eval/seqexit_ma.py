"""Moving-average position per (ticker, day), from daily bars, decision-time only.

The owner's instruction: "5, 10, 20, 50, 100, 150, 200 선 기준 지금 위아래에
있냐가 중요해 청산은" -- whether price sits above or below each of those moving
averages matters for the exit.

**Which resolution.** The set 5/10/20/50/100/150/200 is ambiguous at minute
resolution: MA200 over minute bars is 3.3 hours, over daily bars it is ten
months. They are different hypotheses, not two estimates of one thing, so both
are built and the report says which the policy actually uses. This module
builds the DAILY one; the intraday one is computed inside
`seqexit_core.reduce_row` from the same minute bars the exit already reads.

**Why the daily one is evaluated at the previous close, not intraday.**
`minute_bars` are raw and `daily_bars_adjusted` is split/dividend adjusted, so
comparing today's raw intraday price against an adjusted moving average is
comparing two price scales -- the exact defect that fabricated +8.1 bp of D2
in the hourly transplant and +208 bp/day in the 15:00 seam (research doc 5.1,
5.3). Keeping the daily feature entirely inside the adjusted series avoids it:
the feature is "where did this stock close last night relative to its own
5..200-day moving averages", which is trend CONTEXT and is constant within the
day. The within-day movement is carried by the intraday MA features, which live
entirely inside the raw minute series. Neither feature ever mixes the two
scales.

**Live parity.** Every input ends at the close of d-1, which the live daemon
has in `daily_bars_adjusted` before the session opens. No look-ahead.

**Normalisation.** `ma_frac` is a count over a fixed denominator, so it is
dimensionless. `ma_dist` is the mean of `(close - MA_k) / MA_k` -- a percentage,
never a KRW distance -- divided by that stock's OWN trailing daily-return
standard deviation, so "one unit above the averages" means the same thing on a
quiet name and a volatile one. Nothing here is standardised with a pool-wide
mean or variance.

    .venv/bin/python scripts/eval/seqexit_ma.py \
        --paths data/eval/rlnews/paths_fresh.jsonl \
        --out data/eval/rlnews/ma_daily_fresh.json          # simnode, kr_quant READ-ONLY
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path as FsPath

import numpy as np

sys.path.insert(0, "scripts/eval")

import edge_db  # noqa: E402

MA_PERIODS = (5, 10, 20, 50, 100, 150, 200)
#: Trailing window for the per-stock volatility that scales `ma_dist`. 60 daily
#: returns is a quarter -- long enough to be stable, short enough to track a
#: regime change in the name itself.
VOL_WINDOW = 60
#: A floor on that volatility, for the same reason `rl_exits.entry_sigma` has
#: one: a barely-moving name must not produce an unbounded normalised distance.
VOL_FLOOR = 0.005
#: Calendar days of daily history to pull before the first decision day. 200
#: trading days is ~290 calendar days; 420 leaves room for halts and holidays.
LOOKBACK_DAYS = 420


def build(keys: list[str]) -> dict[str, dict]:
    """`{f"{day}|{code}": {...}}` for every (day, code) that has enough history.

    One SELECT. Rows are ordered by code then date, and for each decision day
    the moving averages are taken over the closes STRICTLY BEFORE it, so the
    decision day's own bar can never enter its own feature.
    """
    by_code_days: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        d, c = key.split("|", 1)
        by_code_days[c].append(d)
    codes = sorted(by_code_days)
    days = sorted({k.split("|", 1)[0] for k in keys})
    first = dt.date.fromisoformat(min(days)) - dt.timedelta(days=LOOKBACK_DAYS)
    last = dt.date.fromisoformat(max(days))
    conn = edge_db.connect()
    series: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT code, date, close FROM daily_bars_adjusted "
            "WHERE code = ANY(%s) AND date >= %s AND date <= %s AND close > 0 "
            "ORDER BY code, date",
            (codes, first, last),
        )
        for code, date, close in cur.fetchall():
            series[code][0].append(date)
            series[code][1].append(float(close))
    conn.close()

    out: dict[str, dict] = {}
    longest = max(MA_PERIODS)
    for code, wanted in by_code_days.items():
        dates, closes = series.get(code, ([], []))
        if len(closes) < longest + 1:
            continue
        arr = np.asarray(closes)
        # Prefix sums so every period's mean is one subtraction.
        cum = np.concatenate([[0.0], np.cumsum(arr)])
        rets = arr[1:] / arr[:-1] - 1.0
        date_arr = np.asarray(dates)
        for day in wanted:
            d = dt.date.fromisoformat(day)
            # i = number of closes strictly before `day`. `searchsorted` on the
            # date list gives exactly that, whether or not `day` itself trades.
            i = int(np.searchsorted(date_arr, d, side="left"))
            if i < longest + 1:
                continue
            last_close = arr[i - 1]
            mas = np.array([(cum[i] - cum[i - k]) / k for k in MA_PERIODS])
            above = last_close > mas
            dist = (last_close - mas) / mas  # percentage, never KRW
            lo = max(i - 1 - VOL_WINDOW, 0)
            vol = float(rets[lo : i - 1].std()) if i - 1 - lo >= 5 else float("nan")
            vol = max(vol, VOL_FLOOR) if np.isfinite(vol) else VOL_FLOOR
            out[f"{day}|{code}"] = {
                "frac": float(above.mean()),
                "n_above": int(above.sum()),
                "dist_norm": float(dist.mean() / vol),
                "vol": vol,
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=FsPath, required=True)
    ap.add_argument("--out", type=FsPath, required=True)
    args = ap.parse_args()

    pairs = set()
    with args.paths.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                pairs.add(f"{r['day']}|{r['ticker']}")
    keys = sorted(pairs)
    codes = {k.split("|", 1)[1] for k in keys}
    days = {k.split("|", 1)[0] for k in keys}
    print(f"{len(keys)} (day, code) pairs; {len(codes)} codes; {len(days)} days", flush=True)

    out = build(keys)
    print(f"resolved {len(out)} of {len(keys)} ({len(keys) - len(out)} lack 201 daily closes)")
    fr = np.array([v["frac"] for v in out.values()])
    dn = np.array([v["dist_norm"] for v in out.values()])
    print(
        f"ma_frac_daily: mean {fr.mean():.3f}; share at 0.0 {np.mean(fr == 0):.3f}, "
        f"at 1.0 {np.mean(fr == 1):.3f}"
    )
    print(
        f"ma_dist_daily (per-stock normalised): median {np.median(dn):+.2f}, "
        f"p10 {np.quantile(dn, 0.1):+.2f}, p90 {np.quantile(dn, 0.9):+.2f}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
