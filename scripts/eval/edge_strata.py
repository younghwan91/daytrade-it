"""Stratum-matched null: same day, same minute, same universe, AND the same
volatility and liquidity bucket.

The plain null in `edge_null.py` compares a news-tagged name to the average
universe name. That is not a fair match: news-tagged names are the big,
volatile, heavily traded ones, and on an up day a high-beta name earns more
than the cross-section for reasons that have nothing to do with the news. This
version buckets the universe by two decision-time-clean variables and compares
each pair only to its own bucket.

    sigma   population sd of the trailing 20 close-to-close returns, ending at
            the PREVIOUS trading day -- never the day being traded
    value   the previous day's traded value, which is also what defines the
            universe, so the rank is free

Both are known before the open. Buckets are formed within each day, so a day
that is volatile everywhere does not move anything across buckets.

Writes two files:
  strata.jsonl  {day, code, sigma, sig_q, val_q}  for every universe code and
                every pool code -- pool codes outside the universe are placed
                against the universe's own breakpoints
  null_*.jsonl  the edge_null.py row plus `cells`: {"<sig_q>|<val_q>":
                {n, mean}} for the cross-section inside each bucket

READ-ONLY on kr_quant.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_db import connect  # noqa: E402
from edge_null import MARKET, fetch, load_anchors, trading_days  # noqa: E402

LOOKBACK = 20
NQ = 4  # quartiles on each axis -> 16 cells


def daily_history(cur, lo: dt.date, hi: dt.date) -> dict[str, list[tuple[dt.date, float, float]]]:
    cur.execute(
        "SELECT code, date, close, trade_value FROM daily_bars_adjusted "
        "WHERE date >= %s AND date <= %s ORDER BY code, date",
        (lo, hi),
    )
    out: dict[str, list[tuple[dt.date, float, float]]] = defaultdict(list)
    for code, date, close, value in cur.fetchall():
        if close and float(close) > 0:
            out[code].append((date, float(close), float(value or 0.0)))
    return out


def sigma_at(hist: list[tuple[dt.date, float, float]], day: dt.date) -> float | None:
    """sd of the last 20 close-to-close returns ending at the previous session."""
    closes = [c for d, c, _ in hist if d < day]
    if len(closes) < 11:
        return None
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))][-LOOKBACK:]
    return statistics.pstdev(rets) if len(rets) >= 10 else None


def quantile_edges(values: list[float], n: int) -> list[float]:
    s = sorted(values)
    return [s[int(len(s) * k / n)] for k in range(1, n)]


def bucket_of(value: float | None, edges: list[float]) -> int:
    return bisect.bisect_right(edges, value) if value is not None else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/eval/prompt_v3/labels.jsonl")
    ap.add_argument("--pool-codes", help="jsonl carrying day+code for pool rows outside labels")
    ap.add_argument("--out", default="data/eval/edge/null_sweep_strat.jsonl")
    ap.add_argument("--strata-out", default="data/eval/edge/strata_sweep.jsonl")
    ap.add_argument("--size", type=int, default=300)
    args = ap.parse_args()

    anchors = load_anchors(Path(args.labels))
    days = sorted(anchors)

    pool_codes: dict[dt.date, set[str]] = defaultdict(set)
    src = Path(args.pool_codes or args.labels)
    with src.open() as f:
        for line in f:
            r = json.loads(line)
            code = r.get("code") or r["key"].split("|")[1]
            pool_codes[dt.date.fromisoformat(r["day"])].add(code)

    conn = connect()
    cur = conn.cursor()
    cal = trading_days(cur, days[0] - dt.timedelta(days=90), days[-1])
    prev_of = {d: cal[i - 1] for i, d in enumerate(cal) if i}
    hist = daily_history(cur, days[0] - dt.timedelta(days=90), days[-1])
    value_on: dict[dt.date, dict[str, float]] = defaultdict(dict)
    for code, rows in hist.items():
        for d, _close, v in rows:
            if v > 0 and len(code) == 6 and code.isdigit():
                value_on[d][code] = v

    out_path, strata_path = Path(args.out), Path(args.strata_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as fo, strata_path.open("w") as fs:
        for day in days:
            prev = prev_of.get(day)
            if prev is None:
                continue
            vals = value_on[prev]
            univ = sorted(vals, key=lambda c: (-vals[c], c))[: args.size]

            sig = {c: sigma_at(hist.get(c, []), day) for c in set(univ) | pool_codes[day]}
            u_sig = [sig[c] for c in univ if sig.get(c) is not None]
            u_val = [vals[c] for c in univ]
            if len(u_sig) < NQ * 4:
                print(f"{day}  too few sigmas ({len(u_sig)}) -- skipped", flush=True)
                continue
            sig_edges = quantile_edges(u_sig, NQ)
            val_edges = quantile_edges(u_val, NQ)

            cell_of: dict[str, str] = {}
            for code in set(univ) | pool_codes[day]:
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
                            "in_universe": code in univ,
                        }
                    )
                    + "\n"
                )

            bars = fetch(cur, day, sorted(set(univ) | {MARKET}))
            mkt = bars.get(MARKET)
            members: dict[str, list] = defaultdict(list)
            for code in univ:
                if code in bars:
                    members[cell_of[code]].append(bars[code])
            have = [bars[c] for c in univ if c in bars]

            for act_min, n_pool in sorted(anchors[day].items()):
                rets = [r for d in have if (r := d.ret(act_min)) is not None]
                if not rets:
                    continue
                cells = {}
                for cell, ds in members.items():
                    cr = [r for d in ds if (r := d.ret(act_min)) is not None]
                    if cr:
                        cells[cell] = {"n": len(cr), "mean": statistics.fmean(cr)}
                fo.write(
                    json.dumps(
                        {
                            "day": str(day),
                            "act_min": act_min,
                            "n_univ": len(rets),
                            "null_mean": statistics.fmean(rets),
                            "null_med": statistics.median(rets),
                            "mkt_ret": mkt.ret(act_min) if mkt else None,
                            "n_pool": n_pool,
                            "cells": cells,
                        }
                    )
                    + "\n"
                )
            print(f"{day}  univ={len(univ)} sig={len(u_sig)} cells={len(members)}", flush=True)
    conn.close()
    print(f"-> {out_path}\n-> {strata_path}")


if __name__ == "__main__":
    main()
