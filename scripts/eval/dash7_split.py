"""Per-day, true-cost, split by provenance -- and is the VWAP exit a beta hedge?

Two checks that bear on every dashboard-pool result I have reported.

**Provenance.** Of the 7 "dashboard" days, only 2026-09-14 and 2026-09-15 are the
real live archive (147 and 127 events). 2026-09-07..09-11 are reconstructed from
kr_quant, which holds only ~25% of what the daemon actually saw, and they carry
41-92 events each. So five of seven days are a thin sample of a day live never
traded, and edge-hunt measures the two populations at +60.6 bp net (reconstructed)
against -33.4 (archive). Any candidate scored across that boundary is being scored
on two different things.

**Beta hedge.** A rule that exits on a VWAP break fires more often when the whole
universe is falling, so its P&L can be concentrated in down-universe days -- a beta
hedge written as an exit rule, which stops working when the market stops falling.
edge-hunt's point: the paired structure already differences out day and stock, so a
matched null adds nothing to the comparison itself, but regressing the per-day
rule-on-minus-off delta on that day's universe null separates "sells beta" (slope
strongly negative, intercept ~0) from "manages exits" (intercept survives).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
from dash7_cost import COSTS, net_of

from daytrade_it.application.trading.policy import NewsPolicy

ARCHIVE = {"2026-09-14", "2026-09-15"}
POLICIES: dict[str, dict] = {
    "add_on=0.005": {"add_on_min_gain": Decimal("0.005")},
    "hold_to_close": {"vwap_exit_gap": None},
    "trail_2_1": {"trail_after_gain": Decimal("0.02"), "trail_gap": Decimal("0.01")},
}


def main() -> None:
    from replay_common import CACHE, load_day, run_day
    from replay_parity import DART, _export_quant_airflow_env

    from daytrade_it.infrastructure.external.quant_airflow_db_client import QuantAirflowDBClient
    from daytrade_it.shared.runtime_host import require_backtest_host

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--dart", type=Path, default=DART)
    ap.add_argument("--cache-dir", type=Path, default=CACHE)
    ap.add_argument("--nulls", type=Path, required=True)
    ap.add_argument("--work-root", type=Path, default=Path("data/eval/rl_minimal/split_work"))
    args = ap.parse_args()

    require_backtest_host()
    _export_quant_airflow_env()
    stocks = QuantAirflowDBClient().get_stock_master()
    days = [dt.date.fromisoformat(p.stem) for p in sorted(args.events.glob("2026-*.jsonl"))]
    market: dict[str, str] = {}
    for p in sorted(args.events.glob("2026-*.jsonl")):
        for line in p.read_text().splitlines():
            if line:
                e = json.loads(line)
                market[e["ticker"]] = e.get("market") or "거래소"

    base = NewsPolicy()
    pols = {"base": base} | {
        n: dataclasses.replace(base, limits=dataclasses.replace(base.limits, **kw))
        for n, kw in POLICIES.items()
    }
    per_day: dict[str, dict[str, float]] = {}
    for day in days:
        data = load_day(day, args.cache_dir, args.events)
        row = {}
        for name, pol in pols.items():
            _r, pnl = run_day(data, pol, stocks, args.dart, args.work_root)
            row[name] = net_of(pnl, market, COSTS)[0]
        per_day[str(day)] = row

    names = [n for n in pols if n != "base"]
    print(f"{'day':12s} {'src':6s} {'base':>10s} " + " ".join(f"{n:>14s}" for n in names))
    for d in sorted(per_day):
        src = "ARCH" if d in ARCHIVE else "recon"
        print(
            f"{d:12s} {src:6s} {per_day[d]['base']:>+10,.0f} "
            + " ".join(f"{per_day[d][n] - per_day[d]['base']:>+14,.0f}" for n in names)
        )
    for label, sel in (
        ("reconstructed", lambda d: d not in ARCHIVE),
        ("ARCHIVE", lambda d: d in ARCHIVE),
    ):
        ds = [d for d in per_day if sel(d)]
        print(
            f"{label:19s} {sum(per_day[d]['base'] for d in ds):>+10,.0f} "
            + " ".join(
                f"{sum(per_day[d][n] - per_day[d]['base'] for d in ds):>+14,.0f}" for n in names
            )
            + f"   ({len(ds)} days)"
        )

    # Beta-hedge regression: is the VWAP exit's P&L concentrated on down-universe days?
    nulls: dict[str, list[float]] = {}
    for line in args.nulls.read_text().splitlines():
        if line:
            r = json.loads(line)
            nulls.setdefault(r["day"], []).append(float(r["null_mean"]))
    x, y = [], []
    for d in sorted(per_day):
        if d in nulls:
            x.append(float(np.mean(nulls[d])))
            # hold_to_close = the rule OFF, so rule-on minus rule-off is the negative
            y.append(per_day[d]["base"] - per_day[d]["hold_to_close"])
    if len(x) >= 3:
        xa, ya = np.array(x), np.array(y)
        slope, intercept = np.polyfit(xa, ya, 1)
        pred = slope * xa + intercept
        ss = 1 - ((ya - pred) ** 2).sum() / max(((ya - ya.mean()) ** 2).sum(), 1e-12)
        print(
            f"\nvwap_exit contribution vs the day's universe null ({len(x)} days):"
            f"\n  slope {slope:+,.0f} KRW per unit null   intercept {intercept:+,.0f} KRW"
            f"   R2 {ss:.2f}"
        )
        print(f"  day nulls (bp): {[round(v * 1e4, 1) for v in x]}")
        print(f"  vwap contribution: {[round(v) for v in y]}")
        print(
            "  reading: a strongly negative slope with a ~0 intercept would mean the rule is"
            " selling beta rather than managing exits."
        )


if __name__ == "__main__":
    main()
