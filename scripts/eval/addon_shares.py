"""Does an add-on buy a round TRIP, or just more size?

edge-hunt's position is that `add_on_min_gain` 0.01 -> 0.005 "buys the most round
trips at 36.7 bp each". Mine is that an add-on puts another share into a position
that still exits once, so it pays cost in proportion to VALUE, not per trade.

Both predict the extra share pays a full round trip's worth of cost on its own
value -- the disagreement is only about the multiplier, and it is settled by
measuring buy_value and share counts rather than arguing. Printed here per policy:
tickers traded, shares bought, total buy value, and the implied cost at the true
per-market rate.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from decimal import Decimal
from pathlib import Path

from dash7_cost import COSTS

from daytrade_it.application.trading.policy import NewsPolicy

POLICIES = {
    "base (0.01)": {},
    "add_on=None": {"add_on_min_gain": None},
    "add_on=0.005": {"add_on_min_gain": Decimal("0.005")},
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
    ap.add_argument("--work-root", type=Path, default=Path("data/eval/rl_minimal/addon_work"))
    args = ap.parse_args()

    require_backtest_host()
    _export_quant_airflow_env()
    stocks = QuantAirflowDBClient().get_stock_master()
    days = [dt.date.fromisoformat(p.stem) for p in sorted(args.events.glob("2026-*.jsonl"))]
    market = {}
    for p in sorted(args.events.glob("2026-*.jsonl")):
        for line in p.read_text().splitlines():
            if line:
                e = json.loads(line)
                market[e["ticker"]] = e.get("market") or "거래소"

    base = NewsPolicy()
    pols = {
        n: dataclasses.replace(base, limits=dataclasses.replace(base.limits, **kw))
        for n, kw in POLICIES.items()
    }
    print(
        f"{'policy':14s} {'tickers':>8s} {'buys':>6s} {'sells':>6s} {'shares':>7s}"
        f" {'buy value':>12s} {'cost@true':>10s} {'net':>10s}"
    )
    for name, pol in pols.items():
        tick = buys = sells = shares = 0
        bv = 0.0
        cost = 0.0
        net = 0.0
        for day in days:
            data = load_day(day, args.cache_dir, args.events)
            res, pnl = run_day(data, pol, stocks, args.dart, args.work_root)
            buys += sum(1 for f in res.fills if f["side"] == "buy")
            sells += sum(1 for f in res.fills if f["side"] != "buy")
            for r in pnl:
                if not r["buy_value"]:
                    continue
                tick += 1
                shares += int(r["buy_qty"])
                bc, sc = COSTS.get(market.get(r["ticker"], "거래소"), COSTS["거래소"])
                proceeds = r["sell_value"] + r["leftover_value"]
                bv += float(r["buy_value"])
                cost += float(r["buy_value"] * bc + proceeds * sc)
                net += float(proceeds * (1 - sc) - r["buy_value"] * (1 + bc))
        print(
            f"{name:14s} {tick:>8d} {buys:>6d} {sells:>6d} {shares:>7d}"
            f" {bv:>12,.0f} {cost:>10,.0f} {net:>+10,.0f}"
        )


if __name__ == "__main__":
    main()
