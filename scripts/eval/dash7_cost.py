"""Re-cost the dashboard-pool candidates at the broker's true round-trip cost.

The research constant `SELL_COST = 0.00215` is the sell tax plus ONE side of
commission with slippage at ZERO. The broker's own model -- what `PaperBroker`
actually charges -- is 거래소 33.0 bp and 코스닥 53.0 bp per round trip, verified
independently against the KRX tick grid (one tick averages 12.5 bp on 132,185 real
entry prices: 20.0 + 3.0 + 12.5 = 35.5, within 1.2 bp of the model).

Differences between policies at the SAME trade count are unaffected -- cost
cancels. What is affected is anything that changes the NUMBER of round trips,
which is precisely the pyramiding and position-cap candidates.

Cost is applied per market, not pool-weighted: market membership is known at
decision time and the 20 bp KOSDAQ penalty is larger than any selection feature
measured tonight. The split keeps the existing shape (tax and fees on the sell
side) and scales it so the round trip totals the modelled figure.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from decimal import Decimal
from pathlib import Path

from daytrade_it.application.trading.policy import NewsPolicy

#: buy-side, sell-side; the pair sums to the modelled round trip for that market.
COSTS = {
    "거래소": (Decimal("0.00015"), Decimal("0.00315")),
    "코스닥": (Decimal("0.00015"), Decimal("0.00515")),
}
OLD = (Decimal("0.00015"), Decimal("0.00215"))

CANDIDATES: dict[str, dict] = {
    "add_on=None": {"add_on_min_gain": None},
    "add_on=0.005": {"add_on_min_gain": Decimal("0.005")},
    "add_on=0.02": {"add_on_min_gain": Decimal("0.02")},
    "add_on=0.03": {"add_on_min_gain": Decimal("0.03")},
    "max_open=10": {"max_open_positions": 10},
    "trail_2_1": {"trail_after_gain": Decimal("0.02"), "trail_gap": Decimal("0.01")},
    "consensus": {
        "trail_after_gain": Decimal("0.01"),
        "trail_gap": Decimal("0.005"),
        "stop_orb": dt.timedelta(minutes=30),
    },
    "hold_to_close": {"vwap_exit_gap": None},
}


def net_of(pnl: list[dict], market: dict[str, str], costs) -> tuple[float, int, int]:  # noqa: ANN001
    """(net, trades, losers) recomputed with per-market costs."""
    total, trades, losers = 0.0, 0, 0
    for r in pnl:
        if not r["buy_value"]:
            continue
        buy_c, sell_c = costs.get(market.get(r["ticker"], "거래소"), costs["거래소"])
        proceeds = r["sell_value"] + r["leftover_value"]
        net = float(proceeds * (1 - sell_c) - r["buy_value"] * (1 + buy_c))
        total += net
        trades += 1
        losers += net < 0
    return total, trades, losers


def main() -> None:
    from bayes_judge import judge
    from replay_common import CACHE, load_day, run_day
    from replay_parity import DART, _export_quant_airflow_env

    from daytrade_it.infrastructure.external.quant_airflow_db_client import QuantAirflowDBClient
    from daytrade_it.shared.runtime_host import require_backtest_host

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--dart", type=Path, default=DART)
    ap.add_argument("--cache-dir", type=Path, default=CACHE)
    ap.add_argument("--work-root", type=Path, default=Path("data/eval/rl_minimal/cost_work"))
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
    kosdaq = sum(1 for v in market.values() if v == "코스닥")
    print(f"days={len(days)} tickers={len(market)} KOSDAQ share {kosdaq / len(market):.3f}")

    base = NewsPolicy()
    policies = {"base (live)": base}
    for name, kw in CANDIDATES.items():
        policies[name] = dataclasses.replace(base, limits=dataclasses.replace(base.limits, **kw))

    res: dict[str, dict] = {n: {} for n in policies}
    for day in days:
        data = load_day(day, args.cache_dir, args.events)
        for name, pol in policies.items():
            _r, pnl = run_day(data, pol, stocks, args.dart, args.work_root)
            old = net_of(pnl, market, {"거래소": OLD, "코스닥": OLD})
            new = net_of(pnl, market, COSTS)
            res[name][str(day)] = {"old": old, "new": new}

    print(
        f"\n{'policy':16s} {'trades':>7s} {'OLD net':>11s} {'TRUE net':>11s} {'delta':>10s}"
        f" {'vs base OLD':>12s} {'vs base TRUE':>13s}"
    )
    base_old = sum(v["old"][0] for v in res["base (live)"].values())
    base_new = sum(v["new"][0] for v in res["base (live)"].values())
    rows_out = {}
    for name, days_d in res.items():
        o = sum(v["old"][0] for v in days_d.values())
        n = sum(v["new"][0] for v in days_d.values())
        t = sum(v["old"][1] for v in days_d.values())
        rows_out[name] = (o, n, t)
        print(
            f"{name:16s} {t:>7d} {o:>+11,.0f} {n:>+11,.0f} {n - o:>+10,.0f}"
            f" {o - base_old:>+12,.0f} {n - base_new:>+13,.0f}"
        )

    print()
    base_days = {
        d: {
            "day": d,
            "net": v["new"][0],
            "trades": v["new"][1],
            "losers": v["new"][2],
            "worst_trade": 0.0,
            "errors": 0,
            "leftovers": 0,
        }
        for d, v in res["base (live)"].items()
    }
    for name, days_d in res.items():
        if name == "base (live)":
            continue
        cand = {
            d: {
                "day": d,
                "net": v["new"][0],
                "trades": v["new"][1],
                "losers": v["new"][2],
                "worst_trade": 0.0,
                "errors": 0,
                "leftovers": 0,
            }
            for d, v in days_d.items()
        }
        v = judge(cand, base_days)
        print(
            f"{name:16s} TRUE-cost p(mu>0)={v['p_mu_gt_0']:.3f} mu={v['mu_mean']:+,.0f}"
            f" days>=0={v['days_ge_0']}/{v['n_days']} losers {v['loser_ratio_candidate']:.3f}"
            f" vs {v['loser_ratio_baseline']:.3f} adopt={v['adopt']}"
        )


if __name__ == "__main__":
    main()
