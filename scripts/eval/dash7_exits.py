"""Re-run the dashboard-pool exit candidates through the REAL replay engine.

Context: the offline exit lab I built today had a live-parity bug (a stock that
prints no 15:00 bar cannot be sold at 15:00). The question this answers is
whether that bug ever touched the 7-day test whose result currently keeps
trailing and stop-loss switched off in live.

It could not have. The bug was in `scripts/eval/exit_lab.py` and
`scripts/eval/exit_fast.py`, both written today. The 7-day test ran through the
replay engine, which models the exit correctly:

- `auto_trader.manage()` calls `flatten()` on EVERY tick at or after
  `flatten_at`, and `_exit_or_retry` re-prices an unfilled sell after
  `exit_retry_after` with a widening discount (capped).
- `SimBroker` holds sells as RESTING orders, filling on whichever later bar
  satisfies `b.high >= limit`.

Neither depends on a bar existing at 15:00 exactly. This script re-runs the
candidates anyway, so the claim rests on a measurement rather than a reading of
the code.

Candidates: the live default; `trail_2_1`, the trailing configuration whose
p=0.016 result turned trailing off; `orb30`, the best single change in that same
table (p=0.801); and `consensus`, the configuration my sweep-pool monthly folds
selected (trailing on in 5/5 folds, stop_orb 30 in 4/5).

Seven days cannot clear the adoption bar and this does not try to.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from pathlib import Path

from daytrade_it.application.trading.policy import NewsPolicy

#: A COORDINATE sweep, not a grid: each entry moves ONE live knob off its current
#: value, everything else at the live default. With 7 days and ~35 trades a full
#: cross would be a multiple-comparison machine; the question worth asking here is
#: narrower and answerable -- "is the hand-set value near the optimum on this axis,
#: or clearly off it?" Nothing here is a selection.
#:
#: These three knobs are the live-expressible exit surface that is currently ON,
#: and all three were set by leave-one-day-out over 6 days -- the methodology
#: `learning-curve` showed is invalid (40 LODO folds sharing 39/40 training days
#: are one model forty times over).
#:
#: `vwap_exit_hit` and `add_on_hit` are byte-identical between this branch and
#: origin/main (verified by diff), so a result here transfers to live.
#: Entry-window sweep. `entry_start` / `entry_end` ARE on main (unlike the exit
#: knobs), and the enforcement is identical between branches -- my branch has one
#: extra gate inside `_deferred_entries`, which is inert because nothing is ever
#: deferred with entry timing at its defaults (verified).
#:
#: Motivation: on the dashboard pool the return decays monotonically through the
#: day (09:00-09:01 +166 bp, 09:02-10:00 +42, 10:00-12:00 -63, 12:00+ -58), while
#: detection DELAY shows nothing (spearman +0.051). Time of day is the candidate;
#: latency is not.
#:
#: Note the trade-count effect runs the other way from an exit change: narrowing
#: the window removes entries outright rather than freeing slots, so this is a
#: comparison against a smaller book.
CANDIDATES: dict[str, dict] = {}
for _h, _m in ((10, 0), (11, 0), (12, 0), (13, 0)):
    CANDIDATES[f"entry_end={_h:02d}:{_m:02d}"] = {"entry_end": dt.time(_h, _m)}
#: If the open batch carries the effect, the other half is whether we should take
#: MORE of it -- the 09:01 batch currently competes for the same five slots as the
#: whole day.
for _n in (8, 10):
    CANDIDATES[f"max_open={_n}"] = {"max_open_positions": _n}
for _n in (15, 20):
    CANDIDATES[f"max_entries={_n}"] = {"max_entries_per_day": _n}
CANDIDATES["open_only+max_open=10"] = {
    "entry_end": dt.time(10, 0),
    "max_open_positions": 10,
}


def main() -> None:
    from bayes_judge import judge
    from replay_common import CACHE, day_net, load_day
    from replay_parity import DART, _export_quant_airflow_env

    from daytrade_it.infrastructure.external.quant_airflow_db_client import QuantAirflowDBClient
    from daytrade_it.shared.runtime_host import require_backtest_host

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--dart", type=Path, default=DART)
    ap.add_argument("--cache-dir", type=Path, default=CACHE)
    ap.add_argument("--work-root", type=Path, default=Path("data/eval/rl_minimal/dash7_work"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    require_backtest_host()
    _export_quant_airflow_env()
    stocks = QuantAirflowDBClient().get_stock_master()
    days = [dt.date.fromisoformat(p.stem) for p in sorted(args.events.glob("2026-*.jsonl"))]
    print(f"days: {[str(d) for d in days]}")

    base = NewsPolicy()
    policies = {"base": base}
    for name, kw in CANDIDATES.items():
        policies[name] = dataclasses.replace(base, limits=dataclasses.replace(base.limits, **kw))

    results: dict[str, list[dict]] = {name: [] for name in policies}
    for day in days:
        data = load_day(day, args.cache_dir, args.events)
        for name, pol in policies.items():
            results[name].append(day_net(data, pol, stocks, args.dart, args.work_root))
        line = "  ".join(f"{n}={float(results[n][-1]['net']):>+8,.0f}" for n in policies)
        errs = sum(int(results[n][-1]["errors"]) for n in policies)
        print(f"{day}  {line}   errors={errs}")

    print()
    base_trades = sum(int(r["trades"]) for r in results["base"])
    print(
        f"{'policy':14s} {'total':>10s} {'trades':>7s} {'d_trades':>9s}"
        f" {'losers':>7s} {'worst day':>11s}"
    )
    for name, rows in results.items():
        tot = sum(float(r["net"]) for r in rows)
        tr = sum(int(r["trades"]) for r in rows)
        lo = sum(int(r["losers"]) for r in rows)
        print(
            f"{name:14s} {tot:>+10,.0f} {tr:>7d} {tr - base_trades:>+9d} {lo:>7d}"
            f" {min(float(r['net']) for r in rows):>+11,.0f}"
        )

    print()
    verdicts = {}
    base_map = {str(r["day"]): r for r in results["base"]}
    for name in CANDIDATES:
        cand = {str(r["day"]): r for r in results[name]}
        v = judge(cand, base_map)
        verdicts[name] = v
        print(
            f"{name:14s} p(mu>0)={v['p_mu_gt_0']:.3f} mu={v['mu_mean']:+,.0f}"
            f" days>=0={v['days_ge_0']}/{v['n_days']}"
            f" worst {v['worst_day_candidate']:+,.0f} vs {v['worst_day_baseline']:+,.0f}"
            f" losers {v['loser_ratio_candidate']:.3f} vs {v['loser_ratio_baseline']:.3f}"
            f" adopt={v['adopt']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"per_day": results, "verdicts": verdicts}, indent=1, default=str)
    )
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
