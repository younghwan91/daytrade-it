"""Counterfactual reward dataset for the entry bandit (spec 2026-09-16-bandit-entry-design.md).

uv run python scripts/eval/bandit_dataset.py --dart data/eval/dart_trader_20260915.db  # simnode

Each day is replayed once with an UNCAPPED copy of the base policy (position,
entry and exposure caps lifted, SimBroker cash 10^12, bandit off; selection,
entry timing and exits as the base), so every candidate that reaches
`trader.enter` trades 1 share-sized entry independently of the others. One row
per such candidate:

- `features`: `news_entry.bandit_features` -- the live verdict's own function --
  evaluated at the pair's `decide_at` on a fresh SimBroker over the same day
  data (only completed bars and the 069500 quote at that instant). None if the
  069500 price or prior close is unavailable.
- `reward` = net / buy value of that ticker's trade (`trade_pnl`: costs,
  add-ons, leftovers at the last regular close); 0.0 if entered or deferred but
  never filled (`status` unfilled/expired). None -- not a training row -- when
  `trader.enter` skipped it (e.g. `skipped:already entered today`: the
  ticker's one entry of the day went to an earlier pair) or the enter failed.

Output: `--out` (default data/eval/bandit/dataset.jsonl) and a per-day summary
(rows, trainable rows, reward mean/std, fraction of constant feature columns).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import multiprocessing
import os
import statistics
import sys
from decimal import Decimal
from functools import partial
from pathlib import Path

import time_machine

from daytrade_it.application.replay.day_replay import SESSION_CLOSE
from daytrade_it.application.replay.events import features_of, to_article
from daytrade_it.application.replay.prev_close import FrozenPrevCloseDB
from daytrade_it.application.signals import news_entry
from daytrade_it.application.trading.bandit import FEATURES
from daytrade_it.application.trading.policy import NewsPolicy
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.brokers.base import KST
from daytrade_it.infrastructure.brokers.sim_broker import SimBroker

UNCAPPED = 1000
UNCAPPED_CASH = Decimal(10**12)
OUT = Path("data/eval/bandit/dataset.jsonl")
WORK = Path("data/eval/bandit_work")
#: Decision outcomes that reached `trader.enter`.
REACHED_ENTER = {"entered", "deferred", "skipped", "enter_failed"}


def uncapped(base: NewsPolicy) -> NewsPolicy:
    """`base` with the caps lifted and the bandit off (a bandit rank key -> surprise)."""
    sel = base.selection
    return NewsPolicy(
        limits=dataclasses.replace(
            base.limits,
            max_open_positions=UNCAPPED,
            max_entries_per_day=UNCAPPED,
            max_trader_exposure=UNCAPPED_CASH,
        ),
        selection=dataclasses.replace(
            sel,
            bandit_model=None,
            rank_key="surprise" if sel.rank_key == "bandit" else sel.rank_key,
        ),
    )


def decision_features(data, event, pre_entry_move: float) -> dict[str, float] | None:  # noqa: ANN001
    """`news_entry.bandit_features` at `event.decide_at` over the day's data (fresh broker)."""
    broker = SimBroker(
        data.bars,
        data.frozen.prices,
        lambda: dt.datetime.now(KST),
        limit_base=data.frozen.krx_prices,
    )
    expires_at = dt.datetime.combine(data.day, SESSION_CLOSE, tzinfo=KST)
    news_entry._MARKET_CACHE.clear()
    with time_machine.travel(event.decide_at, tick=False):
        return news_entry.bandit_features(
            to_article(event),
            Ticker(value=event.ticker),
            features_of(event),
            pre_entry_move,
            broker,
            FrozenPrevCloseDB(data.frozen),
            expires_at,
        )


def _status_reward(decision: dict, pnl_row: dict | None) -> tuple[str, float | None]:
    outcome = decision["outcome"]
    if outcome == "skipped":
        return f"skipped:{decision.get('reason')}", None
    if outcome == "enter_failed":
        return "enter_failed", None
    if pnl_row is None or not pnl_row["buy_value"]:
        return ("expired" if outcome == "deferred" else "unfilled"), 0.0
    return "filled", float(pnl_row["net"] / pnl_row["buy_value"])


def day_rows(data, base: NewsPolicy, stocks: dict, dart: Path, work_root: Path) -> list[dict]:  # noqa: ANN001
    """One row per candidate that reached `trader.enter` in the uncapped replay of `data`."""
    from replay_common import run_day  # noqa: PLC0415

    res, pnl = run_day(data, uncapped(base), stocks, dart, work_root, cash=UNCAPPED_CASH)
    if res.errors:
        raise RuntimeError(f"{data.day}: {len(res.errors)} error(s) logged in the uncapped replay")
    by_ticker = {r["ticker"]: r for r in pnl}
    events = {e.key: e for e in data.events}
    rows = []
    claimed: set[str] = set()
    for d in res.decisions:
        if d["outcome"] not in REACHED_ENTER:
            continue
        e = events[d["key"]]
        status, reward = _status_reward(d, by_ticker.get(d["ticker"]))
        if reward is not None:
            if d["ticker"] in claimed:  # one entry per ticker per day: never reached here
                raise RuntimeError(f"{data.day} {d['ticker']}: two entries in one day")
            claimed.add(d["ticker"])
        pre = float(Decimal(d["pre_entry_move"]))
        rows.append(
            {
                "day": str(data.day),
                "key": d["key"],
                "ticker": d["ticker"],
                "decide_at": e.decide_at.isoformat(),
                "outcome": d["outcome"],
                "status": status,
                "reward": reward,
                "features": decision_features(data, e, pre),
            }
        )
    return rows


def trainable(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["reward"] is not None and r["features"] is not None]


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for day in sorted({r["day"] for r in rows}):
        all_day = [r for r in rows if r["day"] == day]
        train = trainable(all_day)
        rewards = [r["reward"] for r in train]
        constant = sum(len({r["features"][f] for r in train}) <= 1 for f in FEATURES)
        out.append(
            {
                "day": day,
                "rows": len(all_day),
                "trainable": len(train),
                "filled": sum(r["status"] == "filled" for r in all_day),
                "no_market": sum(r["features"] is None for r in all_day),
                "reward_mean": statistics.fmean(rewards) if rewards else 0.0,
                "reward_std": statistics.pstdev(rewards) if rewards else 0.0,
                "constant_features": constant / len(FEATURES) if train else 1.0,
                "statuses": {
                    s: sum(r["status"] == s for r in all_day)
                    for s in sorted({r["status"] for r in all_day})
                },
            }
        )
    return out


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


_WORKER: dict = {}


def _init(
    stocks: dict,
    dart: Path,
    work_root: Path,
    cache_dir: Path,
    base: NewsPolicy,
    events_dir: Path | None = None,
) -> None:
    _WORKER.update(stocks=stocks, dart=dart, work=work_root / str(os.getpid()), cache=cache_dir)
    _WORKER["base"] = base
    _WORKER["events"] = events_dir


def _day_job(day: dt.date) -> list[dict]:
    from replay_common import load_day  # noqa: PLC0415

    w = _WORKER
    data = load_day(day, w["cache"], w["events"])
    return day_rows(data, w["base"], w["stocks"], w["dart"], w["work"])


def main(argv: list[str] | None = None) -> None:
    from replay_common import CACHE, load_day  # noqa: PLC0415
    from replay_parity import DART, EVENTS, _export_quant_airflow_env  # noqa: PLC0415

    from daytrade_it.application.trading.policy import load_policy  # noqa: PLC0415
    from daytrade_it.infrastructure.external.quant_airflow_db_client import (  # noqa: PLC0415
        QuantAirflowDBClient,
    )
    from daytrade_it.shared.runtime_host import require_backtest_host  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-policy", type=Path, help="default: live defaults")
    ap.add_argument("--days", help="comma-separated; default: every replay events day")
    ap.add_argument(
        "--events",
        type=Path,
        default=None,
        help="events directory (default data/eval/replay_events)",
    )
    ap.add_argument("--dart", type=Path, default=DART)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--work-root", type=Path, default=WORK)
    ap.add_argument("--cache-dir", type=Path, default=CACHE)
    ap.add_argument("--jobs", type=int, default=7)
    args = ap.parse_args(argv)

    require_backtest_host()
    if args.base_policy is not None and not args.base_policy.exists():
        ap.error(f"--base-policy {args.base_policy} does not exist")
    base = load_policy(args.base_policy) if args.base_policy else NewsPolicy()
    events_dir = args.events or EVENTS
    days = (
        [dt.date.fromisoformat(d) for d in args.days.split(",")]
        if args.days
        else [dt.date.fromisoformat(p.stem) for p in sorted(events_dir.glob("2026-*.jsonl"))]
    )
    _export_quant_airflow_env()
    stocks = QuantAirflowDBClient().get_stock_master()
    for d in days:
        load_day(d, args.cache_dir, events_dir)
    init = partial(_init, stocks, args.dart, args.work_root, args.cache_dir, base, events_dir)
    with multiprocessing.Pool(min(args.jobs, len(days)), init) as pool:
        per_day = pool.map(_day_job, days, chunksize=1)
    rows = [r for day in per_day for r in day]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    no_market = 0
    for s in summarize(rows):
        print(json.dumps(s, ensure_ascii=False))
        no_market += s["no_market"]
    print(f"{len(rows)} rows ({len(trainable(rows))} trainable) -> {args.out}", file=sys.stderr)
    if no_market:
        print(f"ERROR: {no_market} row(s) without 069500 data (not trainable)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
