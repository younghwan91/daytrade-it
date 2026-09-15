"""Replay one cached day under one NewsPolicy -> net P&L (the optimizer's objective).

`load_day` builds a day's inputs once from kr_quant (`replay_parity.prepare_day`)
and pickles them under `data/eval/cache/days/`; later calls read the pickle, so
a trial costs only the replay. `day_net` runs one policy over one day with a
fresh `SimBroker` and its own run directory, so trials never share state.
"""

from __future__ import annotations

import datetime as dt
import os
import pickle
import sys
import uuid
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from replay_baseline import VALID_TICKER, last_regular_close, trade_pnl

from daytrade_it.application.replay.day_replay import DayConfig, replay_day
from daytrade_it.application.replay.events import ReplayEvent, load_events
from daytrade_it.application.replay.prev_close import FrozenPrevCloseDB, FrozenPrevCloses
from daytrade_it.application.trading.policy import NewsPolicy
from daytrade_it.infrastructure.brokers.base import KST, MinuteBar
from daytrade_it.infrastructure.brokers.sim_broker import SimBroker
from daytrade_it.infrastructure.external.quant_airflow_db_client import StockInfo

CACHE = Path("data/eval/cache/days")


@dataclass
class DayData:
    day: dt.date
    events: list[ReplayEvent]
    bars: dict[str, list[MinuteBar]]
    frozen: FrozenPrevCloses
    stale_before: dt.datetime


def _build_day(day: dt.date) -> DayData:
    """From kr_quant, as `replay_baseline` does (non-equity tickers dropped)."""
    from replay_parity import EVENTS, prepare_day  # noqa: PLC0415

    events = load_events(EVENTS / f"{day}.jsonl")
    bad = sorted({e.ticker for e in events if not VALID_TICKER.match(e.ticker)})
    if bad:
        print(f"{day}: dropping non-equity ticker(s) {bad}", file=sys.stderr)
        events = [e for e in events if VALID_TICKER.match(e.ticker)]
    bars, frozen, _broker, stale_before = prepare_day(day, events)
    return DayData(day=day, events=events, bars=bars, frozen=frozen, stale_before=stale_before)


def load_day(day: dt.date, cache_dir: Path = CACHE) -> DayData:
    """The day's inputs, built and pickled on first use, read from the pickle after."""
    path = cache_dir / f"{day}.pkl"
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)  # noqa: S301 -- our own cache file
    data = _build_day(day)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
    os.replace(tmp, path)
    return data


def day_net(
    data: DayData,
    policy: NewsPolicy,
    stocks: dict[str, StockInfo],
    dart: Path,
    work_root: Path,
) -> dict:
    """Net P&L of `policy` on one day (costs included, leftovers at the last regular close).

    `errors` counts error logs during the replay; `net` is reported as computed
    either way, so the caller decides whether an errored day counts.
    """
    broker = SimBroker(
        data.bars,
        data.frozen.prices,
        lambda: dt.datetime.now(KST),
        limit_base=data.frozen.krx_prices,
    )
    res = replay_day(
        data.day,
        data.events,
        broker,
        FrozenPrevCloseDB(data.frozen),
        stocks,
        data.stale_before,
        DayConfig(policy=policy),
        work_root,
        dart,
    )
    pnl = trade_pnl(res.fills, res.positions, last_regular_close(data.bars))
    nets = [r["net"] for r in pnl]
    return {
        "day": data.day,
        "net": sum(nets, Decimal(0)),
        "trades": len(pnl),
        "losers": sum(n < 0 for n in nets),
        "worst_trade": min(nets, default=Decimal(0)),
        "errors": len(res.errors),
        "leftovers": {r["ticker"]: r["leftover_qty"] for r in pnl if r["leftover_qty"]},
    }
