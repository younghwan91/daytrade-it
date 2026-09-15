"""Replay one cached day under one NewsPolicy -> net P&L (the optimizer's objective).

`load_day` builds a day's inputs once from kr_quant (`replay_parity.prepare_day`)
and pickles them under `data/eval/cache/days/`; later calls read the pickle, so
a trial costs only the replay. The pickle's name carries a hash of its inputs
(the events file, the frozen prev-close file and the day's minute-bar row
count), so a re-exported day or a re-frozen close can never feed a stale cache. `day_net` runs one policy over one day with a
fresh `SimBroker` and its own run directory, so trials never share state.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import pickle
import shutil
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


def _bars_row_count(day: dt.date, events_path: Path) -> int:
    """kr_quant minute_bars rows on `day` for the events' tickers (part of the cache key)."""
    from replay_parity import _conn  # noqa: PLC0415

    codes = sorted({e.ticker for e in load_events(events_path) if VALID_TICKER.match(e.ticker)})
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM minute_bars WHERE code = ANY(%s) AND ts >= %s AND ts < %s",
            (codes, day, day + dt.timedelta(days=1)),
        )
        (n,) = cur.fetchone()
    return int(n)


def _input_key(day: dt.date, events_dir: Path, prev_close_dir: Path) -> str:
    events_path = events_dir / f"{day}.jsonl"
    prev = prev_close_dir / f"{day}.json"
    h = hashlib.sha256()
    h.update(events_path.read_bytes())
    h.update(b"\0prev\0")
    h.update(prev.read_bytes() if prev.exists() else b"<none>")
    h.update(f"\0bars\0{_bars_row_count(day, events_path)}".encode())
    return h.hexdigest()[:16]


def load_day(
    day: dt.date,
    cache_dir: Path = CACHE,
    events_dir: Path | None = None,
    prev_close_dir: Path | None = None,
) -> DayData:
    """The day's inputs: read from the pickle whose input hash matches, else built.

    The key is taken again after a build, because the first build freezes the
    prev-close file; older pickles of the day are removed.
    """
    if events_dir is None or prev_close_dir is None:
        from replay_parity import EVENTS, PREV_CLOSE  # noqa: PLC0415

        events_dir = events_dir or EVENTS
        prev_close_dir = prev_close_dir or PREV_CLOSE
    path = cache_dir / f"{day}-{_input_key(day, events_dir, prev_close_dir)}.pkl"
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)  # noqa: S301 -- our own cache file
    data = _build_day(day)
    path = cache_dir / f"{day}-{_input_key(day, events_dir, prev_close_dir)}.pkl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
    os.replace(tmp, path)
    for old in cache_dir.glob(f"{day}-*.pkl"):
        if old != path:
            old.unlink(missing_ok=True)
    return data


def day_net(
    data: DayData,
    policy: NewsPolicy,
    stocks: dict[str, StockInfo],
    dart: Path,
    work_root: Path,
    keep_run_dir: bool = False,
) -> dict:
    """Net P&L of `policy` on one day (costs included, leftovers at the last regular close).

    `errors` counts error logs during the replay; `net` is reported as computed
    either way, so the caller decides whether an errored day counts. The run
    directory is removed afterwards unless `keep_run_dir` (thousands of trials).
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
    if not keep_run_dir and res.run_dir is not None:
        shutil.rmtree(res.run_dir, ignore_errors=True)
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
