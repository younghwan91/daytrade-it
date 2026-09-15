"""Replay one past session through the live trader.

The daemon's loop (`scripts/news_signal_poller.py::run`) is reproduced as a
timeline: manage ticks every 30 s, each pair judged at its `decide_at`, each
poll's candidates entered at `batch_at` in `rank_batch` order. The clock is
frozen with time-machine at each instant, so every `datetime.now()` inside
the live code -- signal expiry, the no-chase filter's "today", DART lookback --
sees the replayed time. Only the broker (`SimBroker`) and three module paths
(no-chase cap, DART snapshot, kill-switch file) differ from live.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import time_machine
from structlog.processors import format_exc_info
from structlog.testing import capture_logs

from daytrade_it.application.replay.events import ReplayEvent, features_of, to_article
from daytrade_it.application.signals import entry_filter
from daytrade_it.application.signals.news_entry import (
    EntryCandidate,
    build_entry_candidate,
    tradable_tickers,
)
from daytrade_it.application.trading.auto_trader import AutoTrader, TradingLimits, rank_batch
from daytrade_it.domain.services.signal_generator import SignalConfig, SignalGenerator
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.brokers.base import KST
from daytrade_it.infrastructure.brokers.sim_broker import SimBroker
from daytrade_it.infrastructure.external.quant_airflow_db_client import StockInfo
from daytrade_it.interfaces.mcp import tools
from daytrade_it.shared.logging import get_logger

logger = get_logger(__name__)

FIRST_POLL = dt.time(9, 0, 30)
SESSION_CLOSE = dt.time(15, 20)
_ERROR_LEVELS = {"error", "critical"}
MAX_MISSING_BARS = 0.05
_SKIP_EVENTS = {"auto_trader: entry skipped", "entry_filter: extended, no entry"}


class ReplayDataError(RuntimeError):
    """Input data can't support a faithful replay of the day (fail the day loudly)."""


def _enter_outcome(new_logs: list[dict], ticker: str, decision: dict) -> None:
    """Fill `decision` from the log rows `trader.enter` emitted for `ticker`."""
    for r in new_logs:
        if r.get("ticker") != ticker:
            continue
        ev = r.get("event")
        if ev == "auto_trader: ENTRY SUBMITTED":
            decision.update(outcome="entered", limit=r.get("limit"))
            return
        if ev == "auto_trader: entry skipped":
            decision.update(outcome="skipped", reason=r.get("reason"))
            return
    decision.update(outcome="enter_failed")


@dataclass(frozen=True)
class DayConfig:
    limits: TradingLimits
    max_pre_entry_move: Decimal
    manage_every: dt.timedelta = dt.timedelta(seconds=30)
    exits_until: dt.time = dt.time(15, 30)
    #: Poller before d357b88 (live through 09-14): each BUY went to
    #: `trader.enter` right after its judgment, in judgment order, unranked.
    enter_on_decide: bool = False


@dataclass
class DayResult:
    ledger: list[dict] = field(default_factory=list)
    skips: list[dict] = field(default_factory=list)
    fills: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    #: One row per judged pair, in decide order: ticker, decide_at, outcome
    #: (not_tradable | filter_failed | no_signal | extended | entered | skipped
    #: | enter_failed) plus pre_entry_move / reason / limit where known.
    decisions: list[dict] = field(default_factory=list)
    #: `execute_trade` pre-order refusals (ledger `entry_refused`: risk gate,
    #: DART, price band, kill switch).
    refusals: list[dict] = field(default_factory=list)
    #: Resting BUYs SimBroker cancelled at fill time for lack of cash.
    cash_rejections: list[dict] = field(default_factory=list)
    #: Shares held at the broker when the timeline ended (leftovers if > 0).
    positions: dict[str, int] = field(default_factory=dict)
    #: Event tickers with no minute bar that day (<= 5%, else the day fails).
    missing_bars: list[str] = field(default_factory=list)
    #: This call's own directory (ledger under it); never shared between calls.
    run_dir: Path | None = None


def snapshot_dart(src: Path, day: dt.date, dst: Path) -> Path:
    """Copy of dart.db without filings received after `day` (no look-ahead)."""
    if dst.resolve() == src.resolve():
        raise ValueError(f"snapshot destination must differ from source: {dst}")
    dst.unlink(missing_ok=True)
    con = sqlite3.connect(f"file:{dst}", uri=True)  # uri=True so ATTACH honours mode=ro
    con.execute("ATTACH DATABASE ? AS src", (f"file:{src}?mode=ro",))
    con.execute(
        "CREATE TABLE disclosures AS SELECT * FROM src.disclosures WHERE rcept_dt <= ?",
        (day.strftime("%Y%m%d"),),
    )
    con.commit()
    con.execute("DETACH DATABASE src")
    con.close()
    return dst


def dart_snapshot(src: Path, day: dt.date, cache_dir: Path) -> Path:
    """Read-only no-look-ahead snapshot, built once per (source content, day).

    Fails the day if the source has no filing received on or after `day`: a
    lagging copy would silently hide that day's hard disclosures.
    """
    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        (latest,) = con.execute("SELECT max(rcept_dt) FROM disclosures").fetchone()
    finally:
        con.close()
    if latest is None or latest < day.strftime("%Y%m%d"):
        raise ReplayDataError(
            f"DART source {src} max(rcept_dt)={latest} is before {day}: refresh or pass "
            "a newer copy (--dart)"
        )
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    dst = cache_dir / f"{digest}-{day:%Y%m%d}.db"
    if dst.exists():
        return dst
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f".{dst.name}.{uuid.uuid4().hex}.tmp"
    snapshot_dart(src, day, tmp)
    tmp.chmod(0o444)
    os.replace(tmp, dst)
    return dst


def missing_bar_tickers(events: list[ReplayEvent], broker: SimBroker) -> list[str]:
    """Event tickers without bars; raises if they exceed 5% of event tickers."""
    codes = sorted({e.ticker for e in events})
    missing = [c for c in codes if not broker.has_bars(c)]
    if codes and len(missing) / len(codes) > MAX_MISSING_BARS:
        raise ReplayDataError(
            f"{len(missing)}/{len(codes)} event tickers have no minute bars "
            f"(> {MAX_MISSING_BARS:.0%}): {missing}"
        )
    return missing


def _timeline(
    day: dt.date, events: list[ReplayEvent], cfg: DayConfig
) -> list[tuple[dt.datetime, int, Any]]:
    items: list[tuple[dt.datetime, int, Any]] = []
    t = dt.datetime.combine(day, FIRST_POLL, tzinfo=KST)
    end = dt.datetime.combine(day, cfg.exits_until, tzinfo=KST)
    while t <= end:
        items.append((t, 0, None))
        t += cfg.manage_every
    for e in events:
        items.append((e.decide_at, 1, e))
        items.append((e.batch_at, 2, e.batch_at))
    items.sort(key=lambda x: (x[0], x[1]))
    return items


def replay_day(
    day: dt.date,
    events: list[ReplayEvent],
    broker: SimBroker,
    db_client: Any,
    stocks: dict[str, StockInfo],
    stale_before: dt.datetime,
    config: DayConfig,
    work_dir: Path,
    dart_db: Path,
    clock_holder: dict | None = None,
) -> DayResult:
    missing = missing_bar_tickers(events, broker)
    snapshot = dart_snapshot(dart_db, day, work_dir / "dart_snapshots")
    # A fresh directory per call: AutoTrader reloads an existing day's ledger,
    # so two replays (trials) must never share one.
    run_dir = work_dir / f"{day.isoformat()}-{uuid.uuid4().hex[:12]}"
    ledger_dir = run_dir / "ledger"
    path = ledger_dir / f"{day.isoformat()}.jsonl"
    ledger_dir.mkdir(parents=True)
    expires_at = dt.datetime.combine(day, SESSION_CLOSE, tzinfo=KST)
    saved = (entry_filter.MAX_PRE_ENTRY_MOVE, tools._DART_DB_PATH, tools._KILL_SWITCH_PATH)
    n_fills, n_rejections = len(broker.fills), len(broker.rejections)
    positions: dict[str, int] = {}
    if clock_holder is not None:
        clock_holder["now"] = lambda: dt.datetime.now(KST)
    timeline = _timeline(day, events, config)
    decisions: list[dict] = []
    try:
        entry_filter.MAX_PRE_ENTRY_MOVE = config.max_pre_entry_move
        tools._DART_DB_PATH = snapshot
        tools._KILL_SWITCH_PATH = run_dir / "no_kill_switch"
        with (
            time_machine.travel(timeline[0][0], tick=False) as traveller,
            capture_logs(processors=[format_exc_info]) as logs,
        ):
            asyncio.run(
                _run(
                    logs,
                    decisions,
                    timeline,
                    traveller,
                    broker,
                    db_client,
                    stocks,
                    stale_before,
                    expires_at,
                    config,
                    ledger_dir,
                )
            )
            try:
                positions = {p.ticker: int(p.quantity) for p in broker.get_positions()}
            except Exception:
                logger.exception("replay: end-of-day positions lookup failed")
    finally:
        entry_filter.MAX_PRE_ENTRY_MOVE, tools._DART_DB_PATH, tools._KILL_SWITCH_PATH = saved
    ledger = (
        [json.loads(line) for line in path.read_text().splitlines() if line]
        if path.exists()
        else []
    )
    skips = [
        {"ticker": r.get("ticker"), "reason": r.get("reason", "extended")}
        for r in logs
        if r.get("event") in _SKIP_EVENTS
    ]
    errors = [r for r in logs if r.get("log_level") in _ERROR_LEVELS]
    return DayResult(
        ledger=ledger,
        skips=skips,
        fills=broker.fills[n_fills:],
        errors=errors,
        decisions=decisions,
        refusals=[r for r in ledger if r["event"] == "entry_refused"],
        cash_rejections=broker.rejections[n_rejections:],
        positions=positions,
        missing_bars=missing,
        run_dir=run_dir,
    )


async def _run(
    logs: list[dict],
    decisions: list[dict],
    timeline: list[tuple[dt.datetime, int, Any]],
    traveller: Any,
    broker: SimBroker,
    db_client: Any,
    stocks: dict[str, StockInfo],
    stale_before: dt.datetime,
    expires_at: dt.datetime,
    config: DayConfig,
    ledger_dir: Path,
) -> None:
    trader = AutoTrader(
        broker, ledger_dir, live=True, limits=config.limits, clock=lambda: dt.datetime.now(KST)
    )
    gen = SignalGenerator(SignalConfig(min_confidence=0.0))
    pending: dict[dt.datetime, list[EntryCandidate]] = {}
    by_signal: dict[Any, dict] = {}
    for at, kind, payload in timeline:
        traveller.move_to(at)
        if kind == 0:
            try:
                trader.manage()
            except Exception:
                logger.exception("replay: manage failed")
        elif kind == 1:
            e: ReplayEvent = payload
            decision: dict = {"ticker": e.ticker, "decide_at": e.decide_at}
            decisions.append(decision)
            article = to_article(e)
            if not any(
                t.value == e.ticker
                for t, _ in tradable_tickers(article, stocks, stale_before, expires_at, at)
            ):
                decision["outcome"] = "not_tradable"
                continue
            n = len(logs)
            try:
                c = build_entry_candidate(
                    article,
                    Ticker(value=e.ticker),
                    features_of(e),
                    broker,
                    db_client,
                    gen,
                    expires_at,
                )
            except Exception:
                logger.exception("replay: entry filter failed", ticker=e.ticker)
                decision["outcome"] = "filter_failed"
                continue
            if c is None:
                ext = [
                    r
                    for r in logs[n:]
                    if r.get("event") == "entry_filter: extended, no entry"
                    and r.get("ticker") == e.ticker
                ]
                if ext:
                    decision.update(outcome="extended", pre_entry_move=ext[-1]["pre_entry_move"])
                else:
                    decision["outcome"] = "no_signal"
                continue
            decision["pre_entry_move"] = c.signal.metadata.get("pre_entry_move")
            if config.enter_on_decide:
                n = len(logs)
                try:
                    await trader.enter(c.signal)
                except Exception:
                    logger.exception("replay: enter failed", ticker=e.ticker)
                _enter_outcome(logs[n:], e.ticker, decision)
            else:
                decision["outcome"] = "pending"
                by_signal[c.signal.id] = decision
                pending.setdefault(e.batch_at, []).append(c)
        else:
            batch = pending.pop(payload, [])
            if not batch:
                continue
            for signal in rank_batch([(c.signal, c.surprise, c.score) for c in batch]):
                n = len(logs)
                try:
                    await trader.enter(signal)
                except Exception:
                    logger.exception("replay: enter failed", ticker=signal.ticker.value)
                _enter_outcome(logs[n:], signal.ticker.value, by_signal[signal.id])
    trader.report_leftovers()
