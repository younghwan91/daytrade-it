"""Automated execution for news signals: enter 1 share, flat by the close.

Mirrors the rules that were evaluated (docs/research/2026-09-13-news-prompt-
eval.md, docs/research/2026-09-15-news-policy-optimization.md on branch
feat/exit-rule-eval): enter after the news; exit early when a position that
gained >= +2% breaks 1% below the session VWAP (1-minute bars), otherwise at
the session close. No fixed stop-loss or take-profit -- both lost to holding
in the evaluation, and with a 1-share cap the loss per position is bounded by
the share price. The poller ranks each polled batch with `rank_batch` before
entering, so when the slots fill at the open the highest-`surprise` names get
them.

Pyramiding (불타기, from 2026-09-15): once per ticker, 1 more share when the
position is up >= +1% and the last completed 1-minute close is above the
session VWAP with MA5 > MA20. It cannot go through `execute_trade` (that
refuses a BUY on a held ticker), so it goes straight to the broker with the
entry safeguards rebuilt here: 1 share LIMIT at the ask, kill switch, DART
hard-disclosure gate (fail-closed), +/-3% band vs the last close, deposit
cash, and a cap on what the trader holds (`max_trader_exposure`).

Safety design follows scalp-it's order-safety audit
(scalp-it docs/research/manju/41-order-safety-audit.md):

- C-1 live gate: orders go out only when constructed with `live=True`, which
  the poller sets only from two explicit CLI flags (`--execute
  --confirm-real`). No env var can turn it on.
- C-2 user holdings: a holdings snapshot is taken once per day before the
  first order and kept in the ledger. Tickers held in it are never bought,
  and exits sell at most `held - baseline`, so the user's own shares are
  never sold.
- C-3 size: every entry goes through `execute_trade` (1-share hard cap,
  LIMIT-only, price band, kill switch, duplicate-entry and risk gates);
  one entry per ticker per day; daily and concurrent caps below.
- C-4/N-1 exits: sells go straight to the broker as LIMIT orders at the
  bid, bypassing entry gates and the kill switch -- blocking an exit would
  strand a position.
- N-2 crashes: every broker call is caught; one rejected order never stops
  the loop.
- N-3 fills: fills are judged from account positions, never from "the order
  is no longer in the unfilled list".
- Unknown outcome: if `execute_trade` raises unexpectedly (order may or may
  not be live), the ticker is marked entered so it is never retried.

State lives in an append-only JSONL ledger per trading day, so a crash-
restart resumes with the same baseline, entries, and exits.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

from krx_quant_core.market.ticks import tick_size

from daytrade_it.domain.entities.signal import SignalDirection, TradingSignal
from daytrade_it.domain.services.risk_manager import RiskConfig, RiskManager
from daytrade_it.infrastructure.brokers.base import (
    BrokerInterface,
    MinuteBar,
    Order,
    OrderSide,
    OrderType,
    now_kst,
)
from daytrade_it.shared.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TradingLimits:
    max_order_notional: Decimal = Decimal("400000")
    #: 5 from 2026-09-15 (user directive): on 09-14 a cap of 2 filled at
    #: 09:00 and skipped ~25 later signals, so the forward sample was 2 trades.
    #: Still bounded: scalp-it trades the same Kiwoom account and cash (~1.1M
    #: KRW deposit), and RiskManager's exposure check rejects new entries once
    #: the account holds more than ~500k KRW of value.
    max_open_positions: int = 5
    max_entries_per_day: int = 10
    entry_start: dt.time = dt.time(9, 0, 30)
    #: Leave time for the move the eval measured (entry -> close).
    entry_end: dt.time = dt.time(14, 30)
    entry_order_ttl: dt.timedelta = dt.timedelta(minutes=5)
    #: 15:00, with 1-minute retries that get more aggressive, so there are
    #: ~20 attempts before continuous trading ends at 15:20 and the closing
    #: auction (15:20-15:30) is a last chance. It was 15:10 with 3-minute
    #: retries while the daemon stopped at 15:20: a 15:19 attempt that didn't
    #: fill left the share held overnight.
    flatten_at: dt.time = dt.time(15, 0)
    exit_retry_after: dt.timedelta = dt.timedelta(minutes=1)
    #: Each retry prices further below the bid, capped. A sell limit below the
    #: bid still fills at the best bid in continuous trading; the cap bounds
    #: how far a thin book or the closing auction can take it.
    exit_discount_step: Decimal = Decimal("0.005")
    exit_discount_cap: Decimal = Decimal("0.02")
    #: Early exit: sell when the last completed 1-minute close is at least
    #: `vwap_exit_min_gain` above the entry limit and more than
    #: `vwap_exit_gap` below the session VWAP (close x volume from 09:00).
    #: Chosen out-of-fold in 5 of 6 leave-one-day-out folds; +0.63%p per
    #: trade over holding to the close, better on 5 of 6 days. None disables.
    vwap_exit_gap: Decimal | None = Decimal("0.01")
    vwap_exit_min_gain: Decimal = Decimal("0.02")
    #: One ka10080 call per held ticker per interval (<= 5 per minute); the
    #: token is shared with scalp-it's live loop.
    exit_check_every: dt.timedelta = dt.timedelta(minutes=1)
    #: Pyramiding: out-of-fold over 6 days with the +3% no-chase cap, +144.9%p
    #: vs +92.4%p without it, better on 5 of 6 days; +1% chosen in 6/6 folds.
    add_on_min_gain: Decimal | None = Decimal("0.01")
    add_on_price_band: Decimal = Decimal("0.03")
    #: 1,000,000 KRW, user-approved 2026-09-15 for pyramiding. Caps add-ons
    #: (limit value of entries + add-ons not yet exited) and sets the
    #: RiskManager account-exposure check that gates entries (was ~500k).
    #: The deposit-cash check still applies (cash is shared with scalp-it).
    max_trader_exposure: Decimal = Decimal("1000000")
    # -- stop-loss / profit-taking exit rules (`exit_rule_hit`), all off by default --
    #: Stop-loss: sell when the last completed close <= entry x (1 - pct). None disables.
    stop_loss_pct: Decimal | None = None
    #: VWAP stop: close below entry and more than this gap below session VWAP. None disables.
    stop_vwap_gap: Decimal | None = None
    #: Time stop: held this long and still below entry x (1 + stop_time_min_gain). None disables.
    stop_time: dt.timedelta | None = None
    #: Minimum gain the time stop requires to keep holding (used only with stop_time).
    stop_time_min_gain: Decimal = Decimal("0")
    #: Opening-range stop: close below the low of 09:00 + this window, after it. None disables.
    stop_orb: dt.timedelta | None = None
    #: Take-profit: sell when the last completed close >= entry x (1 + pct). None disables.
    take_profit_pct: Decimal | None = None
    #: Trailing stop armed once the peak since entry reaches this gain. None disables.
    trail_after_gain: Decimal | None = None
    #: Trailing stop gap below the peak (used only with trail_after_gain).
    trail_gap: Decimal = Decimal("0.01")
    # -- entry timing (`entry_ready`), all off by default --
    #: Enter only once this long has passed since the signal. None disables.
    entry_delay: dt.timedelta | None = None
    #: Enter once a completed close is this far below the price at the signal. None disables.
    entry_pullback: Decimal | None = None
    #: Enter on the completed bar that closes back above VWAP after one closed below it.
    entry_vwap_reclaim: bool = False
    #: Longest a deferred entry waits for its conditions (then `entry_expired`).
    entry_max_wait: dt.timedelta = dt.timedelta(minutes=30)
    #: Entry limit = ask - tick_size(ask) x this; 0 buys at the ask.
    entry_limit_ticks: int = 0


_ENTRY_CLOSING_EVENTS = (
    "entry_submitted",
    "entry_unknown",
    "entry_refused",
    "entry_dry_run",
    "entry_expired",
)


@dataclass
class _DayState:
    baseline: dict[str, int] | None = None
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    open_entry_orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    exits: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Entry cancelled with zero fill: any later holding of that ticker is
    #: the user's (e.g. bought by hand), not the trader's.
    never_filled: set[str] = field(default_factory=set)
    exit_attempts: dict[str, int] = field(default_factory=dict)
    add_ons: dict[str, dict[str, Any]] = field(default_factory=dict)
    open_add_on_orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: `entry_deferred` rows still waiting for their timing conditions, by ticker.
    deferred: dict[str, dict[str, Any]] = field(default_factory=dict)


RANK_KEYS: tuple[str, ...] = ("surprise", "persistence", "score", "bandit")
_PERSISTENCE_RANK = {"one_off": 0, "multi_quarter": 1, "structural": 2}


def rank_batch(
    items: list[tuple[TradingSignal, int, float]], key: str = "surprise"
) -> list[TradingSignal]:
    """Entry order for one poll's signals (sorted stably, best first).

    - `surprise` (default): v2 `surprise` desc, then news score desc. At the
      open one poll yields more BUYs than there are slots; ranking by
      surprise beat random picks out-of-fold (+1.92 vs +1.37%p per trade).
    - `score`: news score desc, then surprise desc.
    - `persistence`: `signal.metadata["persistence"]` desc (structural >
      multi_quarter > one_off; missing ranks last), then surprise, then score.
    - `bandit`: `signal.metadata["bandit_m"]` desc (missing ranks last), then
      surprise, then score (spec 2026-09-16-bandit-entry-design.md).
    """
    if key == "surprise":
        sort_key = lambda x: (-x[1], -x[2])  # noqa: E731
    elif key == "score":
        sort_key = lambda x: (-x[2], -x[1])  # noqa: E731
    elif key == "persistence":
        sort_key = lambda x: (  # noqa: E731
            -_PERSISTENCE_RANK.get(x[0].metadata.get("persistence"), -1),
            -x[1],
            -x[2],
        )
    elif key == "bandit":
        sort_key = lambda x: (  # noqa: E731
            -float(x[0].metadata.get("bandit_m", float("-inf"))),
            -x[1],
            -x[2],
        )
    else:
        raise ValueError(f"unknown rank key {key!r}; expected one of {RANK_KEYS}")
    return [s for s, _, _ in sorted(items, key=sort_key)]


def vwap_exit_hit(
    bars: list[MinuteBar], entry: Decimal, now: dt.datetime, gap: Decimal, min_gain: Decimal
) -> tuple[bool, Decimal | None, Decimal | None]:
    """(hit, last completed close, VWAP). Only bars whose minute has ended count."""
    minute = now.replace(tzinfo=None, second=0, microsecond=0)
    # Regular session only, as evaluated: Kiwoom also returns NXT pre/after-market bars.
    done = [b for b in bars if b.ts.time() >= dt.time(9, 0) and b.ts < minute]
    volume = sum(b.volume for b in done)
    if not done or volume <= 0 or entry <= 0:
        return False, None, None
    vwap = sum(b.close * b.volume for b in done) / volume
    close = done[-1].close
    hit = close / entry - 1 >= min_gain and close / vwap - 1 < -gap
    return hit, close, vwap


def _any_exit_rule(lim: TradingLimits) -> bool:
    return any(
        v is not None
        for v in (
            lim.stop_loss_pct,
            lim.stop_vwap_gap,
            lim.stop_time,
            lim.stop_orb,
            lim.take_profit_pct,
            lim.trail_after_gain,
            lim.vwap_exit_gap,
        )
    )


def exit_rule_hit(
    bars: list[MinuteBar],
    entry: Decimal,
    entered_at: dt.datetime,
    now: dt.datetime,
    lim: TradingLimits,
) -> str | None:
    """Name of the first exit rule that fires on completed regular-session bars, else None.

    Order: stop_loss, stop_vwap, stop_time, stop_orb, take_profit, trailing, vwap_break.
    """
    minute = now.replace(tzinfo=None, second=0, microsecond=0)
    done = [b for b in bars if b.ts.time() >= dt.time(9, 0) and b.ts < minute]
    volume = sum(b.volume for b in done)
    if not done or volume <= 0 or entry <= 0:
        return None
    close = done[-1].close
    vwap = sum(b.close * b.volume for b in done) / volume

    if lim.stop_loss_pct is not None and close <= entry * (1 - lim.stop_loss_pct):
        return "stop_loss"
    if lim.stop_vwap_gap is not None and close < entry and close < vwap * (1 - lim.stop_vwap_gap):
        return "stop_vwap"
    if (
        lim.stop_time is not None
        and now.replace(tzinfo=None) - entered_at.replace(tzinfo=None) >= lim.stop_time
        and close < entry * (1 + lim.stop_time_min_gain)
    ):
        return "stop_time"
    if lim.stop_orb is not None:
        orb_end = dt.datetime.combine(done[-1].ts.date(), dt.time(9, 0)) + lim.stop_orb
        window = [b for b in done if b.ts < orb_end]
        # done[-1] after the window (so every window bar is complete) and below its low.
        if window and done[-1].ts >= orb_end and close < min(b.low for b in window):
            return "stop_orb"
    if lim.take_profit_pct is not None and close >= entry * (1 + lim.take_profit_pct):
        return "take_profit"
    if lim.trail_after_gain is not None:
        # From the minute after entry: the entry minute's high can predate the fill.
        after = entered_at.replace(tzinfo=None, second=0, microsecond=0) + dt.timedelta(minutes=1)
        since = [b.high for b in done if b.ts >= after]
        if since:
            peak = max(since)
            if peak >= entry * (1 + lim.trail_after_gain) and close <= peak * (1 - lim.trail_gap):
                return "trailing"
    if lim.vwap_exit_gap is not None:
        hit, _, _ = vwap_exit_hit(bars, entry, now, lim.vwap_exit_gap, lim.vwap_exit_min_gain)
        if hit:
            return "vwap_break"
    return None


def _any_entry_timing(lim: TradingLimits) -> bool:
    return lim.entry_delay is not None or lim.entry_pullback is not None or lim.entry_vwap_reclaim


def entry_ready(
    bars: list[MinuteBar],
    ref_price: Decimal,
    signaled_at: dt.datetime,
    now: dt.datetime,
    lim: TradingLimits,
) -> bool:
    """True when every enabled timing rule holds on completed regular-session bars.

    Pullback and VWAP reclaim look only at bars from the signal's minute on,
    so a dip that predates the signal never counts.
    """
    now_naive = now.replace(tzinfo=None)
    sig_naive = signaled_at.replace(tzinfo=None)
    if lim.entry_delay is not None and now_naive - sig_naive < lim.entry_delay:
        return False
    if lim.entry_pullback is None and not lim.entry_vwap_reclaim:
        return True
    minute = now_naive.replace(second=0, microsecond=0)
    sig_minute = sig_naive.replace(second=0, microsecond=0)
    done = [b for b in bars if b.ts.time() >= dt.time(9, 0) and b.ts < minute]
    if not done or done[-1].ts < sig_minute:
        return False
    close = done[-1].close
    if lim.entry_pullback is not None and not (
        ref_price > 0 and close <= ref_price * (1 - lim.entry_pullback)
    ):
        return False
    if lim.entry_vwap_reclaim:
        pv, vol = Decimal(0), 0
        below = False
        above_last = False
        for b in done:
            pv += b.close * b.volume
            vol += b.volume
            if b.ts < sig_minute or vol <= 0:
                continue
            vwap = pv / vol
            if b is done[-1]:
                above_last = b.close > vwap
            elif b.close < vwap:
                below = True
        if not (below and above_last):
            return False
    return True


def _drive(coro: Any) -> Any:
    """Run a coroutine that never suspends to completion, from sync code.

    `_enter_now` only awaits `execute_trade`, which only awaits
    `_SignalLookup.get_by_id` (returns at once), so a deferred entry can run
    from the sync `manage()` even while an event loop is running.
    """
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    coro.close()
    raise RuntimeError("coroutine suspended; cannot run it synchronously")


def add_on_hit(bars: list[MinuteBar], entry: Decimal, now: dt.datetime, min_gain: Decimal) -> bool:
    """Up >= min_gain vs entry, last completed close above VWAP, MA5 > MA20."""
    minute = now.replace(tzinfo=None, second=0, microsecond=0)
    done = [b for b in bars if b.ts.time() >= dt.time(9, 0) and b.ts < minute]
    volume = sum(b.volume for b in done)
    if len(done) < 20 or volume <= 0 or entry <= 0:
        return False
    close = done[-1].close
    vwap = sum(b.close * b.volume for b in done) / volume
    ma5 = sum(b.close for b in done[-5:]) / 5
    ma20 = sum(b.close for b in done[-20:]) / 20
    return close / entry - 1 >= min_gain and close > vwap and ma5 > ma20


def _disclosure_blocks(ticker: str) -> bool:
    """True if a hard-severity DART filing exists or the data can't vouch (fail-closed)."""
    from daytrade_it.infrastructure.external.dart_disclosure_db import (  # noqa: PLC0415
        DartDisclosureDB,
        DisclosureDataUnavailable,
    )
    from daytrade_it.interfaces.mcp import tools  # noqa: PLC0415

    try:
        return DartDisclosureDB(tools._DART_DB_PATH).hard_disclosure(ticker) is not None
    except DisclosureDataUnavailable:
        return True


def _kill_switch_on() -> bool:
    from daytrade_it.interfaces.mcp import tools  # noqa: PLC0415

    return tools._KILL_SWITCH_PATH.exists()


class _SignalLookup:
    """Minimal `signal_repository` for `execute_trade`'s `get_by_id`."""

    def __init__(self) -> None:
        self._signals: dict[UUID, TradingSignal] = {}

    def put(self, signal: TradingSignal) -> None:
        self._signals[signal.id] = signal

    def get(self, signal_id: str) -> TradingSignal | None:
        try:
            return self._signals.get(UUID(signal_id))
        except ValueError:
            return None

    async def get_by_id(self, signal_id: UUID) -> TradingSignal | None:
        return self._signals.get(signal_id)


class AutoTrader:
    def __init__(
        self,
        broker: BrokerInterface,
        ledger_dir: Path,
        *,
        live: bool,
        limits: TradingLimits | None = None,
        clock: Callable[[], dt.datetime] = now_kst,
    ) -> None:
        self._broker = broker
        self._live = live
        self._limits = limits or TradingLimits()
        self._clock = clock
        self._ledger_dir = ledger_dir
        self._signals = _SignalLookup()
        self._day: dt.date | None = None
        self._state = _DayState()
        #: In-memory only: after a restart the first check simply runs at once.
        self._last_exit_check: dict[str, dt.datetime] = {}
        self._last_entry_check: dict[str, dt.datetime] = {}

    # -- ledger ----------------------------------------------------------------

    def _ledger_path(self, day: dt.date) -> Path:
        return self._ledger_dir / f"{day.isoformat()}.jsonl"

    def _record(self, event: str, **payload: Any) -> None:
        self._ledger_dir.mkdir(parents=True, exist_ok=True)
        row = {"ts": self._clock().isoformat(), "event": event, "live": self._live, **payload}
        with self._ledger_path(self._today()).open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._apply(row)

    def _apply(self, row: dict[str, Any]) -> None:
        s, ev = self._state, row["event"]
        if ev == "baseline":
            s.baseline = {k: int(v) for k, v in row["positions"].items()}
        elif ev in ("entry_submitted", "entry_unknown"):
            s.entries[row["ticker"]] = row
            if row.get("order_id"):
                s.open_entry_orders[row["order_id"]] = row
        elif ev in ("entry_cancelled", "entry_resolved"):
            s.open_entry_orders.pop(row["order_id"], None)
            if ev == "entry_cancelled" and int(row.get("filled", 0)) == 0:
                s.never_filled.add(row["ticker"])
        elif ev in ("exit_submitted", "exit_failed"):
            # exit_failed is tracked too so flatten() waits exit_retry_after
            # instead of re-sending a rejected sell on every tick.
            s.exits[row["ticker"]] = row
            s.exit_attempts[row["ticker"]] = s.exit_attempts.get(row["ticker"], 0) + 1
        elif ev in ("add_on_submitted", "add_on_failed", "add_on_dry_run"):
            s.add_ons[row["ticker"]] = row
            if row.get("order_id"):
                s.open_add_on_orders[row["order_id"]] = row
        elif ev in ("add_on_cancelled", "add_on_resolved"):
            s.open_add_on_orders.pop(row["order_id"], None)
        if ev == "entry_deferred":
            s.deferred[row["ticker"]] = row
        elif ev in _ENTRY_CLOSING_EVENTS and row.get("ticker") in s.deferred:
            s.deferred.pop(row["ticker"])

    def _today(self) -> dt.date:
        return self._clock().date()

    def _ensure_day(self) -> None:
        today = self._today()
        if self._day == today:
            return
        self._day, self._state = today, _DayState()
        path = self._ledger_path(today)
        if path.exists():
            for line in path.open():
                self._apply(json.loads(line))
        # Deferred signals live in memory only: after a restart they can't be entered.
        for ticker, row in list(self._state.deferred.items()):
            if self._signals.get(row["signal_id"]) is None:
                self._record(
                    "entry_expired", ticker=ticker, signal_id=row["signal_id"], reason="restart"
                )

    def _ensure_baseline(self) -> bool:
        if self._state.baseline is not None:
            return True
        try:
            positions = self._broker.get_positions()
        except Exception:
            logger.exception("auto_trader: baseline holdings lookup failed (no entries)")
            return False
        self._record(
            "baseline", positions={p.ticker: int(p.quantity) for p in positions if p.quantity > 0}
        )
        return True

    # -- entries ---------------------------------------------------------------

    async def enter(self, signal: TradingSignal) -> None:
        self._ensure_day()
        ticker = signal.ticker.value
        now = self._clock()
        lim = self._limits

        def skip(reason: str) -> None:
            logger.info("auto_trader: entry skipped", ticker=ticker, reason=reason)

        if signal.direction != SignalDirection.BUY:
            return skip("not a BUY signal")
        if not (lim.entry_start <= now.time() < lim.entry_end):
            return skip("outside entry window")
        if ticker in self._state.entries:
            return skip("already entered today")
        if len(self._state.entries) >= lim.max_entries_per_day:
            return skip("daily entry cap reached")
        if not self._ensure_baseline():
            return skip("no baseline holdings")
        assert self._state.baseline is not None
        if self._state.baseline.get(ticker, 0) > 0:
            return skip("user already holds this ticker")
        if _any_entry_timing(lim):
            return self._defer(signal, now)
        await self._enter_now(signal)
        return None

    def _defer(self, signal: TradingSignal, now: dt.datetime) -> None:
        ticker = signal.ticker.value
        if ticker in self._state.deferred:
            logger.info(
                "auto_trader: entry skipped", ticker=ticker, reason="entry already deferred"
            )
            return
        try:
            quote = self._broker.get_quote(ticker)
        except Exception:
            logger.exception("auto_trader: quote failed", ticker=ticker)
            return
        ref_price = quote.last_price or quote.ask
        if not ref_price or ref_price <= 0:
            logger.info("auto_trader: entry skipped", ticker=ticker, reason="no reference price")
            return
        self._signals.put(signal)
        self._record(
            "entry_deferred",
            ticker=ticker,
            signal_id=str(signal.id),
            ref_price=str(ref_price),
            signaled_at=now.isoformat(),
        )
        logger.info("auto_trader: entry deferred", ticker=ticker, ref_price=str(ref_price))

    async def _enter_now(self, signal: TradingSignal, *, deferred: bool = False) -> None:
        """Concurrent cap, quote, notional, cash, then `execute_trade`."""
        ticker = signal.ticker.value
        lim = self._limits

        def skip(reason: str) -> None:
            logger.info("auto_trader: entry skipped", ticker=ticker, reason=reason)

        open_count = self._open_position_count()
        if open_count is None:
            return skip("positions lookup failed")
        if open_count >= lim.max_open_positions:
            return skip("concurrent position cap reached")

        try:
            quote = self._broker.get_quote(ticker)
        except Exception:
            logger.exception("auto_trader: quote failed", ticker=ticker)
            return None
        if quote.ask <= 0:
            return skip("no ask (upper limit or empty book)")
        if deferred and self._extended_at_entry(signal, quote.ask):
            return None
        if quote.ask > lim.max_order_notional:
            return skip(f"1 share at {quote.ask} exceeds notional cap")

        # Kiwoom's orderable amount includes margin (13M vs a 1.1M deposit on
        # 2026-09-13), so an order can go through on credit (미수). Require
        # deposit cash for it, net of what today's own entries already use.
        try:
            cash = self._broker.get_account().cash
        except Exception:
            logger.exception("auto_trader: account lookup failed", ticker=ticker)
            return None
        committed = self._committed()
        if cash - committed < quote.ask * Decimal("1.01"):
            return skip(f"insufficient deposit cash ({cash} - {committed} committed)")

        limit = quote.ask
        if lim.entry_limit_ticks > 0:
            limit = quote.ask - tick_size(quote.ask) * lim.entry_limit_ticks
        if not self._live:
            logger.info("auto_trader: DRY RUN would buy", ticker=ticker, limit=str(limit))
            self._record("entry_dry_run", ticker=ticker, limit=str(limit), signal_id=str(signal.id))
            return None

        from daytrade_it.interfaces.mcp.tools import MCPToolContext, execute_trade  # noqa: PLC0415

        self._signals.put(signal)
        MCPToolContext.set_instance(
            MCPToolContext(
                signal_repository=self._signals,
                broker=self._broker,
                risk_manager=RiskManager(
                    # check_total_exposure allows max_position_value * 10.
                    RiskConfig(max_position_value=self._limits.max_trader_exposure / 10)
                ),
            )
        )
        try:
            result = await execute_trade(
                signal_id=str(signal.id), order_type="LIMIT", limit_price=float(limit)
            )
        except ValueError as exc:
            # Pre-order refusals: kill switch, price band, holdings, risk gate.
            logger.warning("auto_trader: entry refused", ticker=ticker, reason=str(exc))
            self._record("entry_refused", ticker=ticker, reason=str(exc))
            return None
        except Exception as exc:
            # Could be a broker rejection or a failure after submission --
            # never retry this ticker today.
            logger.exception("auto_trader: entry outcome unknown", ticker=ticker)
            self._record("entry_unknown", ticker=ticker, error=str(exc), signal_id=str(signal.id))
            return None

        self._record(
            "entry_submitted",
            ticker=ticker,
            order_id=result.broker_order_id,
            limit=str(limit),
            quantity=result.quantity,
            signal_id=str(signal.id),
        )
        logger.info(
            "auto_trader: ENTRY SUBMITTED",
            ticker=ticker,
            order_id=result.broker_order_id,
            limit=str(limit),
        )
        return None

    def _extended_at_entry(self, signal: TradingSignal, ask: Decimal) -> bool:
        """No-chase re-applied at deferred entry: ask vs the signal's prev_close."""
        from daytrade_it.application.signals import entry_filter  # noqa: PLC0415

        prev = signal.metadata.get("prev_close")
        if not prev or Decimal(str(prev)) <= 0:
            return False
        move = ask / Decimal(str(prev)) - 1
        if move <= entry_filter.MAX_PRE_ENTRY_MOVE:
            return False
        self._record(
            "entry_expired",
            ticker=signal.ticker.value,
            signal_id=str(signal.id),
            reason="extended_at_entry",
            ask=str(ask),
            prev_close=str(prev),
        )
        return True

    def _deferred_entries(self, now: dt.datetime) -> None:
        """Enter deferred signals whose timing rules hold; expire the stale ones."""
        lim = self._limits
        for ticker, row in list(self._state.deferred.items()):
            signal = self._signals.get(row["signal_id"])

            def expire(reason: str, ticker: str = ticker, row: dict = row) -> None:
                self._record(
                    "entry_expired", ticker=ticker, signal_id=row["signal_id"], reason=reason
                )

            if signal is None:
                expire("restart")
                continue
            signaled_at = dt.datetime.fromisoformat(row["signaled_at"])
            if now - signaled_at > lim.entry_max_wait:
                expire("max_wait")
                continue
            if not (lim.entry_start <= now.time() < lim.entry_end):
                expire("outside entry window")
                continue
            if ticker in self._state.entries:
                expire("already entered today")
                continue
            if len(self._state.entries) >= lim.max_entries_per_day:
                expire("daily entry cap reached")
                continue
            if lim.entry_delay is not None and now - signaled_at < lim.entry_delay:
                continue
            # One attempt per ticker per `exit_check_every`, whatever the rules.
            last = self._last_entry_check.get(ticker)
            if last is not None and now - last < lim.exit_check_every:
                continue
            self._last_entry_check[ticker] = now
            bars: list[MinuteBar] = []
            if lim.entry_pullback is not None or lim.entry_vwap_reclaim:
                try:
                    bars = self._broker.get_minute_bars(ticker)
                except Exception:
                    logger.exception("auto_trader: minute bars failed", ticker=ticker)
                    continue
            if entry_ready(bars, Decimal(row["ref_price"]), signaled_at, now, lim):
                try:
                    _drive(self._enter_now(signal, deferred=True))
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as exc:
                    # Suspended or escaped after possibly placing the order: like
                    # the default path's unknown outcome, close the deferral and
                    # block the ticker for the day -- never re-send.
                    logger.exception("auto_trader: deferred entry outcome unknown", ticker=ticker)
                    self._record(
                        "entry_unknown",
                        ticker=ticker,
                        error=repr(exc),
                        signal_id=row["signal_id"],
                    )

    def _committed(self) -> Decimal:
        """Limit value of today's entries and add-ons not yet exited."""
        s = self._state
        rows = [*s.entries.items(), *s.add_ons.items()]
        return sum(
            (Decimal(row["limit"]) for t, row in rows if t not in s.exits and row.get("limit")),
            Decimal(0),
        )

    def _open_position_count(self) -> int | None:
        """Pending entry orders plus shares the trader holds above baseline.

        An entry that was cancelled unfilled frees its slot; None if the
        account can't be read (the caller then refuses to enter).
        """
        try:
            held = {p.ticker: int(p.quantity) for p in self._broker.get_positions()}
        except Exception:
            logger.exception("auto_trader: positions lookup failed")
            return None
        baseline = self._state.baseline or {}
        pending = {row["ticker"] for row in self._state.open_entry_orders.values()}
        holding = {t for t in self._state.entries if held.get(t, 0) - baseline.get(t, 0) > 0}
        return len(pending | holding)

    # -- management ------------------------------------------------------------

    def manage(self) -> None:
        """Cancel stale entry orders; flatten from `flatten_at`. Call every tick."""
        self._ensure_day()
        now = self._clock()
        for order_id, row in list(self._state.open_entry_orders.items()):
            submitted = dt.datetime.fromisoformat(row["ts"])
            if now - submitted >= self._limits.entry_order_ttl:
                self._cancel_entry(order_id, row["ticker"], reason="ttl")
        for order_id, row in list(self._state.open_add_on_orders.items()):
            if now - dt.datetime.fromisoformat(row["ts"]) >= self._limits.entry_order_ttl:
                self._cancel_add_on(order_id, row["ticker"], reason="ttl")
        if self._state.deferred:
            self._deferred_entries(now)
        if now.time() >= self._limits.flatten_at:
            self.flatten()
        else:
            self._early_exits(now)

    def _early_exits(self, now: dt.datetime) -> None:
        """Rule exits (`exit_rule_hit`) before `flatten_at`; re-price unfilled early sells."""
        lim = self._limits
        if not _any_exit_rule(lim) or not self._state.entries:
            return
        own = self._own_holdings()
        if not own:
            return
        for ticker, qty in own.items():
            if ticker in self._state.exits:
                self._exit_or_retry(ticker, qty, now)
                continue
            last = self._last_exit_check.get(ticker)
            if last is not None and now - last < lim.exit_check_every:
                continue
            self._last_exit_check[ticker] = now
            entry_limit = self._state.entries[ticker].get("limit")
            if not entry_limit:
                continue
            try:
                bars = self._broker.get_minute_bars(ticker)
            except NotImplementedError:
                return
            except Exception:
                logger.exception("auto_trader: minute bars failed", ticker=ticker)
                continue
            entered_at = dt.datetime.fromisoformat(self._state.entries[ticker]["ts"])
            rule = exit_rule_hit(bars, Decimal(entry_limit), entered_at, now, lim)
            if rule is None:
                if (
                    lim.add_on_min_gain is not None
                    and ticker not in self._state.add_ons
                    and now.time() < lim.entry_end
                    and add_on_hit(bars, Decimal(entry_limit), now, lim.add_on_min_gain)
                ):
                    self._add_on(ticker, bars)
                continue
            else:
                # Last completed close and session VWAP, for the ledger row.
                _, close, vwap = vwap_exit_hit(
                    bars, Decimal(entry_limit), now, Decimal(0), lim.vwap_exit_min_gain
                )
                self._record(
                    "exit_signal",
                    ticker=ticker,
                    reason=rule,
                    close=str(close),
                    vwap=str(vwap),
                )
                if rule == "vwap_break":
                    logger.info(
                        "auto_trader: VWAP EXIT", ticker=ticker, close=str(close), vwap=str(vwap)
                    )
                else:
                    logger.info(
                        "auto_trader: RULE EXIT",
                        ticker=ticker,
                        rule=rule,
                        close=str(close),
                        vwap=str(vwap),
                    )
                self._cancel_pending_add_ons(ticker, reason="exit")
                qty = self._own_holdings_for(ticker) or qty
                self._sell(ticker, qty)

    def _add_on(self, ticker: str, bars: list[MinuteBar]) -> None:
        """One more share of a held ticker (pyramiding), with entry safeguards rebuilt."""
        lim = self._limits

        def skip(reason: str) -> None:
            logger.info("auto_trader: add-on skipped", ticker=ticker, reason=reason)

        if _kill_switch_on():
            return skip("kill switch")
        try:
            quote = self._broker.get_quote(ticker)
            cash = self._broker.get_account().cash
        except Exception:
            logger.exception("auto_trader: add-on quote/account failed", ticker=ticker)
            return None
        last_close = bars[-1].close
        if quote.ask <= 0 or last_close <= 0:
            return skip("no ask")
        if abs(quote.ask / last_close - 1) > lim.add_on_price_band:
            return skip(f"ask {quote.ask} outside band vs last close {last_close}")
        committed = self._committed()
        if committed + quote.ask > lim.max_trader_exposure:
            return skip(f"exposure cap ({committed} + {quote.ask} > {lim.max_trader_exposure})")
        if cash - committed < quote.ask * Decimal("1.01"):
            return skip(f"insufficient deposit cash ({cash} - {committed} committed)")
        try:
            blocked = _disclosure_blocks(ticker)
        except Exception:
            logger.exception("auto_trader: add-on disclosure check failed", ticker=ticker)
            blocked = True
        if blocked:
            return skip("DART hard disclosure or data unavailable")
        if not self._live:
            self._record("add_on_dry_run", ticker=ticker, limit=str(quote.ask))
            return None
        try:
            result = self._broker.place_order(
                Order(
                    ticker=ticker,
                    quantity=Decimal(1),
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    limit_price=quote.ask,
                )
            )
        except Exception as exc:
            logger.exception("auto_trader: ADD-ON ORDER FAILED", ticker=ticker)
            self._record("add_on_failed", ticker=ticker, error=str(exc))
            return None
        self._record(
            "add_on_submitted", ticker=ticker, order_id=result.order_id, limit=str(quote.ask)
        )
        logger.info("auto_trader: ADD-ON SUBMITTED", ticker=ticker, limit=str(quote.ask))
        return None

    def _cancel_add_on(self, order_id: str, ticker: str, *, reason: str) -> None:
        try:
            status = self._broker.get_order_status(order_id)
        except Exception:
            self._record("add_on_resolved", order_id=order_id, ticker=ticker, reason=reason)
            return
        if status.remaining_quantity <= 0:
            self._record("add_on_resolved", order_id=order_id, ticker=ticker, reason=reason)
            return
        try:
            cancelled = self._broker.cancel_order(order_id)
        except Exception:
            logger.exception("auto_trader: add-on cancel failed", order_id=order_id, ticker=ticker)
            return
        if cancelled:
            self._record("add_on_cancelled", order_id=order_id, ticker=ticker, reason=reason)

    def _cancel_pending_add_ons(self, ticker: str | None, *, reason: str) -> None:
        for order_id, row in list(self._state.open_add_on_orders.items()):
            if ticker is None or row["ticker"] == ticker:
                self._cancel_add_on(order_id, row["ticker"], reason=reason)

    def _own_holdings_for(self, ticker: str) -> int | None:
        own = self._own_holdings()
        return None if own is None else own.get(ticker)

    def _cancel_entry(self, order_id: str, ticker: str, *, reason: str) -> None:
        try:
            status = self._broker.get_order_status(order_id)
        except Exception:
            # Not in the unfilled list: filled or already gone. Positions
            # are the truth for whether shares exist (N-3).
            self._record("entry_resolved", order_id=order_id, ticker=ticker, reason=reason)
            return
        if status.remaining_quantity <= 0:
            self._record("entry_resolved", order_id=order_id, ticker=ticker, reason=reason)
            return
        try:
            cancelled = self._broker.cancel_order(order_id)
        except Exception:
            logger.exception("auto_trader: cancel failed", order_id=order_id, ticker=ticker)
            return
        if cancelled:
            self._record(
                "entry_cancelled",
                order_id=order_id,
                ticker=ticker,
                reason=reason,
                filled=int(status.filled_quantity),
            )
        else:
            logger.error("auto_trader: cancel refused by broker", order_id=order_id, ticker=ticker)

    def _own_holdings(self) -> dict[str, int] | None:
        """Shares the trader bought today and still holds; None if unreadable."""
        try:
            held = {p.ticker: int(p.quantity) for p in self._broker.get_positions()}
        except Exception:
            logger.exception("auto_trader: positions lookup failed")
            return None
        baseline = self._state.baseline or {}
        own = {}
        for ticker in self._state.entries:
            if ticker in self._state.never_filled:
                continue
            cap = 2 if ticker in self._state.add_ons else 1
            qty = min(held.get(ticker, 0) - baseline.get(ticker, 0), cap)
            if qty > 0:
                own[ticker] = qty
        return own

    def report_leftovers(self) -> dict[str, int] | None:
        """Call at shutdown: record and loudly log any share still held."""
        self._ensure_day()
        if not self._state.entries:
            return {}
        own = self._own_holdings()
        if own:
            logger.error("auto_trader: POSITIONS LEFT OPEN AT SHUTDOWN", positions=own)
            self._record("overnight_positions", positions=own)
        return own

    def flatten(self) -> None:
        """Sell what the trader bought today (never the user's baseline shares)."""
        self._ensure_day()
        if not self._state.entries:
            return
        for order_id, row in list(self._state.open_entry_orders.items()):
            self._cancel_entry(order_id, row["ticker"], reason="flatten")
        self._cancel_pending_add_ons(None, reason="flatten")

        own_holdings = self._own_holdings()
        if own_holdings is None:
            return

        now = self._clock()
        for ticker, own in own_holdings.items():
            self._exit_or_retry(ticker, own, now)

    def _exit_or_retry(self, ticker: str, quantity: int, now: dt.datetime) -> None:
        """Sell, or re-price a sell still unfilled after `exit_retry_after`."""
        prior = self._state.exits.get(ticker)
        if prior is not None:
            if now - dt.datetime.fromisoformat(prior["ts"]) < self._limits.exit_retry_after:
                return
            if prior.get("order_id"):
                try:
                    self._broker.cancel_order(prior["order_id"])
                except Exception:
                    logger.exception("auto_trader: stale exit cancel failed", ticker=ticker)
        self._sell(ticker, quantity)

    def _sell(self, ticker: str, quantity: int) -> None:
        try:
            quote = self._broker.get_quote(ticker)
        except Exception:
            logger.exception("auto_trader: exit quote failed", ticker=ticker)
            return
        if quote.bid <= 0:
            logger.error("auto_trader: NO BID -- cannot exit (lower limit?)", ticker=ticker)
            return
        attempt = self._state.exit_attempts.get(ticker, 0)
        discount = min(self._limits.exit_discount_step * attempt, self._limits.exit_discount_cap)
        limit = quote.bid * (1 - discount)
        if not self._live:
            self._record(
                "exit_dry_run", ticker=ticker, quantity=quantity, limit=str(limit), attempt=attempt
            )
            return
        try:
            result = self._broker.place_order(
                Order(
                    ticker=ticker,
                    quantity=Decimal(quantity),
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    limit_price=limit,
                )
            )
        except Exception as exc:
            logger.exception("auto_trader: EXIT ORDER FAILED", ticker=ticker)
            self._record("exit_failed", ticker=ticker, error=str(exc), attempt=attempt)
            return
        self._record(
            "exit_submitted",
            ticker=ticker,
            order_id=result.order_id,
            quantity=quantity,
            limit=str(limit),
            attempt=attempt,
        )
        logger.info("auto_trader: EXIT SUBMITTED", ticker=ticker, limit=str(limit), attempt=attempt)
