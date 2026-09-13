"""Automated execution for news signals: enter 1 share, flat by the close.

Mirrors the rule that was evaluated (docs/research/2026-09-13-news-prompt-
eval.md): enter after the news, exit at the session close. No stop-loss or
take-profit -- none was evaluated, and with a 1-share cap the loss per
position is bounded by the share price.

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

from daytrade_it.domain.entities.signal import SignalDirection, TradingSignal
from daytrade_it.domain.services.risk_manager import RiskManager
from daytrade_it.infrastructure.brokers.base import (
    BrokerInterface,
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
    #: 2, not more: scalp-it trades the same Kiwoom account and cash (~1.1M
    #: KRW deposit on 2026-09-13), and RiskManager's default exposure check
    #: rejects past ~500k KRW of held value anyway.
    max_open_positions: int = 2
    max_entries_per_day: int = 10
    entry_start: dt.time = dt.time(9, 0, 30)
    #: Leave time for the move the eval measured (entry -> close).
    entry_end: dt.time = dt.time(14, 30)
    entry_order_ttl: dt.timedelta = dt.timedelta(minutes=5)
    flatten_at: dt.time = dt.time(15, 10)
    exit_retry_after: dt.timedelta = dt.timedelta(minutes=3)


@dataclass
class _DayState:
    baseline: dict[str, int] | None = None
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    open_entry_orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    exits: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Entry cancelled with zero fill: any later holding of that ticker is
    #: the user's (e.g. bought by hand), not the trader's.
    never_filled: set[str] = field(default_factory=set)


class _SignalLookup:
    """Minimal `signal_repository` for `execute_trade`'s `get_by_id`."""

    def __init__(self) -> None:
        self._signals: dict[UUID, TradingSignal] = {}

    def put(self, signal: TradingSignal) -> None:
        self._signals[signal.id] = signal

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
        committed = sum(
            Decimal(row["limit"])
            for t, row in self._state.entries.items()
            if t not in self._state.exits and row.get("limit")
        )
        if cash - committed < quote.ask * Decimal("1.01"):
            return skip(f"insufficient deposit cash ({cash} - {committed} committed)")

        if not self._live:
            logger.info("auto_trader: DRY RUN would buy", ticker=ticker, limit=str(quote.ask))
            self._record(
                "entry_dry_run", ticker=ticker, limit=str(quote.ask), signal_id=str(signal.id)
            )
            return None

        from daytrade_it.interfaces.mcp.tools import MCPToolContext, execute_trade  # noqa: PLC0415

        self._signals.put(signal)
        MCPToolContext.set_instance(
            MCPToolContext(
                signal_repository=self._signals, broker=self._broker, risk_manager=RiskManager()
            )
        )
        try:
            result = await execute_trade(
                signal_id=str(signal.id), order_type="LIMIT", limit_price=float(quote.ask)
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
            limit=str(quote.ask),
            quantity=result.quantity,
            signal_id=str(signal.id),
        )
        logger.info(
            "auto_trader: ENTRY SUBMITTED",
            ticker=ticker,
            order_id=result.broker_order_id,
            limit=str(quote.ask),
        )
        return None

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
        if now.time() >= self._limits.flatten_at:
            self.flatten()

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

    def flatten(self) -> None:
        """Sell what the trader bought today (never the user's baseline shares)."""
        self._ensure_day()
        if not self._state.entries:
            return
        for order_id, row in list(self._state.open_entry_orders.items()):
            self._cancel_entry(order_id, row["ticker"], reason="flatten")

        try:
            held = {p.ticker: int(p.quantity) for p in self._broker.get_positions()}
        except Exception:
            logger.exception("auto_trader: positions lookup failed during flatten")
            return

        baseline = self._state.baseline or {}
        now = self._clock()
        for ticker in self._state.entries:
            if ticker in self._state.never_filled:
                continue
            own = min(held.get(ticker, 0) - baseline.get(ticker, 0), 1)
            if own <= 0:
                continue
            prior = self._state.exits.get(ticker)
            if prior is not None:
                if now - dt.datetime.fromisoformat(prior["ts"]) < self._limits.exit_retry_after:
                    continue
                if prior.get("order_id"):
                    try:
                        self._broker.cancel_order(prior["order_id"])
                    except Exception:
                        logger.exception("auto_trader: stale exit cancel failed", ticker=ticker)
            self._sell(ticker, own)

    def _sell(self, ticker: str, quantity: int) -> None:
        try:
            quote = self._broker.get_quote(ticker)
        except Exception:
            logger.exception("auto_trader: exit quote failed", ticker=ticker)
            return
        if quote.bid <= 0:
            logger.error("auto_trader: NO BID -- cannot exit (lower limit?)", ticker=ticker)
            return
        if not self._live:
            self._record("exit_dry_run", ticker=ticker, quantity=quantity, limit=str(quote.bid))
            return
        try:
            result = self._broker.place_order(
                Order(
                    ticker=ticker,
                    quantity=Decimal(quantity),
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    limit_price=quote.bid,
                )
            )
        except Exception as exc:
            logger.exception("auto_trader: EXIT ORDER FAILED", ticker=ticker)
            self._record("exit_failed", ticker=ticker, error=str(exc))
            return
        self._record(
            "exit_submitted",
            ticker=ticker,
            order_id=result.order_id,
            quantity=quantity,
            limit=str(quote.bid),
        )
        logger.info("auto_trader: EXIT SUBMITTED", ticker=ticker, limit=str(quote.bid))
