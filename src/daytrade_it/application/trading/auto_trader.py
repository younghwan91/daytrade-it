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
- Fills (2026-10): open entry / add-on / exit orders are polled once per
  `exit_check_every`; when the broker reports the order done with an average
  price, a `fill` row (ticker, order_id, side, quantity, avg_price, role) is
  written so the forward gate can score actual fills instead of rebuilding
  them from minute bars. The poll never decides anything: a status error
  leaves entries to the TTL path above, and only exits are marked resolved
  when their order has vanished (Kiwoom's unfilled list drops filled and
  cancelled orders alike, without a price).

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
from daytrade_it.domain.services.risk_manager import RiskConfig, RiskManager
from daytrade_it.infrastructure.brokers.base import (
    BrokerInterface,
    MinuteBar,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    now_kst,
)
from daytrade_it.shared.killswitch import stop_file_active
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
    #: 본절 손절: 한 번 이 이상 올랐던 포지션은 손실로 끝내지 않는다. 되돌아와
    #: 본전선(진입가 + `breakeven_buffer`)에 닿으면 판다. 사용자 지시
    #: 2026-09-16 -- 09-16 실매매에서 다섯 종목 모두 +1.51%~+14.19% 까지 갔다가
    #: 전부 마이너스로 끝났다. None 이면 끔.
    #: 트레일링(운영 풀 7일에서 +39,915 -> -547)과 달리 진입가 아래로는 자르지
    #: 않고 이익이 난 뒤에만 켜지므로, 되돌림에 털리는 폭이 훨씬 좁다.
    breakeven_after_gain: Decimal | None = None
    #: 왕복 수수료·거래세(매도 ~0.18% + 수수료)를 덮는 만큼 본전선을 올린다.
    breakeven_buffer: Decimal = Decimal("0.003")
    #: 주춤 익절: 이익 구간에서 `stall_minutes` 분 동안 고점을 새로 못 쓰면 판다.
    #: 사용자 지시 2026-09-16 -- "저항에 부딪치거나 주춤거리면 그냥 익절해,
    #: 오르면 피라미딩이고." 오르는 동안에는 고점이 갱신되어 걸리지 않는다.
    #: None 이면 끔.
    stall_exit_min_gain: Decimal | None = None
    stall_exit_minutes: int = 5
    #: 불타기 끔(2026-09-23 사용자 지시). 두 책에서 모두 보유보다 나빴다:
    #: -63.0 vs -35.1 bp, -81.9 vs -50.6 bp (feat/rl-exit-seq).
    add_on_min_gain: Decimal | None = None
    add_on_price_band: Decimal = Decimal("0.03")
    #: 1,000,000 KRW, user-approved 2026-09-15 for pyramiding. Caps add-ons
    #: (limit value of entries + add-ons not yet exited) and sets the
    #: RiskManager account-exposure check that gates entries (was ~500k).
    #: The deposit-cash check still applies (cash is shared with scalp-it).
    max_trader_exposure: Decimal = Decimal("1000000")


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
    #: Exit orders still polled for a fill price; popped on `exit_resolved`.
    open_exit_orders: dict[str, dict[str, Any]] = field(default_factory=dict)


def rank_batch(items: list[tuple[TradingSignal, int, float]]) -> list[TradingSignal]:
    """Entry order for one poll's signals: v2 `surprise` desc, then news score desc.

    At the open one poll yields more BUYs than there are slots; ranking by
    surprise beat random picks out-of-fold (+1.92 vs +1.37%p per trade).
    """
    return [s for s, _, _ in sorted(items, key=lambda x: (-x[1], -x[2]))]


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


def _entry_time(entry_row: dict[str, Any]) -> dt.datetime | None:
    """진입 기록의 체결/제출 시각, 못 읽으면 None (그 경우 규칙을 건너뛴다).

    시각을 모를 때 0 시로 떨어지면 하루 전체가 '진입 이후'가 되어 규칙이
    오전 고점을 물고 오작동한다 -- 모르면 쏘지 않는 쪽이 안전하다.
    """
    raw = entry_row.get("ts")
    if not raw:
        return None
    try:
        return dt.datetime.fromisoformat(str(raw))
    except ValueError:
        return None


def _after_entry(
    bars: list[MinuteBar], now: dt.datetime, entered_at: dt.datetime
) -> tuple[list[MinuteBar], list[MinuteBar]]:
    """(완료된 정규장 봉 전부, 그중 진입 이후 봉).

    진입 분의 봉은 체결 **전** 고가를 담을 수 있으므로 진입 다음 분부터 센다
    (`feat/a-plan` 의 `exit_rule_hit` 트레일링 규칙과 같은 처리). 이 구분이
    없으면 오후에 산 포지션이 오전 고점을 자기 고점으로 착각한다.
    """
    minute = now.replace(tzinfo=None, second=0, microsecond=0)
    done = [b for b in bars if b.ts.time() >= dt.time(9, 0) and b.ts < minute]
    after = entered_at.replace(tzinfo=None, second=0, microsecond=0) + dt.timedelta(minutes=1)
    return done, [b for b in done if b.ts >= after]


def breakeven_stop_hit(
    bars: list[MinuteBar],
    entry: Decimal,
    now: dt.datetime,
    arm_gain: Decimal,
    buffer: Decimal,
    entered_at: dt.datetime,
) -> tuple[bool, Decimal | None, Decimal | None]:
    """(hit, peak gain so far, last completed close) for the break-even stop.

    사용자 원칙(2026-09-16): "익절이 될 걸 손실로 바꾸진 않아 절대." 한 번
    `arm_gain` 이상 올랐던 포지션은 손실로 끝내지 않는다 -- 되돌아와 본전선에
    닿으면 판다. `buffer` 는 왕복 수수료·세금만큼 본전선을 위로 올려, 본전에
    팔았는데 실제로는 마이너스가 되는 것을 막는다.

    트레일링과 다른 점: 진입가 아래로는 절대 자르지 않고, 이익이 난 뒤에만
    켜진다. 완료된 1분봉만 본다(vwap_exit_hit 과 같은 규칙).
    """
    done, since = _after_entry(bars, now, entered_at)
    if not done or not since or entry <= 0:
        return False, None, None
    peak_gain = max(b.high for b in since) / entry - 1
    close = done[-1].close
    if peak_gain < arm_gain:
        return False, peak_gain, close
    return close <= entry * (1 + buffer), peak_gain, close


def stall_exit_hit(
    bars: list[MinuteBar],
    entry: Decimal,
    now: dt.datetime,
    min_gain: Decimal,
    stall_minutes: int,
    entered_at: dt.datetime,
) -> tuple[bool, Decimal | None]:
    """(hit, gain at the last completed close) for the stall / resistance exit.

    사용자 원칙(2026-09-16): "저항에 부딪치거나 주춤거리면 그냥 익절해, 오르면
    피라미딩이고." 이익 구간에서 고점을 `stall_minutes` 분 동안 새로 못 쓰면
    움직임이 멈춘 것으로 보고 익절한다. 오르는 동안에는 고점이 계속 갱신되므로
    걸리지 않는다 -- 그때는 `add_on_hit` 이 피라미딩을 맡는다.

    손실 구간에서는 켜지지 않는다(그쪽은 `breakeven_stop_hit` 과 손절의 몫).
    완료된 1분봉만 본다.
    """
    done, since = _after_entry(bars, now, entered_at)
    # 고점 비교는 진입 이후 봉만으로 한다. 오전 고점을 기준으로 삼으면 오후에
    # 산 포지션은 그 고점을 넘을 일이 없어 "이익이 나자마자 매도"로 퇴화한다.
    if len(since) <= stall_minutes or not done or entry <= 0 or stall_minutes <= 0:
        return False, None
    close = done[-1].close
    gain = close / entry - 1
    if gain < min_gain:
        return False, gain
    recent = since[-stall_minutes:]
    earlier_peak = max(b.high for b in since[:-stall_minutes])
    # 최근 stall_minutes 분 안에 이전 고점을 넘지 못했다 = 주춤.
    return max(b.high for b in recent) <= earlier_peak, gain


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

    return stop_file_active(tools._KILL_SWITCH_PATH)


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
        #: In-memory only: after a restart the first check simply runs at once.
        self._last_exit_check: dict[str, dt.datetime] = {}
        self._last_fill_check: dict[str, dt.datetime] = {}

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
            if row.get("order_id"):
                s.open_exit_orders[row["order_id"]] = row
        elif ev == "exit_resolved":
            s.open_exit_orders.pop(row["order_id"], None)
        elif ev in ("add_on_submitted", "add_on_failed", "add_on_dry_run"):
            s.add_ons[row["ticker"]] = row
            if row.get("order_id"):
                s.open_add_on_orders[row["order_id"]] = row
        elif ev in ("add_on_cancelled", "add_on_resolved"):
            s.open_add_on_orders.pop(row["order_id"], None)

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
        committed = self._committed()
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
        self._poll_fills(now)
        for order_id, row in list(self._state.open_entry_orders.items()):
            submitted = dt.datetime.fromisoformat(row["ts"])
            if now - submitted >= self._limits.entry_order_ttl:
                self._cancel_entry(order_id, row["ticker"], reason="ttl")
        for order_id, row in list(self._state.open_add_on_orders.items()):
            if now - dt.datetime.fromisoformat(row["ts"]) >= self._limits.entry_order_ttl:
                self._cancel_add_on(order_id, row["ticker"], reason="ttl")
        if now.time() >= self._limits.flatten_at:
            self.flatten()
        else:
            self._early_exits(now)

    # -- fills -----------------------------------------------------------------

    def _poll_fills(self, now: dt.datetime) -> None:
        """Record `fill` for open orders the broker reports done with a price.

        Observation only -- no order is sent or cancelled here. One status call
        per open order per `exit_check_every` (the same budget as the chart
        checks; Kiwoom's status is one ka10075 call).
        """
        s = self._state
        todo = [
            *((oid, row, "entry") for oid, row in s.open_entry_orders.items()),
            *((oid, row, "add_on") for oid, row in s.open_add_on_orders.items()),
            *((oid, row, "exit") for oid, row in s.open_exit_orders.items()),
        ]
        for order_id, row, role in todo:
            last = self._last_fill_check.get(order_id)
            if last is not None and now - last < self._limits.exit_check_every:
                continue
            self._last_fill_check[order_id] = now
            ticker = row["ticker"]
            try:
                status = self._broker.get_order_status(order_id)
            except Exception:
                # Entries: the TTL path (`_cancel_entry`) owns "not found" --
                # a transient API error must not stop a stale order's cancel.
                if role == "exit":
                    self._record("exit_resolved", order_id=order_id, ticker=ticker, reason="gone")
                continue
            if self._record_fill_if_done(order_id, ticker, role, status):
                reason = "filled" if status.filled_quantity > 0 else "gone"
                self._record(f"{role}_resolved", order_id=order_id, ticker=ticker, reason=reason)

    def _record_fill_if_done(
        self, order_id: str, ticker: str, role: str, status: OrderStatus
    ) -> bool:
        """True when nothing is left to fill; writes `fill` if the broker gave a price."""
        if status.remaining_quantity > 0:
            return False
        if status.filled_quantity > 0 and status.filled_avg_price is not None:
            self._record(
                "fill",
                ticker=ticker,
                order_id=order_id,
                side="SELL" if role == "exit" else "BUY",
                quantity=int(status.filled_quantity),
                avg_price=str(status.filled_avg_price),
                role=role,
            )
        return True

    def _early_exits(self, now: dt.datetime) -> None:
        """VWAP-break exits before `flatten_at`; re-price unfilled early sells."""
        lim = self._limits
        rules_on = (
            lim.vwap_exit_gap is not None
            or lim.breakeven_after_gain is not None
            or lim.stall_exit_min_gain is not None
        )
        if not rules_on or not self._state.entries:
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
            entry_row = self._state.entries[ticker]
            entry_limit = entry_row.get("limit")
            entered_at = _entry_time(entry_row)
            if not entry_limit:
                continue
            try:
                bars = self._broker.get_minute_bars(ticker)
            except NotImplementedError:
                return
            except Exception:
                logger.exception("auto_trader: minute bars failed", ticker=ticker)
                continue
            entry_px = Decimal(entry_limit)
            hit, close, vwap = (False, None, None)
            reason = ""
            if lim.vwap_exit_gap is not None:
                hit, close, vwap = vwap_exit_hit(
                    bars, entry_px, now, lim.vwap_exit_gap, lim.vwap_exit_min_gain
                )
                reason = "vwap_break" if hit else ""
            # 주춤 익절이 본절보다 먼저다: 아직 이익일 때 나가는 쪽이 낫다.
            # 둘 다 진입 시각을 받는다 -- 진입 전 고점을 자기 고점으로 세면
            # 오전에 오른 종목을 오후에 산 순간 곧바로 매도가 나간다.
            if not hit and lim.stall_exit_min_gain is not None and entered_at is not None:
                hit, gain = stall_exit_hit(
                    bars,
                    entry_px,
                    now,
                    lim.stall_exit_min_gain,
                    lim.stall_exit_minutes,
                    entered_at,
                )
                if hit:
                    reason, close = "stall", gain
            if not hit and lim.breakeven_after_gain is not None and entered_at is not None:
                hit, peak_gain, close = breakeven_stop_hit(
                    bars,
                    entry_px,
                    now,
                    lim.breakeven_after_gain,
                    lim.breakeven_buffer,
                    entered_at,
                )
                if hit:
                    reason, vwap = "breakeven", peak_gain
            if not hit:
                if (
                    lim.add_on_min_gain is not None
                    and ticker not in self._state.add_ons
                    and now.time() < lim.entry_end
                    and add_on_hit(bars, Decimal(entry_limit), now, lim.add_on_min_gain)
                ):
                    self._add_on(ticker, bars)
                continue
            if hit:
                self._record(
                    "exit_signal",
                    ticker=ticker,
                    reason=reason,
                    close=str(close),
                    vwap=str(vwap),
                )
                logger.info(
                    "auto_trader: EARLY EXIT",
                    ticker=ticker,
                    reason=reason,
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
        if self._record_fill_if_done(order_id, ticker, "add_on", status):
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
        if self._record_fill_if_done(order_id, ticker, "entry", status):
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
