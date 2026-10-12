"""No-chase entry filter for news BUY signals.

Replaces the earlier "technical confirmation" rule (BUY only once price was
already up >= 0.3% vs. the prior close). Measured against real outcomes
(`docs/research/2026-09-13-news-prompt-eval.md`) that rule selected the
worse half of news BUYs:

    news BUY & already up > +0.3%   dev lift +0.15%p   (≈ no edge)
    news BUY & up <= +2%            test mean +2.16%, hit 82%  (held-out)

i.e. by the time a news stock has visibly moved, the same-session move is
mostly spent -- the same "추격 금지" result scalp-it reached independently.

This filter emits a `SignalSource.TECHNICAL` BUY when the stock is NOT
extended, which is what `SignalGenerator.combine_signals()` needs alongside
a NEWS signal. It never emits SELL: news SELLs are not entries here (KRX
retail can't short), and `execute_trade` refuses a SELL without a holding.

Prior close (R17) = the KRX **regular-session** close: the prior session's last
minute bar 09:00 <= ts < 15:33 (`domain.services.krx_session`). From 2026-09-14
the daily close in `daily_bars_adjusted` includes KRX after-market trades
(16:00-20:00), and from then until 2026-10-12 this filter measured the move
against that after-market close; the 3% cap was chosen on pre-09-14 days, when
the daily close was the regular close. Back to the regular close from the
2026-10-13 session (user decision; forward record restarts then,
`docs/research/frozen/2026-10-13-no-chase-regular-close-amend1.json`).

A ticker with no regular minute bar for the prior session (scalp-it collects
minute bars for its top-1400 universe only; ~3% of filtered ticker-days in
09-15..10-12) falls back to the daily close, logged and tagged
`prev_close_source=daily_fallback`. That is the pre-10-13 behaviour for those
names; refusing them instead would silently shrink the universe to the minute-
bar universe, and the after-market gap is a few % on most names.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal

from daytrade_it.domain.entities.signal import SignalDirection, SignalSource, TradingSignal
from daytrade_it.domain.value_objects import Confidence, Ticker
from daytrade_it.infrastructure.brokers.base import BrokerInterface
from daytrade_it.infrastructure.external.quant_airflow_db_client import (
    AdjustedDailyBar,
    QuantAirflowDBClient,
    RegularSessionClose,
)
from daytrade_it.shared.logging import get_logger

logger = get_logger(__name__)

#: Refuse entry once the price is up more than this vs. the prior close.
#: 2% was pre-declared on the 09-13 prompt eval. Raised to 3% on 2026-09-15:
#: with the live policy (surprise-ranked entries + VWAP-break exit, live
#: limits, leave-one-day-out over 6 days) the 6-day total was +72.0%p at 2%,
#: +92.4%p at 3%, +89.5%p at 5%, +81.8%p with no cap; the per-feature path
#: analysis also showed +2..+5% names still gaining and >+5% names losing.
#: scripts/eval/combined_policy.py (PRE_MOVE_CAP) on feat/exit-rule-eval.
MAX_PRE_ENTRY_MOVE = Decimal("0.03")

#: Fixed: this is a gate, not a predictor. It shouldn't outvote the news
#: signal's confidence in `combine_signals()`'s weighted average.
_FILTER_CONFIDENCE = 0.5

_LOOKBACK_DAYS = 10

#: First live session whose prior close is the regular-session close (deployed by the
#: 07:50 pull that day). Before it, live measured against the daily close (after-market
#: included from 2026-09-14); the replay serves each day the basis live used then.
REGULAR_CLOSE_FROM = dt.date(2026, 10, 13)

#: `PrevClose.source` values (signal metadata / logs).
KRX_REGULAR = "krx_regular"
DAILY_FALLBACK = "daily_fallback"


@dataclass(frozen=True)
class PrevClose:
    price: Decimal
    date: dt.date
    source: str


def choose_prev_close(
    daily: AdjustedDailyBar | None, regular: RegularSessionClose | None
) -> PrevClose | None:
    """The regular-session close, unless it is missing or older than the latest daily bar.

    An older regular bar means the prior session's minute bars are missing; its close
    would be a stale session's, so the daily close is used instead.
    """
    daily = daily if daily is not None and daily.close else None
    if regular is not None and regular.close and (daily is None or regular.date >= daily.date):
        return PrevClose(Decimal(str(regular.close)), regular.date, KRX_REGULAR)
    if daily is not None:
        return PrevClose(Decimal(str(daily.close)), daily.date, DAILY_FALLBACK)
    return None


def generate_no_chase_confirmation(
    ticker: Ticker,
    broker: BrokerInterface,
    db_client: QuantAirflowDBClient,
    expires_at: dt.datetime,
) -> TradingSignal | None:
    """TECHNICAL BUY if the price is not already up > MAX_PRE_ENTRY_MOVE, else None."""
    quote = broker.get_quote(ticker.value)
    current_price = quote.last_price or quote.mid_price
    if not current_price or current_price <= 0:
        logger.warning("entry_filter: no usable current price", ticker=ticker.value)
        return None

    today = dt.date.today()
    bars = db_client.get_adjusted_daily_bars(
        ticker, start=today - dt.timedelta(days=_LOOKBACK_DAYS), end=today - dt.timedelta(days=1)
    )
    regular = db_client.get_regular_session_close(
        ticker, before=today, lookback_days=_LOOKBACK_DAYS
    )
    prev = choose_prev_close(bars[-1] if bars else None, regular)
    if prev is None:
        logger.warning("entry_filter: no prior daily close", ticker=ticker.value)
        return None
    if prev.source == DAILY_FALLBACK:
        logger.warning(
            "entry_filter: no regular-session close, daily close fallback",
            ticker=ticker.value,
            prev_close=str(prev.price),
            prev_close_date=str(prev.date),
            regular_close_date=None if regular is None else str(regular.date),
        )

    prev_close = prev.price
    move = (current_price - prev_close) / prev_close
    if move > MAX_PRE_ENTRY_MOVE:
        logger.info(
            "entry_filter: extended, no entry",
            ticker=ticker.value,
            pre_entry_move=str(move),
            prev_close=str(prev_close),
            prev_close_source=prev.source,
        )
        return None

    return TradingSignal(
        ticker=ticker,
        direction=SignalDirection.BUY,
        confidence=Confidence(value=_FILTER_CONFIDENCE),
        source=SignalSource.TECHNICAL,
        expires_at=expires_at,
        metadata={
            "source_type": "no_chase_entry_filter",
            "prev_close": str(prev_close),
            "prev_close_date": str(prev.date),
            "prev_close_source": prev.source,
            "current_price": str(current_price),
            "pre_entry_move": str(move),
            "max_pre_entry_move": str(MAX_PRE_ENTRY_MOVE),
        },
    )
