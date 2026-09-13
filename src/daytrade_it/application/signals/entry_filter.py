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
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from daytrade_it.domain.entities.signal import SignalDirection, SignalSource, TradingSignal
from daytrade_it.domain.value_objects import Confidence, Ticker
from daytrade_it.infrastructure.brokers.base import BrokerInterface
from daytrade_it.infrastructure.external.quant_airflow_db_client import QuantAirflowDBClient
from daytrade_it.shared.logging import get_logger

logger = get_logger(__name__)

#: Refuse entry once the price is up more than this vs. the prior close.
#: Chosen on the dev split before looking at test (0.3% looked better on dev
#: but rested on n=10; 2% was pre-declared as the primary to limit overfit).
MAX_PRE_ENTRY_MOVE = Decimal("0.02")

#: Fixed: this is a gate, not a predictor. It shouldn't outvote the news
#: signal's confidence in `combine_signals()`'s weighted average.
_FILTER_CONFIDENCE = 0.5

_LOOKBACK_DAYS = 10


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
    if not bars or not bars[-1].close:
        logger.warning("entry_filter: no prior daily close", ticker=ticker.value)
        return None

    prev_close = Decimal(str(bars[-1].close))
    move = (current_price - prev_close) / prev_close
    if move > MAX_PRE_ENTRY_MOVE:
        logger.info(
            "entry_filter: extended, no entry", ticker=ticker.value, pre_entry_move=str(move)
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
            "prev_close_date": str(bars[-1].date),
            "current_price": str(current_price),
            "pre_entry_move": str(move),
            "max_pre_entry_move": str(MAX_PRE_ENTRY_MOVE),
        },
    )
