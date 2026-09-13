"""Simple, rule-based technical-confirmation signal.

This is deliberately NOT the ML classifier evaluated 2026-09-13 (a
price-technical XGBoost model trained on ~48k daily-bar samples scored
38.5% accuracy -- essentially the majority-class baseline, no real edge;
see that session's summary). A trained model with no demonstrated edge
must not be wired into live orders.

What this module *is*: the same kind of plain, auditable rule scalp-it
uses for its own corroboration checks (a same-direction price move, not a
predictive model) -- just enough for `SignalGenerator.combine_signals()`'s
"news alone is not actionable, needs an ML/TECHNICAL/FUNDAMENTAL signal
too" gate (Task 11 / deep-research finding #1) to be satisfied honestly,
by an actual (if simple) technical check rather than a rubber stamp or a
disabled gate.
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

#: Minimum |change| vs. the last recorded daily close before this module
#: will confirm a direction at all. Below this, "confirmation" would just
#: be noise -- no signal is produced (not a HOLD, no signal at all).
_CONFIRMATION_THRESHOLD = Decimal("0.003")

#: |change| at or above this is treated as maximum confidence (0.9, never
#: 1.0 -- this is a simple rule, not a validated model, and should never
#: present itself as fully certain).
_MAX_CONFIDENCE_CHANGE = Decimal("0.02")
_MAX_CONFIDENCE = 0.9

_LOOKBACK_DAYS = 10
_SIGNAL_VALIDITY_HOURS = 4


def generate_technical_confirmation(
    ticker: Ticker,
    broker: BrokerInterface,
    db_client: QuantAirflowDBClient,
) -> TradingSignal | None:
    """Compare the ticker's current quote to its last recorded daily close.

    Returns a `SignalSource.TECHNICAL` signal in the direction of that
    move if it clears `_CONFIRMATION_THRESHOLD`, else None (no signal --
    this function never fabricates a HOLD/neutral signal).
    """
    quote = broker.get_quote(ticker.value)
    current_price = quote.last_price or quote.mid_price
    if not current_price or current_price <= 0:
        logger.warning("technical_confirmation: no usable current price", ticker=ticker.value)
        return None

    end = dt.date.today()
    start = end - dt.timedelta(days=_LOOKBACK_DAYS)
    bars = db_client.get_adjusted_daily_bars(ticker, start=start, end=end)
    if not bars:
        logger.warning("technical_confirmation: no recent daily bars", ticker=ticker.value)
        return None

    prev_close = Decimal(str(bars[-1].close))
    if prev_close <= 0:
        return None

    change = (current_price - prev_close) / prev_close

    if abs(change) < _CONFIRMATION_THRESHOLD:
        return None

    direction = SignalDirection.BUY if change > 0 else SignalDirection.SELL
    confidence_value = min(
        _MAX_CONFIDENCE, float(abs(change) / _MAX_CONFIDENCE_CHANGE) * _MAX_CONFIDENCE
    )

    return TradingSignal(
        ticker=ticker,
        direction=direction,
        confidence=Confidence(value=confidence_value),
        source=SignalSource.TECHNICAL,
        expires_at=dt.datetime.now(dt.UTC) + dt.timedelta(hours=_SIGNAL_VALIDITY_HOURS),
        metadata={
            "source_type": "technical_confirmation",
            "prev_close": str(prev_close),
            "current_price": str(current_price),
            "change_pct": str(change),
            "bar_date": str(bars[-1].date),
        },
    )
