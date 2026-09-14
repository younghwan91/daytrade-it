"""No-chase entry filter: BUY confirmation only when the stock isn't extended."""

import datetime as dt
from decimal import Decimal

from daytrade_it.application.signals.entry_filter import (
    MAX_PRE_ENTRY_MOVE,
    generate_no_chase_confirmation,
)
from daytrade_it.domain.entities.signal import SignalDirection, SignalSource
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.brokers.base import Quote
from daytrade_it.infrastructure.external.quant_airflow_db_client import AdjustedDailyBar

_EXPIRES = dt.datetime.now(dt.UTC) + dt.timedelta(hours=1)


class _Broker:
    def __init__(self, bid: str, ask: str) -> None:
        self._quote = Quote(ticker="005930", bid=Decimal(bid), ask=Decimal(ask))

    def get_quote(self, ticker: str) -> Quote:  # noqa: ARG002
        return self._quote


class _DB:
    def __init__(self, prev_close: float | None) -> None:
        self._prev_close = prev_close
        self.calls: list[dict] = []

    def get_adjusted_daily_bars(self, ticker, *, start, end):  # noqa: ANN001, ANN201
        self.calls.append({"start": start, "end": end})
        if self._prev_close is None:
            return []
        return [
            AdjustedDailyBar(code=ticker.value, date=end, close=self._prev_close, source="kiwoom")
        ]


def _run(bid: str, ask: str, prev_close: float | None):  # noqa: ANN202
    db = _DB(prev_close)
    signal = generate_no_chase_confirmation(Ticker(value="005930"), _Broker(bid, ask), db, _EXPIRES)
    return signal, db


def test_flat_stock_confirms_buy() -> None:
    signal, _ = _run("70000", "70100", 70000.0)
    assert signal is not None
    assert signal.direction == SignalDirection.BUY
    assert signal.source == SignalSource.TECHNICAL
    assert signal.expires_at == _EXPIRES


def test_stock_down_before_entry_still_confirms() -> None:
    """Down-before-entry news names did best in the eval (+2.8% excess)."""
    signal, _ = _run("65000", "65100", 70000.0)
    assert signal is not None


def test_extended_stock_is_refused() -> None:
    up = 70000 * (1 + float(MAX_PRE_ENTRY_MOVE)) + 500
    signal, _ = _run(str(int(up)), str(int(up) + 100), 70000.0)
    assert signal is None


def test_up_two_and_a_half_percent_still_confirms() -> None:
    """2.5% was refused under the old 2% cap; the 3% cap won the policy eval."""
    signal, _ = _run("71700", "71750", 70000.0)
    assert signal is not None


def test_limit_up_stock_with_empty_ask_is_refused_not_misread_as_crash() -> None:
    """ask=0 at 상한가 must read as the bid price (+30%, refused), not bid/2."""
    signal, _ = _run("91000", "0", 70000.0)
    assert signal is None


def test_no_prior_close_refuses() -> None:
    signal, _ = _run("70000", "70100", None)
    assert signal is None


def test_prior_close_lookup_excludes_today() -> None:
    """If today's bar is already loaded, it must not be used as 'prior close'."""
    _, db = _run("70000", "70100", 70000.0)
    assert db.calls[0]["end"] < dt.date.today()
