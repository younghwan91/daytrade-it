"""No-chase entry filter: BUY confirmation only when the stock isn't extended."""

import datetime as dt
from decimal import Decimal

from daytrade_it.application.signals.entry_filter import (
    DAILY_FALLBACK,
    KRX_REGULAR,
    MAX_PRE_ENTRY_MOVE,
    choose_prev_close,
    generate_no_chase_confirmation,
)
from daytrade_it.domain.entities.signal import SignalDirection, SignalSource
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.brokers.base import Quote
from daytrade_it.infrastructure.external.quant_airflow_db_client import (
    AdjustedDailyBar,
    RegularSessionClose,
)

_EXPIRES = dt.datetime.now(dt.UTC) + dt.timedelta(hours=1)


class _Broker:
    def __init__(self, bid: str, ask: str) -> None:
        self._quote = Quote(ticker="005930", bid=Decimal(bid), ask=Decimal(ask))

    def get_quote(self, ticker: str) -> Quote:  # noqa: ARG002
        return self._quote


_YESTERDAY = dt.date.today() - dt.timedelta(days=1)


class _DB:
    """Daily close (after-market incl. from 2026-09-14) and the regular-session close.

    `regular` defaults to the same close as `daily` on the same date; pass `regular=None`
    for a ticker without regular minute bars.
    """

    _SAME = object()

    def __init__(
        self,
        daily: float | None,
        regular: float | None | object = _SAME,
        *,
        regular_date: dt.date = _YESTERDAY,
    ) -> None:
        self._daily = daily
        self._regular = daily if regular is self._SAME else regular
        self._regular_date = regular_date
        self.calls: list[dict] = []
        self.regular_calls: list[dict] = []

    def get_adjusted_daily_bars(self, ticker, *, start, end):  # noqa: ANN001, ANN201
        self.calls.append({"start": start, "end": end})
        if self._daily is None:
            return []
        return [AdjustedDailyBar(code=ticker.value, date=end, close=self._daily, source="kiwoom")]

    def get_regular_session_close(self, ticker, *, before, lookback_days):  # noqa: ANN001, ANN201
        self.regular_calls.append({"before": before, "lookback_days": lookback_days})
        if self._regular is None:
            return None
        return RegularSessionClose(code=ticker.value, date=self._regular_date, close=self._regular)


def _run(bid: str, ask: str, prev_close: float | None, **db_kw):  # noqa: ANN003, ANN202
    db = _DB(prev_close, **db_kw)
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
    assert db.regular_calls[0]["before"] == dt.date.today()


# R17 basis: the KRX regular-session close, not the after-market-inclusive daily close.
# Numbers from 008930 on 2026-10-08: daily close 66,200 (after-market) vs regular 58,100.


def test_move_is_measured_against_the_regular_session_close() -> None:
    """+3.3% over the regular close is extended even though it is -9% vs the daily close."""
    signal, _ = _run("60000", "60000", 66200.0, regular=58100.0)
    assert signal is None


def test_after_market_drop_does_not_block_a_stock_flat_on_the_regular_close() -> None:
    """Daily close 5% below the regular close (after-market sell-off): +1% vs regular passes."""
    signal, _ = _run("101000", "101000", 95000.0, regular=100000.0)
    assert signal is not None
    assert signal.metadata["prev_close"] == "100000.0"
    assert signal.metadata["prev_close_source"] == KRX_REGULAR


def test_no_regular_minute_bar_falls_back_to_the_daily_close() -> None:
    """Small caps outside scalp-it's minute-bar universe: daily close, tagged."""
    signal, _ = _run("70000", "70100", 70000.0, regular=None)
    assert signal is not None
    assert signal.metadata["prev_close_source"] == DAILY_FALLBACK
    assert signal.metadata["prev_close"] == "70000.0"


def test_regular_bar_older_than_the_daily_bar_is_stale_and_not_used() -> None:
    """Minute bars missing for the prior session: the regular close found is an older
    session's; the daily close is the prior session's and wins."""
    signal, _ = _run(
        "70000", "70100", 70000.0, regular=60000.0, regular_date=_YESTERDAY - dt.timedelta(days=1)
    )
    assert signal is not None
    assert signal.metadata["prev_close_source"] == DAILY_FALLBACK


def test_choose_prev_close_prefers_a_regular_bar_newer_than_the_daily_bar() -> None:
    """daily_bars_adjusted not loaded for the prior session yet: the minute bar is newer."""
    d0, d1 = dt.date(2026, 10, 7), dt.date(2026, 10, 8)
    daily = AdjustedDailyBar(code="A", date=d0, close=100.0, source="kiwoom")
    regular = RegularSessionClose(code="A", date=d1, close=110.0)
    got = choose_prev_close(daily, regular)
    assert got is not None
    assert (got.price, got.date, got.source) == (Decimal("110.0"), d1, KRX_REGULAR)
    assert choose_prev_close(None, None) is None
    only_daily = choose_prev_close(daily, None)
    assert only_daily is not None
    assert only_daily.source == DAILY_FALLBACK


# M1: a failing minute_bars read (DB outage, statement timeout) must not drop the pair.


class _RecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):  # noqa: ANN204
        def log(event: str, **kw: object) -> None:
            self.records.append((level, event, kw))

        return log


class _FailingRegularDB(_DB):
    def get_regular_session_close(self, ticker, *, before, lookback_days):  # noqa: ANN001, ANN201
        from daytrade_it.shared.errors.exceptions import DatabaseError

        raise DatabaseError(message="canceling statement due to statement timeout")


def test_regular_close_query_error_falls_back_to_the_daily_close(monkeypatch) -> None:  # noqa: ANN001
    from daytrade_it.application.signals import entry_filter

    log = _RecordingLogger()
    monkeypatch.setattr(entry_filter, "logger", log)
    signal = generate_no_chase_confirmation(
        Ticker(value="005930"), _Broker("70000", "70100"), _FailingRegularDB(70000.0), _EXPIRES
    )
    assert signal is not None
    assert signal.metadata["prev_close_source"] == DAILY_FALLBACK
    [(level, _ev, kw)] = [r for r in log.records if "fallback" in r[1]]
    assert level == "warning"
    assert "statement timeout" in kw["regular_error"]


def test_pre_switch_fallback_is_debug_not_a_warning(monkeypatch) -> None:  # noqa: ANN001
    """Replaying a day before REGULAR_CLOSE_FROM: the daily close was the rule, every pair
    falls back -- one warning per pair would bury real ones."""
    import time_machine

    from daytrade_it.application.signals import entry_filter

    log = _RecordingLogger()
    monkeypatch.setattr(entry_filter, "logger", log)
    day = entry_filter.REGULAR_CLOSE_FROM - dt.timedelta(days=1)
    with time_machine.travel(dt.datetime.combine(day, dt.time(9, 30)), tick=False):
        signal = generate_no_chase_confirmation(
            Ticker(value="005930"), _Broker("70000", "70100"), _DB(70000.0, None), _EXPIRES
        )
    assert signal is not None
    assert [lvl for lvl, ev, _ in log.records if "fallback" in ev] == ["debug"]
