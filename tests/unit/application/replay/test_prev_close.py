"""Frozen previous closes served to replay: daily-close basis before the R17 switch, regular after."""

import datetime as dt
from decimal import Decimal
from pathlib import Path

import time_machine

from daytrade_it.application.replay.prev_close import (
    FrozenPrevCloseDB,
    FrozenPrevCloses,
    dump_prev_closes,
    load_prev_closes,
)
from daytrade_it.application.signals.entry_filter import (
    REGULAR_CLOSE_FROM,
    generate_no_chase_confirmation,
)
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.brokers.base import KST, Quote

DAY = dt.date(2026, 9, 15)
PREV = dt.date(2026, 9, 14)
FROZEN = FrozenPrevCloses(
    day=DAY,
    closes={"424870": (Decimal(2900), PREV)},
    krx={"424870": (Decimal(2880), PREV)},
    flags=[{"code": "424870", "frozen": "2900", "daily_adjusted": "2720", "dev": "0.0662"}],
)


class _Broker:
    def get_quote(self, ticker: str) -> Quote:
        p = Decimal(2950)
        return Quote(ticker=ticker, bid=p, ask=p, last_price=p)


def test_dump_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "2026-09-15.json"
    dump_prev_closes(path, FROZEN)
    assert load_prev_closes(path) == FROZEN


def test_stub_serves_one_bar_with_the_frozen_close_and_prior_date() -> None:
    db = FrozenPrevCloseDB(FROZEN)
    [bar] = db.get_adjusted_daily_bars(
        Ticker(value="424870"), start=DAY - dt.timedelta(days=10), end=PREV
    )
    assert (bar.date, Decimal(str(bar.close))) == (PREV, Decimal(2900))
    assert db.get_adjusted_daily_bars(Ticker(value="000001"), start=PREV, end=PREV) == []


def test_no_chase_filter_reads_the_frozen_close() -> None:
    with time_machine.travel(dt.datetime(2026, 9, 15, 9, 30, tzinfo=KST), tick=False):
        sig = generate_no_chase_confirmation(
            Ticker(value="424870"),
            _Broker(),
            FrozenPrevCloseDB(FROZEN),
            dt.datetime(2026, 9, 15, 15, 20, tzinfo=KST),
        )
    assert sig is not None
    assert Decimal(sig.metadata["prev_close"]) == Decimal(2900)
    assert sig.metadata["prev_close_date"] == "2026-09-14"


def test_prev_session_is_the_latest_close_date() -> None:
    f = FrozenPrevCloses(
        day=DAY, closes={"a": (Decimal(1), PREV), "b": (Decimal(2), dt.date(2026, 9, 11))}
    )
    assert f.prev_session == PREV
    assert f.prices == {"a": Decimal(1), "b": Decimal(2)}


def test_krx_prices_is_the_regular_session_close_not_the_unified_close() -> None:
    """R18: price-limit base must be the KRX close, distinct from the unified close."""
    f = FrozenPrevCloses(
        day=DAY,
        closes={"424870": (Decimal(2900), PREV)},
        krx={"424870": (Decimal(2705), PREV)},
    )
    assert f.krx_prices == {"424870": Decimal(2705)}


# R17 basis per day: live read the daily close before REGULAR_CLOSE_FROM and the
# regular-session close from it; the replay must serve the same.
NEW_DAY = REGULAR_CLOSE_FROM
NEW_PREV = NEW_DAY - dt.timedelta(days=1)


def _frozen_new(**kw) -> FrozenPrevCloses:  # noqa: ANN003
    return FrozenPrevCloses(
        day=NEW_DAY,
        closes={"424870": (Decimal(2900), NEW_PREV), "000001": (Decimal(500), NEW_PREV)},
        krx={"424870": (Decimal(2850), NEW_PREV)},
        **kw,
    )


def _filter(frozen: FrozenPrevCloses, ticker: str, when: dt.date):  # noqa: ANN202
    with time_machine.travel(dt.datetime.combine(when, dt.time(9, 30), tzinfo=KST), tick=False):
        return generate_no_chase_confirmation(
            Ticker(value=ticker),
            _Broker(),
            FrozenPrevCloseDB(frozen),
            dt.datetime.combine(when, dt.time(15, 20), tzinfo=KST),
        )


def test_before_the_switch_the_stub_serves_no_regular_close() -> None:
    """Pre-switch days replay the after-market-inclusive daily close live used."""
    db = FrozenPrevCloseDB(FROZEN)
    assert not FROZEN.regular_no_chase
    assert (
        db.get_regular_session_close(Ticker(value="424870"), before=DAY, lookback_days=10) is None
    )
    assert FROZEN.no_chase_prices == FROZEN.prices
    assert FROZEN.no_chase_fallbacks == []


def test_from_the_switch_the_filter_measures_against_the_frozen_regular_close() -> None:
    f = _frozen_new()
    assert f.regular_no_chase
    sig = _filter(f, "424870", NEW_DAY)  # 2950 / 2850 = +3.5% > 3%: extended
    assert sig is None
    assert f.no_chase_prices["424870"] == Decimal(2850)


def test_from_the_switch_a_code_without_regular_bar_falls_back_to_the_daily_close() -> None:
    f = _frozen_new()
    assert f.no_chase_fallbacks == ["000001"]
    assert f.no_chase_prices["000001"] == Decimal(500)


def test_regular_close_older_than_the_daily_close_is_a_fallback() -> None:
    f = FrozenPrevCloses(
        day=NEW_DAY,
        closes={"A": (Decimal(100), NEW_PREV)},
        krx={"A": (Decimal(90), NEW_PREV - dt.timedelta(days=1))},
    )
    assert f.no_chase_fallbacks == ["A"]
    assert f.no_chase_prices == {"A": Decimal(100)}


def test_daily_round_trips_and_feeds_the_filter_from_the_switch(tmp_path: Path) -> None:
    """From REGULAR_CLOSE_FROM the stub's daily input is the frozen daily_bars_adjusted row,
    not `closes`: a code whose minute bars stop a session early (closes and krx both on the
    older date) is compared, as live does, to the prior session's daily row and falls back."""
    older = NEW_PREV - dt.timedelta(days=3)
    f = FrozenPrevCloses(
        day=NEW_DAY,
        closes={"328380": (Decimal(3770), older)},
        krx={"328380": (Decimal(3760), older)},
        daily={"328380": (Decimal(3600), NEW_PREV)},
    )
    path = tmp_path / "f.json"
    dump_prev_closes(path, f)
    assert load_prev_closes(path) == f
    assert f.no_chase_prices == {"328380": Decimal(3600)}
    assert f.no_chase_fallbacks == ["328380"]
    # before the switch the daily input stays `closes` (files frozen before `daily`)
    pre = FrozenPrevCloses(day=DAY, closes=FROZEN.closes, daily={"424870": (Decimal(1), PREV)})
    assert pre.no_chase_prices == {"424870": Decimal(2900)}
