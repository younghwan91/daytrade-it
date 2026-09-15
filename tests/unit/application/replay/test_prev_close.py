"""Frozen previous closes: the prior session's last regular minute bar, served to replay."""

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
from daytrade_it.application.signals.entry_filter import generate_no_chase_confirmation
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
