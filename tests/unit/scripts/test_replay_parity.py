"""replay_parity: live decision parsing and live-vs-replay divergence classes."""

import datetime as dt
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from replay_parity import (  # noqa: E402
    divergences,
    live_decisions,
    match_decisions,
    reproducible_from_open,
)

from daytrade_it.infrastructure.brokers.base import KST, MinuteBar

DAY = dt.date(2026, 9, 15)


def _row(hms: str, event: str, **kw: object) -> dict:
    h, m, s = map(int, hms.split(":"))
    utc = dt.datetime(2026, 9, 15, h, m, s, tzinfo=KST).astimezone(dt.UTC)
    return {"event": event, "timestamp": utc.isoformat().replace("+00:00", "Z"), **kw}


def _judged(hms: str, t: str) -> dict:
    return _row(hms, "news_signal_poller: judged", ticker=t)


def _bar(hm: str, o: int) -> MinuteBar:
    h, m = map(int, hm.split(":"))
    p = Decimal(o)
    return MinuteBar(ts=dt.datetime(2026, 9, 15, h, m), open=p, high=p, low=p, close=p, volume=1)


ROWS = [
    _judged("09:01:40", "000001"),
    _row("09:01:40", "entry_filter: extended, no entry", ticker="000001", pre_entry_move="0.05"),
    _judged("09:01:42", "000002"),
    _row(
        "09:01:42",
        "news_signal_poller: ACTIONABLE SIGNAL PERSISTED",
        ticker="000002",
        pre_entry_move="0.01",
    ),
    _judged("09:01:43", "000003"),
    _row("09:01:43", "news_signal_poller: ACTIONABLE SIGNAL PERSISTED", ticker="000003"),
    _judged("09:01:44", "000004"),
    _row("09:01:45", "auto_trader: ENTRY SUBMITTED", ticker="000002", order_id="1", limit="10100"),
    _row("09:01:46", "auto_trader: ENTRY SUBMITTED", ticker="000002", order_id="2", limit="10100"),
    _row("09:01:47", "auto_trader: entry skipped", ticker="000003", reason="cap"),
    _row("09:01:48", "auto_trader: entry skipped", ticker="000002", reason="already entered today"),
]


def test_live_decisions_follow_judged_rows_and_skip_excluded_orders() -> None:
    got = live_decisions(ROWS, DAY, exclude_orders={"2"})
    assert [(d["ticker"], d["outcome"], d.get("reason")) for d in got] == [
        ("000001", "extended", None),
        ("000002", "entered", None),
        ("000003", "skipped", "cap"),
        ("000004", "no_signal", None),
    ]
    assert got[0]["pre_entry_move"] == "0.05"
    assert got[1]["limit"] == "10100"
    assert got[1]["entered_at"].time() == dt.time(9, 1, 45)


def test_match_decisions_pairs_nearest_same_ticker_within_window() -> None:
    live = live_decisions(ROWS, DAY, exclude_orders=set())
    sim = [
        {"ticker": "000002", "decide_at": dt.datetime(2026, 9, 15, 9, 1, 41, tzinfo=KST)},
        {"ticker": "000009", "decide_at": dt.datetime(2026, 9, 15, 9, 1, 41, tzinfo=KST)},
    ]
    pairs = match_decisions(sim, live, within=dt.timedelta(seconds=5))
    assert pairs[0][1] is live[1] and pairs[1][1] is None


def test_reproducible_from_open_rounds_implied_price_to_tick() -> None:
    assert reproducible_from_open(Decimal("71000.4"), Decimal(71000))
    assert not reproducible_from_open(Decimal(71000), Decimal(70800))


def test_divergences_classify_price_not_reproducible_from_open_as_intra_minute() -> None:
    at = dt.datetime(2026, 9, 15, 9, 1, 42, tzinfo=KST)
    live = [
        {"ticker": "A", "at": at, "outcome": "extended", "pre_entry_move": "0.03"},
        {"ticker": "B", "at": at, "outcome": "extended", "pre_entry_move": "0.03"},
        {"ticker": "C", "at": at, "outcome": "no_signal"},
    ]
    sim = [
        {"ticker": "A", "decide_at": at, "outcome": "entered", "limit": "10200"},
        {"ticker": "B", "decide_at": at, "outcome": "entered", "limit": "10300"},
        {"ticker": "C", "decide_at": at, "outcome": "extended", "pre_entry_move": "0.05"},
    ]
    bars = {"A": [_bar("09:01", 10200)], "B": [_bar("09:01", 10300)], "C": [_bar("09:01", 10500)]}
    prev = {"A": Decimal(10000), "B": Decimal(10000), "C": Decimal(10000)}
    rows = divergences(list(zip(sim, live, strict=True)), [], [], [], bars, prev)
    got = {(r["ticker"], r["cls"]) for r in rows}
    # A: live 10300 vs open 10200 -> intra; B: live 10300 == open -> other; C: no live price.
    assert got == {("A", "intra_minute_price"), ("B", "other"), ("C", "other")}


def test_divergences_add_on_and_exit_follow_entry_price_reproducibility() -> None:
    live_ledger = [
        {
            "event": "entry_submitted",
            "ticker": "A",
            "limit": "74900",
            "ts": "2026-09-15T09:01:58+09:00",
        },
        {"event": "exit_submitted", "ticker": "A", "quantity": 1, "attempt": 0},
    ]
    sim_ledger = [
        {"event": "entry_submitted", "ticker": "A", "limit": "74800"},
        {"event": "add_on_submitted", "ticker": "A", "limit": "75600"},
        {"event": "exit_submitted", "ticker": "A", "quantity": 2, "attempt": 0},
    ]
    fills = [{"ticker": "A", "side": "buy", "price": Decimal(74800)}]
    bars = {"A": [_bar("09:01", 74800)]}
    rows = divergences([], live_ledger, sim_ledger, fills, bars, {"A": Decimal(74600)})
    assert {(r["kind"], r["cls"]) for r in rows} == {
        ("add_on", "intra_minute_price"),
        ("exit_qty", "intra_minute_price"),
    }
