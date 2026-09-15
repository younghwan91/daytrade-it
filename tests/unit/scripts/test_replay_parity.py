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
    match_summary,
    reproducible_from_open,
)

from daytrade_it.application.trading.auto_trader import TradingLimits
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


def test_divergences_fill_over_half_percent_with_baseline_rows_in_ledger() -> None:
    live_ledger = [
        {"event": "baseline", "positions": {}},
        {
            "event": "entry_submitted",
            "ticker": "A",
            "limit": "11030",
            "ts": "2026-09-15T09:00:46+09:00",
        },
    ]
    sim_ledger = [{"event": "entry_submitted", "ticker": "A", "limit": "11120"}]
    fills = [{"ticker": "A", "side": "buy", "price": Decimal(11120)}]
    rows = divergences([], live_ledger, sim_ledger, fills, {"A": [_bar("09:00", 11120)]}, {})
    assert [(r["kind"], r["cls"]) for r in rows] == [("fill", "intra_minute_price")]


def test_live_decisions_merge_duplicate_daemon_judgments_of_same_article() -> None:
    rows = [
        _row("09:00:35", "news_signal_poller: judged", ticker="000921", title="x"),
        _row("09:00:36", "news_signal_poller: judged", ticker="000921", title="x"),
        _row("09:00:36", "news_signal_poller: ACTIONABLE SIGNAL PERSISTED", ticker="000921"),
        _row("09:00:38", "news_signal_poller: ACTIONABLE SIGNAL PERSISTED", ticker="000921"),
        _row("09:00:39", "news_signal_poller: judged", ticker="000921", title="other article"),
        _row("09:01:55", "auto_trader: ENTRY SUBMITTED", ticker="000921", order_id="1", limit="1"),
        _row("09:01:56", "auto_trader: ENTRY SUBMITTED", ticker="000921", order_id="2", limit="1"),
    ]
    got = live_decisions(rows, DAY, exclude_orders={"2"})
    assert [d["outcome"] for d in got] == ["entered", "no_signal"]


def test_divergence_with_no_bar_yet_is_intra_minute() -> None:
    """Call auction (e.g. VI at the open): live had a quote, minute bars start later."""
    at = dt.datetime(2026, 9, 15, 9, 1, 35, tzinfo=KST)
    live = {"ticker": "A", "at": at, "outcome": "extended", "pre_entry_move": "0.1"}
    sim = {"ticker": "A", "decide_at": at, "outcome": "no_signal"}
    rows = divergences(
        [(sim, live)], [], [], [], {"A": [_bar("09:02", 11000)]}, {"A": Decimal(10000)}
    )
    assert [r["cls"] for r in rows] == ["intra_minute_price"]
    assert "no bar" in rows[0]["note"]


def test_filter_row_belongs_to_the_latest_judgment_of_that_ticker() -> None:
    """A ticker: a no-signal judgment, then a second article's signal 6 s later."""
    rows = [
        _row("10:51:17", "news_signal_poller: judged", ticker="000316", title="a"),
        _row("10:51:23", "news_signal_poller: judged", ticker="000316", title="b"),
        _row("10:51:24", "news_signal_poller: ACTIONABLE SIGNAL PERSISTED", ticker="000316"),
        _row("10:51:25", "auto_trader: entry skipped", ticker="000316", reason="cap"),
    ]
    got = live_decisions(rows, DAY, exclude_orders=set())
    assert [(d["title"], d["outcome"]) for d in got] == [("a", "no_signal"), ("b", "skipped")]


def test_reproducible_allows_bid_ask_mid_half_tick_below_open() -> None:
    assert reproducible_from_open(Decimal(35675), Decimal(35700))
    assert reproducible_from_open(Decimal(35675), Decimal("35675.0"))
    assert not reproducible_from_open(Decimal(35650), Decimal(35700))


def test_no_bar_takes_precedence_over_prev_close_vintage() -> None:
    at = dt.datetime(2026, 9, 15, 9, 1, 49, tzinfo=KST)
    live = {
        "ticker": "A",
        "at": at,
        "outcome": "extended",
        "pre_entry_move": "0.2922794117647058823529411765",
    }
    sim = {"ticker": "A", "decide_at": at, "outcome": "no_signal"}
    rows = divergences(
        [(sim, live)], [], [], [], {"A": [_bar("09:02", 3515)]}, {"A": Decimal(2900)}
    )
    assert [r["cls"] for r in rows] == ["intra_minute_price"]


AT = dt.datetime(2026, 9, 15, 9, 1, 42, tzinfo=KST)


def _pair(t: str, live_outcome: str, sim_outcome: str, move: str | None, at: dt.datetime = AT):
    live = {"ticker": t, "at": at, "outcome": live_outcome}
    if move is not None:
        live["pre_entry_move"] = move
    if live_outcome == "skipped":
        live["reason"] = "concurrent position cap reached"
    sim = {"ticker": t, "decide_at": at, "outcome": sim_outcome}
    if sim_outcome == "skipped":
        sim["reason"] = "concurrent position cap reached"
    return sim, live


def test_filter_divergence_is_intra_minute_only_when_prices_straddle_the_cap() -> None:
    # A: live 10400 (+4%) extended, sim quote 10200 (+2%) entered -> crosses the 3% cap.
    # B: live 10500 (+5%) extended, sim quote 10450 (+4.5%) entered -> no crossing explains it.
    bars = {"A": [_bar("09:01", 10200)], "B": [_bar("09:01", 10450)]}
    prev = {"A": Decimal(10000), "B": Decimal(10000)}
    pairs = [_pair("A", "extended", "entered", "0.04"), _pair("B", "extended", "entered", "0.05")]
    rows = divergences(pairs, [], [], [], bars, prev, max_move=Decimal("0.03"))
    assert {(r["ticker"], r["cls"]) for r in rows} == {("A", "intra_minute_price"), ("B", "other")}
    assert "+2.00%" in rows[0]["note"] and "+4.00%" in rows[0]["note"]


def test_pre_entry_move_without_tick_grid_solution_is_other() -> None:
    pairs = [_pair("A", "extended", "entered", "0.0412345678901")]
    rows = divergences(
        pairs,
        [],
        [],
        [],
        {"A": [_bar("09:01", 10200)]},
        {"A": Decimal(10000)},
        max_move=Decimal("0.03"),
    )
    assert [(r["cls"], r["note"]) for r in rows] == [
        ("other", "pre_entry_move has no tick-grid solution")
    ]


def test_divergence_from_prev_close_vintage_is_data_vintage() -> None:
    """+2.08% over live prev 324500 passed; +3.03% over kr_quant 321500."""
    pairs = [_pair("A", "skipped", "extended", "0.02080123266563944530046224961")]
    rows = divergences(
        pairs,
        [],
        [],
        [],
        {"A": [_bar("09:01", 331250)]},
        {"A": Decimal(321500)},
        max_move=Decimal("0.03"),
    )
    assert [r["cls"] for r in rows] == ["data_vintage"]
    assert "live prev 324500" in rows[0]["note"]


def test_prev_close_within_a_tenth_of_a_percent_is_not_data_vintage() -> None:
    """Live prev 100000 vs kr_quant 100050 (0.05%): the price crossing decides."""
    pairs = [_pair("A", "extended", "entered", "0.031")]
    rows = divergences(
        pairs,
        [],
        [],
        [],
        {"A": [_bar("09:01", 102000)]},
        {"A": Decimal(100050)},
        max_move=Decimal("0.03"),
    )
    assert [r["cls"] for r in rows] == ["intra_minute_price"]


def test_slot_divergence_is_knock_on_only_after_an_earlier_passing_divergence() -> None:
    """Root X entered in replay (intra), so follow-on Y hit the cap there."""
    early = AT - dt.timedelta(seconds=8)
    bars = {"X": [_bar("09:01", 10200)], "Y": [_bar("09:01", 6850)]}
    prev = {"X": Decimal(10000), "Y": Decimal(7040)}
    root = _pair("X", "extended", "entered", "0.04", at=early)
    follow = _pair("Y", "entered", "skipped", None)
    rows = divergences([root, follow], [], [], [], bars, prev, max_move=Decimal("0.03"))
    assert [(r["ticker"], r["cls"]) for r in rows] == [
        ("X", "intra_minute_price"),
        ("Y", "knock_on"),
    ]
    assert "X" in rows[1]["note"]
    alone = divergences([follow], [], [], [], bars, prev, max_move=Decimal("0.03"))
    assert [r["cls"] for r in alone] == ["other"]


def test_unmatched_live_judgment_is_other_divergence() -> None:
    live = {"ticker": "Z", "at": AT, "outcome": "extended", "pre_entry_move": "0.05"}
    rows = divergences([], [], [], [], {}, {}, unmatched_live=[live])
    assert [(r["kind"], r["ticker"], r["cls"]) for r in rows] == [("decision", "Z", "other")]


def _rising(last_close: int) -> list[MinuteBar]:
    out, t = [], dt.datetime(2026, 9, 15, 9, 0)
    for i in range(30):
        c = Decimal(74000 + (last_close - 74000) * i // 29)
        out.append(MinuteBar(ts=t, open=c, high=c, low=c, close=c, volume=100))
        t += dt.timedelta(minutes=1)
    return out


def _add_on_ledgers() -> tuple[list[dict], list[dict], list[dict]]:
    live = [
        {
            "event": "entry_submitted",
            "ticker": "A",
            "limit": "74900",
            "ts": "2026-09-15T09:01:58+09:00",
        },
        {"event": "exit_submitted", "ticker": "A", "quantity": 1, "attempt": 0},
    ]
    sim = [
        {
            "event": "entry_submitted",
            "ticker": "A",
            "limit": "74800",
            "ts": "2026-09-15T09:01:51+09:00",
        },
        {
            "event": "add_on_submitted",
            "ticker": "A",
            "limit": "75600",
            "ts": "2026-09-15T09:30:00+09:00",
        },
        {"event": "exit_submitted", "ticker": "A", "quantity": 2, "attempt": 0},
    ]
    fills = [{"ticker": "A", "side": "buy", "price": Decimal(74800)}]
    return live, sim, fills


def test_add_on_divergence_is_intra_minute_when_entry_limit_flips_the_trigger() -> None:
    """close 75600 is +1.07% over 74800 but +0.93% over 74900."""
    live, sim, fills = _add_on_ledgers()
    rows = divergences(
        [],
        live,
        sim,
        fills,
        {"A": _rising(75600)},
        {"A": Decimal(74600)},
        limits=TradingLimits(),
    )
    assert {(r["kind"], r["cls"]) for r in rows} == {
        ("add_on", "intra_minute_price"),
        ("exit_qty", "intra_minute_price"),
    }


def test_add_on_divergence_without_trigger_flip_is_other() -> None:
    live, sim, fills = _add_on_ledgers()
    rows = divergences(
        [],
        live,
        sim,
        fills,
        {"A": _rising(76500)},
        {"A": Decimal(74600)},
        limits=TradingLimits(),
    )
    assert {(r["kind"], r["cls"]) for r in rows} == {("add_on", "other"), ("exit_qty", "other")}


def test_match_summary_counts_and_returns_unused_live_judgments() -> None:
    live = live_decisions(ROWS, DAY, exclude_orders=set())
    sim = [{"ticker": "000002", "decide_at": dt.datetime(2026, 9, 15, 9, 1, 41, tzinfo=KST)}]
    pairs = match_decisions(sim, live, within=dt.timedelta(seconds=5))
    unused, counts = match_summary(pairs, live)
    assert [d["ticker"] for d in unused] == ["000001", "000003", "000004"]
    assert counts == {"live": 4, "replay": 1, "matched": 1, "union": 4}
