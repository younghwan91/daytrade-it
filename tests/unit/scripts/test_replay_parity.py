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


def _knock(follow_live: str, follow_sim: str, root_at: dt.datetime, sim_ledger: list[dict]):  # noqa: ANN202
    bars = {"X": [_bar("09:01", 10200)], "Y": [_bar("09:01", 6850)]}
    prev = {"X": Decimal(10000), "Y": Decimal(7040)}
    root = _pair("X", "extended", "entered", "0.04", at=root_at)
    follow = _pair("Y", follow_live, follow_sim, None)
    rows = divergences([root, follow], [], sim_ledger, [], bars, prev, max_move=Decimal("0.03"))
    return [(r["ticker"], r["cls"]) for r in rows if r["kind"] == "decision"]


def test_knock_on_requires_cap_skip_not_other_pass_mismatch() -> None:
    early = AT - dt.timedelta(seconds=8)
    assert _knock("entered", "enter_failed", early, []) == [
        ("X", "intra_minute_price"),
        ("Y", "other"),
    ]


def test_knock_on_requires_strictly_earlier_root() -> None:
    assert _knock("entered", "skipped", AT, []) == [("X", "intra_minute_price"), ("Y", "other")]


def test_knock_on_requires_root_position_still_open_on_the_skipped_side() -> None:
    early = AT - dt.timedelta(seconds=8)
    exited = [
        {
            "event": "entry_submitted",
            "ticker": "X",
            "limit": "10200",
            "ts": "2026-09-15T09:01:34+09:00",
        },
        {
            "event": "exit_submitted",
            "ticker": "X",
            "quantity": 1,
            "attempt": 0,
            "ts": "2026-09-15T09:01:40+09:00",
        },
    ]
    assert _knock("entered", "skipped", early, exited) == [
        ("X", "intra_minute_price"),
        ("Y", "other"),
    ]
    held = [exited[0], {**exited[1], "ts": "2026-09-15T15:00:00+09:00"}]
    assert _knock("entered", "skipped", early, held) == [
        ("X", "intra_minute_price"),
        ("Y", "knock_on"),
    ]


def test_knock_on_requires_root_entered_on_the_side_that_was_cap_skipped() -> None:
    """Root entered in the replay; here the replay entered and live was cap-skipped -> other."""
    early = AT - dt.timedelta(seconds=8)
    assert _knock("skipped", "entered", early, []) == [("X", "intra_minute_price"), ("Y", "other")]


def test_exit_qty_inherits_add_on_class_only_when_quantities_reconcile() -> None:
    live, sim, fills = _add_on_ledgers()
    bars, prev = {"A": _rising(75600)}, {"A": Decimal(74600)}
    rows = divergences([], live, sim, fills, bars, prev, limits=TradingLimits())
    assert ("exit_qty", "intra_minute_price") in {(r["kind"], r["cls"]) for r in rows}
    sim0 = [r for r in sim if r["event"] != "exit_submitted"]
    rows = divergences([], live, sim0, fills, bars, prev, limits=TradingLimits())
    assert {(r["kind"], r["cls"]) for r in rows} == {
        ("add_on", "intra_minute_price"),
        ("exit_qty", "other"),
    }


# -- final review fix wave -----------------------------------------------------


def test_minute_bar_ts_is_converted_to_kst_wall_time() -> None:
    from replay_parity import to_kst_naive  # noqa: PLC0415

    aware_utc = dt.datetime(2026, 9, 15, 0, 1, tzinfo=dt.UTC)
    assert to_kst_naive(aware_utc) == dt.datetime(2026, 9, 15, 9, 1)
    naive_kst = dt.datetime(2026, 9, 15, 9, 1)
    assert to_kst_naive(naive_kst) == naive_kst


def test_session_bars_before_0800_fail_loudly() -> None:
    import pytest  # noqa: PLC0415
    from replay_parity import check_session_hours  # noqa: PLC0415

    from daytrade_it.application.replay.day_replay import ReplayDataError  # noqa: PLC0415

    check_session_hours({"A": [_bar("08:00", 1), _bar("15:30", 1)]})
    with pytest.raises(ReplayDataError, match="A"):
        check_session_hours({"A": [_bar("00:01", 1)]})


def test_last_regular_closes_ignore_after_hours_and_the_day_itself() -> None:
    from replay_parity import last_regular_closes  # noqa: PLC0415

    rows = [
        ("A", dt.datetime(2026, 9, 14, 15, 30), 2900.0),
        ("A", dt.datetime(2026, 9, 14, 15, 35), 2950.0),  # NXT after-market
        ("A", dt.datetime(2026, 9, 14, 19, 59), 2960.0),
        ("A", dt.datetime(2026, 9, 11, 15, 30), 2800.0),
        ("B", dt.datetime(2026, 9, 11, 15, 19), 500.0),  # halted on 09-14
        ("B", dt.datetime(2026, 9, 11, 15, 32), 510.0),  # closing auction extended by VI
        ("B", dt.datetime(2026, 9, 11, 15, 35), 520.0),  # NXT after-market
        ("A", dt.datetime(2026, 9, 15, 9, 0), 3000.0),
    ]
    got = last_regular_closes(rows, DAY)
    assert got == {
        "A": (Decimal(2900), dt.date(2026, 9, 14)),
        "B": (Decimal(510), dt.date(2026, 9, 11)),
    }


def test_vintage_flags_list_closes_off_by_more_than_a_tenth_of_a_percent() -> None:
    from replay_parity import vintage_flags  # noqa: PLC0415

    closes = {"A": (Decimal(2900), DAY), "B": (Decimal(10000), DAY), "C": (Decimal(10), DAY)}
    daily = {"A": Decimal(2720), "B": Decimal(10005)}
    flags = vintage_flags(closes, daily)
    assert [(f["code"], f["daily_adjusted"]) for f in flags] == [("A", "2720"), ("C", "None")]


def test_frozen_prev_closes_are_built_once_and_reused(tmp_path: Path) -> None:
    from replay_parity import frozen_prev_closes  # noqa: PLC0415

    calls: list[str] = []

    def fetch_bars(codes, day):  # noqa: ANN001, ANN202
        calls.append("bars")
        return [("A", dt.datetime(2026, 9, 14, 15, 30), 2900.0)]

    def fetch_daily(codes, day):  # noqa: ANN001, ANN202
        calls.append("daily")
        return {"A": Decimal(2720)}

    first = frozen_prev_closes(["A", "B"], DAY, tmp_path, fetch_bars, fetch_daily)
    second = frozen_prev_closes(["A", "B"], DAY, tmp_path, fetch_bars, fetch_daily)
    assert calls == ["bars", "daily"]
    assert first == second
    assert first.prices == {"A": Decimal(2900)} and first.missing == ["B"]
    assert (tmp_path / "2026-09-15.json").exists()
    assert [f["code"] for f in first.flags] == ["A"]


def test_frozen_prev_closes_refuse_a_file_missing_event_codes(tmp_path: Path) -> None:
    import pytest  # noqa: PLC0415
    from replay_parity import frozen_prev_closes  # noqa: PLC0415

    frozen_prev_closes(["A"], DAY, tmp_path, lambda _c, _d: [], lambda _c, _d: {})
    with pytest.raises(ValueError, match="Z"):
        frozen_prev_closes(["A", "Z"], DAY, tmp_path, lambda _c, _d: [], lambda _c, _d: {})


def _fill(hms: str, price: int, limit: int, *, resting: bool, side: str = "buy") -> dict:
    h, m, s = map(int, hms.split(":"))
    return {
        "ts": dt.datetime(2026, 9, 15, h, m, s, tzinfo=KST),
        "ticker": "A",
        "side": side,
        "qty": 1,
        "price": Decimal(price),
        "order_id": "S1",
        "limit": Decimal(limit),
        "resting": resting,
    }


def _ohlc(hm: str, o: int, lo: int) -> MinuteBar:
    h, m = map(int, hm.split(":"))
    return MinuteBar(
        ts=dt.datetime(2026, 9, 15, h, m),
        open=Decimal(o),
        high=Decimal(o),
        low=Decimal(lo),
        close=Decimal(o),
        volume=1,
    )


def test_fill_rule_check_flags_buys_off_simbrokers_own_price_rule() -> None:
    from replay_parity import fill_rule_divergences  # noqa: PLC0415

    bars = {"A": [_ohlc("09:01", 1005, 1000), _ohlc("09:02", 1010, 980)]}
    ok = [_fill("09:01:10", 1005, 1010, resting=False), _fill("09:03:00", 985, 985, resting=True)]
    assert fill_rule_divergences(ok, bars) == []
    bad = [_fill("09:01:10", 1010, 1010, resting=False), _fill("09:03:00", 980, 985, resting=True)]
    rows = fill_rule_divergences(bad, bars)
    assert [(r["kind"], r["cls"]) for r in rows] == [("fill_rule", "other")] * 2


def _flat(price: int, n: int = 30) -> list[MinuteBar]:
    out, t = [], dt.datetime(2026, 9, 15, 9, 0)
    for _ in range(n):
        p = Decimal(price)
        out.append(MinuteBar(ts=t, open=p, high=p, low=p, close=p, volume=100))
        t += dt.timedelta(minutes=1)
    return out


def _exit_ledgers(live_entry: str, sim_entry: str, live_exit: str, sim_exit: str, live_limit: str):  # noqa: ANN202
    def led(entry: str, exit_ts: str, limit: str, oid: str) -> list[dict]:
        return [
            {
                "event": "entry_submitted",
                "ticker": "A",
                "limit": entry,
                "ts": "2026-09-15T09:01:00+09:00",
            },
            {
                "event": "exit_submitted",
                "ticker": "A",
                "quantity": 1,
                "attempt": 0,
                "limit": limit,
                "order_id": oid,
                "ts": f"2026-09-15T{exit_ts}+09:00",
            },
        ]

    return led(live_entry, live_exit, live_limit, "L1"), led(sim_entry, sim_exit, "10990", "S9")


def test_exit_within_one_manage_tick_and_half_percent_is_no_divergence() -> None:
    from replay_parity import exit_divergences  # noqa: PLC0415

    live, sim = _exit_ledgers("10700", "10700", "09:29:40", "09:30:00", "10990")
    fills = [{**_fill("09:30:00", 10990, 10990, resting=False, side="sell"), "order_id": "S9"}]
    assert exit_divergences(live, sim, fills, {"A": _flat(11000)}, TradingLimits()) == []


def test_exit_time_gap_is_intra_minute_only_if_entry_limits_flip_the_vwap_trigger() -> None:
    from replay_parity import exit_divergences  # noqa: PLC0415

    bars = {"A": [*_flat(11000, 29), _ohlc("09:29", 10880, 10880)]}
    fills = [{**_fill("09:30:00", 10879, 10879, resting=False, side="sell"), "order_id": "S9"}]
    # close 10880 is +2.06% over 10660 (hit) but +1.68% over 10700 (no hit)
    live, sim = _exit_ledgers("10660", "10700", "09:30:00", "09:40:00", "10879")
    rows = exit_divergences(live, sim, fills, bars, TradingLimits())
    assert [(r["kind"], r["cls"]) for r in rows] == [("exit_time", "intra_minute_price")]
    live, sim = _exit_ledgers("10700", "10700", "09:30:00", "09:40:00", "10879")
    rows = exit_divergences(live, sim, fills, bars, TradingLimits())
    assert [(r["kind"], r["cls"]) for r in rows] == [("exit_time", "other")]


def test_sell_fill_over_half_percent_is_intra_minute_only_if_live_bid_is_off_the_open() -> None:
    from replay_parity import exit_divergences  # noqa: PLC0415

    fills = [{**_fill("09:30:00", 10990, 10990, resting=False, side="sell"), "order_id": "S9"}]
    live, sim = _exit_ledgers("10700", "10700", "09:30:00", "09:30:00", "10900")
    rows = exit_divergences(live, sim, fills, {"A": _flat(11000)}, TradingLimits())
    assert [(r["kind"], r["cls"]) for r in rows] == [("sell_fill", "intra_minute_price")]
    # live limit equals the sim bid, but the sim sold far above it: engine difference
    far = [{**fills[0], "price": Decimal(11200)}]
    live, sim = _exit_ledgers("10700", "10700", "09:30:00", "09:30:00", "10990")
    rows = exit_divergences(live, sim, far, {"A": _flat(11000)}, TradingLimits())
    assert [(r["kind"], r["cls"]) for r in rows] == [("sell_fill", "other")]


def test_day_fails_when_over_5pct_of_event_tickers_lack_a_frozen_prev_close() -> None:
    import pytest  # noqa: PLC0415
    from replay_parity import check_prev_close_coverage  # noqa: PLC0415

    from daytrade_it.application.replay.day_replay import ReplayDataError  # noqa: PLC0415
    from daytrade_it.application.replay.prev_close import FrozenPrevCloses  # noqa: PLC0415

    closes = {f"{i:06d}": (Decimal(1000), dt.date(2026, 9, 14)) for i in range(20)}
    check_prev_close_coverage(FrozenPrevCloses(day=DAY, closes=closes, missing=["999999"]))
    with pytest.raises(ReplayDataError, match="999998"):
        check_prev_close_coverage(
            FrozenPrevCloses(day=DAY, closes=closes, missing=["999998", "999999"])
        )
