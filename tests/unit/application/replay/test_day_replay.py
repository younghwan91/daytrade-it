"""replay_day drives the live AutoTrader through a synthetic session."""

import datetime as dt
import json
import os
import sqlite3
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from daytrade_it.application.replay.day_replay import (
    DayConfig,
    ReplayDataError,
    replay_day,
    snapshot_dart,
)
from daytrade_it.application.replay.events import ReplayEvent
from daytrade_it.application.signals.news_entry import SelectionConfig
from daytrade_it.application.trading.auto_trader import TradingLimits
from daytrade_it.application.trading.policy import NewsPolicy
from daytrade_it.infrastructure.brokers.base import KST, MinuteBar
from daytrade_it.infrastructure.brokers.sim_broker import SimBroker
from daytrade_it.infrastructure.external.quant_airflow_db_client import StockInfo

DAY = dt.date(2026, 9, 15)
V2 = {
    "subject": "main",
    "reports_price_move": False,
    "material_type": "수주",
    "persistence": "multi_quarter",
    "specificity": 3,
    "freshness": "new",
    "surprise": 2,
    "sentiment_direction": 1,
    "sentiment_strength": 3,
    "scale_vs_size": "large",
}


def _bars(ticker_open: int) -> list[MinuteBar]:
    out, t = [], dt.datetime(2026, 9, 15, 9, 0)
    while t.time() <= dt.time(15, 30):
        p = Decimal(ticker_open)
        out.append(MinuteBar(ts=t, open=p, high=p + 10, low=p - 10, close=p, volume=1000))
        t += dt.timedelta(minutes=1)
    return out


def _event(aid_suffix: str, ticker: str, hms: str) -> ReplayEvent:
    h, m, s = map(int, hms.split(":"))
    at = dt.datetime(2026, 9, 15, h, m, s, tzinfo=KST)
    return ReplayEvent(
        key=f"k{aid_suffix}",
        article_id=f"0a0a0a0a-0000-5000-8000-00000000000{aid_suffix}",
        ticker=ticker,
        name="n",
        market="KOSPI",
        sector="s",
        title="t",
        category="stock",
        published_at=at - dt.timedelta(minutes=5),
        decide_at=at,
        batch_at=at + dt.timedelta(seconds=1),
        features=V2,
        source="archive",
    )


class _Bar:
    def __init__(self) -> None:
        self.close, self.date = 10000.0, DAY - dt.timedelta(days=1)


class _DB:
    def get_adjusted_daily_bars(self, ticker, *, start=None, end=None):  # noqa: ANN001, ANN201
        return [_Bar()]


def _dart(tmp: Path) -> Path:
    p = tmp / "dart.db"
    con = sqlite3.connect(p)
    con.execute(
        "CREATE TABLE disclosures (rcept_no TEXT PRIMARY KEY, rcept_dt TEXT NOT NULL, "
        "code TEXT NOT NULL, corp_name TEXT, corp_cls TEXT, report_nm TEXT, "
        "material_type TEXT, is_correction INTEGER)"
    )
    con.execute("INSERT INTO disclosures VALUES ('1','20260916','000002','x','Y','상장폐지','x',0)")
    con.commit()
    con.close()
    return p


def test_snapshot_dart_drops_future_filings(tmp_path: Path) -> None:
    snap = snapshot_dart(_dart(tmp_path), DAY, tmp_path / "snap.db")
    assert sqlite3.connect(snap).execute("SELECT count(*) FROM disclosures").fetchone() == (0,)


def test_replay_enters_ranked_batch_respects_cap_and_flattens(tmp_path: Path) -> None:
    clock_holder: dict = {}
    bars = {"000001": _bars(10100), "000002": _bars(10100), "000003": _bars(10100)}
    broker = SimBroker(bars, {t: Decimal(10000) for t in bars}, lambda: clock_holder["now"]())
    stocks = {t: StockInfo(code=t, name=t, market="KOSPI", sector="s") for t in bars}
    events = [
        _event("1", "000001", "09:00:35"),
        _event("2", "000002", "09:00:35"),
        _event("3", "000003", "09:00:35"),
    ]
    cfg = DayConfig(
        policy=NewsPolicy(
            limits=TradingLimits(max_open_positions=2, vwap_exit_gap=None, add_on_min_gain=None)
        ),
    )
    res = replay_day(
        DAY,
        events,
        broker,
        _DB(),
        stocks,
        dt.datetime(2026, 9, 14, 15, 20, tzinfo=KST),
        cfg,
        tmp_path,
        _dart(tmp_path),
        clock_holder=clock_holder,
    )
    entries = [r["ticker"] for r in res.ledger if r["event"] == "entry_submitted"]
    exits = [r["ticker"] for r in res.ledger if r["event"] == "exit_submitted"]
    assert len(entries) == 2 and sorted(exits) == sorted(entries)
    assert any(s["reason"] == "concurrent position cap reached" for s in res.skips)
    assert all(r["ts"].startswith("2026-09-15") for r in res.ledger)
    assert [f["side"] for f in res.fills].count("sell") == 2


def _run_day(tmp_path: Path, broker: SimBroker, clock_holder: dict):  # noqa: ANN202
    stocks = {
        t: StockInfo(code=t, name=t, market="KOSPI", sector="s") for t in ("000001", "000002")
    }
    events = [_event("1", "000001", "09:00:35"), _event("2", "000002", "09:00:35")]
    cfg = DayConfig(
        policy=NewsPolicy(
            limits=TradingLimits(max_open_positions=2, vwap_exit_gap=None, add_on_min_gain=None)
        ),
    )
    return replay_day(
        DAY,
        events,
        broker,
        _DB(),
        stocks,
        dt.datetime(2026, 9, 14, 15, 20, tzinfo=KST),
        cfg,
        tmp_path / "work",
        tmp_path / "dart.db" if (tmp_path / "dart.db").exists() else _dart(tmp_path),
        clock_holder=clock_holder,
    )


def _fresh_broker(cls: type[SimBroker] = SimBroker) -> tuple[SimBroker, dict]:
    clock_holder: dict = {}
    bars = {"000001": _bars(10100), "000002": _bars(10100)}
    broker = cls(bars, {t: Decimal(10000) for t in bars}, lambda: clock_holder["now"]())
    return broker, clock_holder


def test_rerunning_same_day_into_same_work_dir_is_identical(tmp_path: Path) -> None:
    first = _run_day(tmp_path, *_fresh_broker())
    second = _run_day(tmp_path, *_fresh_broker())

    def tickers(res, event):  # noqa: ANN001, ANN202
        return [r["ticker"] for r in res.ledger if r["event"] == event]

    assert tickers(first, "entry_submitted")
    assert tickers(second, "entry_submitted") == tickers(first, "entry_submitted")
    assert tickers(second, "exit_submitted") == tickers(first, "exit_submitted")
    assert len(second.ledger) == len(first.ledger)
    assert [f["side"] for f in second.fills] == [f["side"] for f in first.fills]
    assert second.skips == first.skips


class _BrokenPositions(SimBroker):
    def get_positions(self):  # noqa: ANN201
        raise RuntimeError("positions endpoint down")


def test_swallowed_errors_surface_in_result(tmp_path: Path) -> None:
    res = _run_day(tmp_path, *_fresh_broker(_BrokenPositions))
    assert res.errors
    assert not [r for r in res.ledger if r["event"] == "entry_submitted"]
    assert any("positions endpoint down" in e.get("exception", "") for e in res.errors)


def _cap1_day(tmp_path: Path, enter_on_decide: bool):  # noqa: ANN202
    """Two pairs in one poll: 000001 judged first (surprise 1), 000002 later (surprise 3)."""
    clock_holder: dict = {}
    bars = {"000001": _bars(10100), "000002": _bars(10100)}
    broker = SimBroker(bars, {t: Decimal(10000) for t in bars}, lambda: clock_holder["now"]())
    stocks = {t: StockInfo(code=t, name=t, market="KOSPI", sector="s") for t in bars}
    first = replace(_event("1", "000001", "09:00:35"), features={**V2, "surprise": 1})
    second = replace(_event("2", "000002", "09:00:40"), features={**V2, "surprise": 3})
    first = replace(first, batch_at=second.batch_at)
    cfg = DayConfig(
        policy=NewsPolicy(
            limits=TradingLimits(max_open_positions=1, vwap_exit_gap=None, add_on_min_gain=None)
        ),
        enter_on_decide=enter_on_decide,
    )
    return replay_day(
        DAY,
        [first, second],
        broker,
        _DB(),
        stocks,
        dt.datetime(2026, 9, 14, 15, 20, tzinfo=KST),
        cfg,
        tmp_path,
        _dart(tmp_path),
        clock_holder=clock_holder,
    )


def test_ranked_batch_enters_highest_surprise_at_batch_end(tmp_path: Path) -> None:
    res = _cap1_day(tmp_path, enter_on_decide=False)
    entries = [r for r in res.ledger if r["event"] == "entry_submitted"]
    assert [r["ticker"] for r in entries] == ["000002"]
    assert entries[0]["ts"].startswith("2026-09-15T09:00:41")


def test_enter_on_decide_enters_each_pair_when_judged_unranked(tmp_path: Path) -> None:
    """Poller before d357b88 (live 09-14): trader.enter right after each judgment."""
    res = _cap1_day(tmp_path, enter_on_decide=True)
    entries = [r for r in res.ledger if r["event"] == "entry_submitted"]
    assert [r["ticker"] for r in entries] == ["000001"]
    assert entries[0]["ts"].startswith("2026-09-15T09:00:35")


def test_decisions_record_each_judged_pair_outcome(tmp_path: Path) -> None:
    res = _cap1_day(tmp_path, enter_on_decide=False)
    got = [(d["ticker"], d["outcome"], d.get("reason")) for d in res.decisions]
    assert got == [
        ("000001", "skipped", "concurrent position cap reached"),
        ("000002", "entered", None),
    ]
    assert res.decisions[1]["decide_at"] == dt.datetime(2026, 9, 15, 9, 0, 40, tzinfo=KST)
    assert res.decisions[1]["limit"] == "10100"


def test_decisions_record_extended_pairs_with_pre_entry_move(tmp_path: Path) -> None:
    clock_holder: dict = {}
    bars = {"000001": _bars(10500)}
    broker = SimBroker(bars, {"000001": Decimal(10000)}, lambda: clock_holder["now"]())
    stocks = {"000001": StockInfo(code="000001", name="n", market="KOSPI", sector="s")}
    cfg = DayConfig(
        policy=NewsPolicy(limits=TradingLimits(vwap_exit_gap=None, add_on_min_gain=None)),
        enter_on_decide=True,
    )
    res = replay_day(
        DAY,
        [_event("1", "000001", "09:00:35")],
        broker,
        _DB(),
        stocks,
        dt.datetime(2026, 9, 14, 15, 20, tzinfo=KST),
        cfg,
        tmp_path,
        _dart(tmp_path),
        clock_holder=clock_holder,
    )
    assert [(d["outcome"], d["pre_entry_move"]) for d in res.decisions] == [("extended", "0.05")]


# -- final review fix wave -----------------------------------------------------


def _replay(
    tmp_path: Path,
    broker: SimBroker,
    clock_holder: dict,
    cfg: DayConfig,
    events: list[ReplayEvent] | None = None,
    dart: Path | None = None,
    work: Path | None = None,
):  # noqa: ANN202
    events = events or [_event("1", "000001", "09:00:35"), _event("2", "000002", "09:00:35")]
    stocks = {
        e.ticker: StockInfo(code=e.ticker, name="n", market="KOSPI", sector="s") for e in events
    }
    if dart is None:
        dart = tmp_path / "dart.db" if (tmp_path / "dart.db").exists() else _dart(tmp_path)
    return replay_day(
        DAY,
        events,
        broker,
        _DB(),
        stocks,
        dt.datetime(2026, 9, 14, 15, 20, tzinfo=KST),
        cfg,
        work or tmp_path / "work",
        dart,
        clock_holder=clock_holder,
    )


def _cfg(**kw: object) -> DayConfig:
    limits = {"max_open_positions": 2, "vwap_exit_gap": None, "add_on_min_gain": None, **kw}
    return DayConfig(policy=NewsPolicy(limits=TradingLimits(**limits)))


def test_risk_gate_refusals_are_counted_with_marked_positions(tmp_path: Path) -> None:
    """Exposure cap 10,000: the second entry sees 10,100 of marked holdings and is refused."""
    res = _replay(tmp_path, *_fresh_broker(), _cfg(max_trader_exposure=Decimal(10000)))
    assert len([r for r in res.ledger if r["event"] == "entry_submitted"]) == 1
    assert len(res.refusals) == 1
    assert "exposure" in res.refusals[0]["reason"]


def test_reused_broker_reports_only_this_calls_fills(tmp_path: Path) -> None:
    broker, clock_holder = _fresh_broker()
    first = _replay(tmp_path, broker, clock_holder, _cfg())
    second = _replay(tmp_path, broker, clock_holder, _cfg())
    assert first.fills
    assert {f["order_id"] for f in first.fills}.isdisjoint({f["order_id"] for f in second.fills})
    assert len(second.fills) == len(first.fills)


def test_two_configs_into_one_work_root_keep_independent_ledgers(tmp_path: Path) -> None:
    root = tmp_path / "root"
    one = _replay(tmp_path, *_fresh_broker(), _cfg(max_open_positions=1), work=root)
    two = _replay(tmp_path, *_fresh_broker(), _cfg(max_open_positions=2), work=root)
    assert one.run_dir != two.run_dir
    on_disk = [
        [json.loads(x) for x in (r.run_dir / "ledger" / f"{DAY}.jsonl").read_text().splitlines()]
        for r in (one, two)
    ]
    assert on_disk == [one.ledger, two.ledger]
    count = [sum(r["event"] == "entry_submitted" for r in led) for led in on_disk]
    assert count == [1, 2]
    [snap] = list((root / "dart_snapshots").glob("*.db"))
    assert not os.access(snap, os.W_OK)


def test_replay_fails_loudly_when_dart_source_lags_the_day(tmp_path: Path) -> None:
    p = tmp_path / "old.db"
    con = sqlite3.connect(p)
    con.execute(
        "CREATE TABLE disclosures (rcept_no TEXT, rcept_dt TEXT, code TEXT, report_nm TEXT)"
    )
    con.execute("INSERT INTO disclosures VALUES ('1','20260914','000002','x')")
    con.commit()
    con.close()
    with pytest.raises(ReplayDataError, match="rcept_dt"):
        _replay(tmp_path, *_fresh_broker(), _cfg(), dart=p)


def test_replay_fails_when_over_5pct_of_event_tickers_lack_bars(tmp_path: Path) -> None:
    events = [_event("1", "000001", "09:00:35"), _event("2", "000009", "09:00:35")]
    with pytest.raises(ReplayDataError, match="000009"):
        _replay(tmp_path, *_fresh_broker(), _cfg(), events=events)


def test_leftover_positions_at_end_of_replay_are_reported(tmp_path: Path) -> None:
    cfg = replace(_cfg(), exits_until=dt.time(14, 0))  # stop before flatten_at
    res = _replay(tmp_path, *_fresh_broker(), cfg)
    assert res.positions == {"000001": 1, "000002": 1}


# -- Task 5: replay driven by NewsPolicy ---------------------------------------


def test_policy_max_pre_move_sets_the_no_chase_cap(tmp_path: Path) -> None:
    """10,250 vs prev 10,000 = +2.5%: enters under the default 3% cap, extended at 2%."""
    limits = TradingLimits(vwap_exit_gap=None, add_on_min_gain=None)
    outcomes = []
    for sel in (SelectionConfig(), SelectionConfig(max_pre_move=Decimal("0.02"))):
        clock_holder: dict = {}
        bars = {"000001": _bars(10250)}
        broker = SimBroker(bars, {"000001": Decimal(10000)}, lambda h=clock_holder: h["now"]())
        cfg = DayConfig(policy=NewsPolicy(limits=limits, selection=sel), enter_on_decide=True)
        res = _replay(
            tmp_path, broker, clock_holder, cfg, events=[_event("1", "000001", "09:00:35")]
        )
        outcomes.append([d["outcome"] for d in res.decisions])
    assert outcomes == [["entered"], ["extended"]]


def test_policy_rank_key_orders_the_batch(tmp_path: Path) -> None:
    """Cap 1: surprise ranks 000002 (surprise 3) first; score ranks 000001 (strength 3)."""
    first = replace(
        _event("1", "000001", "09:00:35"),
        features={**V2, "surprise": 1, "sentiment_strength": 3},
    )
    second = replace(
        _event("2", "000002", "09:00:35"),
        features={**V2, "surprise": 3, "sentiment_strength": 2},
    )
    entered = []
    for key in ("surprise", "score"):
        limits = TradingLimits(max_open_positions=1, vwap_exit_gap=None, add_on_min_gain=None)
        cfg = DayConfig(policy=NewsPolicy(limits=limits, selection=SelectionConfig(rank_key=key)))
        res = _replay(tmp_path, *_fresh_broker(), cfg, events=[first, second])
        entered.append([r["ticker"] for r in res.ledger if r["event"] == "entry_submitted"])
    assert entered == [["000002"], ["000001"]]


def test_policy_selection_min_score_filters_pairs(tmp_path: Path) -> None:
    cfg = DayConfig(policy=NewsPolicy(selection=SelectionConfig(min_score=1.0)))
    res = _replay(tmp_path, *_fresh_broker(), cfg)
    assert {d["outcome"] for d in res.decisions} == {"no_signal"}


def test_decisions_carry_the_event_key_and_a_deferred_outcome(tmp_path: Path) -> None:
    clock_holder: dict = {}
    bars = {"000001": _bars(10100)}
    broker = SimBroker(bars, {"000001": Decimal(10000)}, lambda: clock_holder["now"]())
    stocks = {"000001": StockInfo(code="000001", name="n", market="KOSPI", sector="s")}
    limits = TradingLimits(
        vwap_exit_gap=None, add_on_min_gain=None, entry_delay=dt.timedelta(minutes=3)
    )
    res = replay_day(
        DAY,
        [_event("1", "000001", "09:00:35")],
        broker,
        _DB(),
        stocks,
        dt.datetime(2026, 9, 14, 15, 20, tzinfo=KST),
        DayConfig(policy=NewsPolicy(limits=limits)),
        tmp_path,
        _dart(tmp_path),
        clock_holder=clock_holder,
    )
    assert [(d["key"], d["outcome"]) for d in res.decisions] == [("k1", "deferred")]
    assert [r["ticker"] for r in res.ledger if r["event"] == "entry_submitted"] == ["000001"]


def test_replay_day_starts_with_an_empty_market_cache(tmp_path: Path) -> None:
    from daytrade_it.application.signals import news_entry  # noqa: PLC0415

    stale = dt.datetime(2026, 9, 15, 9, 0)
    news_entry._MARKET_CACHE[stale] = (None, None)
    _run_day(tmp_path, *_fresh_broker())
    assert stale not in news_entry._MARKET_CACHE
