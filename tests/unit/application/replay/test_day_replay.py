"""replay_day drives the live AutoTrader through a synthetic session."""

import datetime as dt
import sqlite3
from decimal import Decimal
from pathlib import Path

from daytrade_it.application.replay.day_replay import DayConfig, replay_day, snapshot_dart
from daytrade_it.application.replay.events import ReplayEvent
from daytrade_it.application.trading.auto_trader import TradingLimits
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
        limits=TradingLimits(max_open_positions=2, vwap_exit_gap=None, add_on_min_gain=None),
        max_pre_entry_move=Decimal("0.03"),
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
        limits=TradingLimits(max_open_positions=2, vwap_exit_gap=None, add_on_min_gain=None),
        max_pre_entry_move=Decimal("0.03"),
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
