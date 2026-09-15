"""day_net: one policy over one cached day -> net P&L (the optimizer's objective)."""

import datetime as dt
import pickle
import sqlite3
import sys
from decimal import Decimal
from pathlib import Path

from daytrade_it.application.replay.events import ReplayEvent
from daytrade_it.application.replay.prev_close import FrozenPrevCloses
from daytrade_it.application.trading.auto_trader import TradingLimits
from daytrade_it.application.trading.policy import NewsPolicy
from daytrade_it.infrastructure.brokers.base import KST, MinuteBar
from daytrade_it.infrastructure.external.quant_airflow_db_client import StockInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

import replay_common  # noqa: E402
from replay_common import DayData, day_net, load_day  # noqa: E402

DAY = dt.date(2026, 9, 15)
PREV = dt.date(2026, 9, 14)
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


def _bars() -> list[MinuteBar]:
    """10,100 flat, a 10:00-10:59 dip to 9,900, then back to 10,100."""
    out, t = [], dt.datetime(2026, 9, 15, 9, 0)
    while t.time() <= dt.time(15, 30):
        p = Decimal(9900) if dt.time(10, 0) <= t.time() < dt.time(11, 0) else Decimal(10100)
        out.append(MinuteBar(ts=t, open=p, high=p + 10, low=p - 10, close=p, volume=1000))
        t += dt.timedelta(minutes=1)
    return out


def _event() -> ReplayEvent:
    at = dt.datetime(2026, 9, 15, 9, 0, 35, tzinfo=KST)
    return ReplayEvent(
        key="k1",
        article_id="0a0a0a0a-0000-5000-8000-000000000001",
        ticker="000001",
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


def _data() -> DayData:
    return DayData(
        day=DAY,
        events=[_event()],
        bars={"000001": _bars()},
        frozen=FrozenPrevCloses(
            day=DAY,
            closes={"000001": (Decimal(10000), PREV)},
            krx={"000001": (Decimal(10000), PREV)},
        ),
        stale_before=dt.datetime(2026, 9, 14, 15, 20, tzinfo=KST),
    )


def _dart(tmp: Path) -> Path:
    p = tmp / "dart.db"
    con = sqlite3.connect(p)
    con.execute(
        "CREATE TABLE disclosures (rcept_no TEXT PRIMARY KEY, rcept_dt TEXT NOT NULL, "
        "code TEXT NOT NULL, corp_name TEXT, corp_cls TEXT, report_nm TEXT, "
        "material_type TEXT, is_correction INTEGER)"
    )
    con.execute("INSERT INTO disclosures VALUES ('1','20260916','000002','x','Y','x','x',0)")
    con.commit()
    con.close()
    return p


STOCKS = {"000001": StockInfo(code="000001", name="n", market="KOSPI", sector="s")}
BASE = TradingLimits(vwap_exit_gap=None, add_on_min_gain=None)


def test_day_net_differs_by_policy_and_is_deterministic(tmp_path: Path) -> None:
    data, dart, work = _data(), _dart(tmp_path), tmp_path / "work"
    default = NewsPolicy(limits=BASE)
    stop = NewsPolicy(limits=TradingLimits(**{**BASE.__dict__, "stop_loss_pct": Decimal("0.01")}))

    a = day_net(data, default, STOCKS, dart, work)
    b = day_net(data, default, STOCKS, dart, work)
    s = day_net(data, stop, STOCKS, dart, work)

    assert set(a) == {"day", "net", "trades", "losers", "worst_trade", "errors", "leftovers"}
    assert a == b
    assert a["day"] == DAY and a["trades"] == 1 and a["errors"] == 0 and a["leftovers"] == {}
    assert s["trades"] == 1 and s["errors"] == 0
    assert s["net"] < a["net"]  # stopped out in the dip instead of flattening at 10,100
    assert s["worst_trade"] == s["net"] and s["losers"] == 1


def test_load_day_builds_once_then_reads_the_pickle(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    calls: list[dt.date] = []
    data = _data()

    def fake_build(day: dt.date) -> DayData:
        calls.append(day)
        return data

    monkeypatch.setattr(replay_common, "_build_day", fake_build)
    first = load_day(DAY, cache_dir=tmp_path)
    second = load_day(DAY, cache_dir=tmp_path)
    assert calls == [DAY]
    assert second == first
    assert pickle.loads((tmp_path / f"{DAY}.pkl").read_bytes()) == data
