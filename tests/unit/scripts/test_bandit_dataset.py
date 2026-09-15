"""bandit_dataset: counterfactual rewards + decision-time features from an uncapped replay."""

import dataclasses
import datetime as dt
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

import bandit_dataset as bd  # noqa: E402
from replay_common import DayData, run_day  # noqa: E402

from daytrade_it.application.replay.events import ReplayEvent  # noqa: E402
from daytrade_it.application.replay.prev_close import FrozenPrevCloses  # noqa: E402
from daytrade_it.application.trading.auto_trader import TradingLimits  # noqa: E402
from daytrade_it.application.trading.bandit import FEATURES  # noqa: E402
from daytrade_it.application.trading.policy import NewsPolicy  # noqa: E402
from daytrade_it.infrastructure.brokers.base import KST, MinuteBar  # noqa: E402
from daytrade_it.infrastructure.external.quant_airflow_db_client import StockInfo  # noqa: E402
from tests.unit.scripts.test_replay_common import PREV, STOCKS, V2, _bars, _dart  # noqa: E402

DAY = dt.date(2026, 9, 15)


def _flat(price: int) -> list[MinuteBar]:
    out, t = [], dt.datetime(2026, 9, 15, 9, 0)
    while t.time() <= dt.time(15, 30):
        p = Decimal(price)
        out.append(MinuteBar(ts=t, open=p, high=p, low=p, close=p, volume=1000))
        t += dt.timedelta(minutes=1)
    return out


def _event(n: str, ticker: str, hms: str) -> ReplayEvent:
    h, m, s = map(int, hms.split(":"))
    at = dt.datetime(2026, 9, 15, h, m, s, tzinfo=KST)
    return ReplayEvent(
        key=f"k{n}",
        article_id=f"0a0a0a0a-0000-5000-8000-00000000000{n}",
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


def _day() -> DayData:
    closes = {
        "000001": (Decimal(10000), PREV),
        "000002": (Decimal(10000), PREV),
        "069500": (Decimal(30000), PREV),
    }
    return DayData(
        day=DAY,
        events=[
            _event("1", "000001", "09:00:35"),
            _event("2", "000002", "09:00:35"),
            _event("3", "000001", "10:30:10"),  # same ticker again: already entered
        ],
        bars={"000001": _bars(), "000002": _flat(10100), "069500": _flat(30300)},
        frozen=FrozenPrevCloses(day=DAY, closes=closes, krx=closes),
        stale_before=dt.datetime(2026, 9, 14, 15, 20, tzinfo=KST),
    )


STOCKS2 = {**STOCKS, "000002": StockInfo(code="000002", name="m", market="KOSPI", sector="s")}
BASE = NewsPolicy(
    limits=TradingLimits(max_open_positions=1, vwap_exit_gap=None, add_on_min_gain=None)
)


def test_uncapped_lifts_caps_and_turns_the_bandit_off() -> None:
    base = dataclasses.replace(
        BASE,
        selection=dataclasses.replace(BASE.selection, bandit_model="m.json", rank_key="bandit"),
    )
    u = bd.uncapped(base)
    assert u.limits.max_open_positions == 1000 and u.limits.max_entries_per_day == 1000
    assert u.limits.max_trader_exposure == Decimal(10**12)
    assert u.selection.bandit_model is None and u.selection.rank_key == "surprise"
    assert u.limits.stop_loss_pct == base.limits.stop_loss_pct
    assert dataclasses.replace(u.selection, bandit_model="m.json", rank_key="bandit") == (
        base.selection
    )


def test_day_rows_rewards_and_features(tmp_path: Path) -> None:
    data, dart = _day(), _dart(tmp_path)
    rows = bd.day_rows(data, BASE, STOCKS2, dart, tmp_path / "work")

    assert [(r["key"], r["status"]) for r in rows] == [
        ("k1", "filled"),
        ("k2", "filled"),
        ("k3", "skipped:already entered today"),
    ]
    _, pnl = run_day(data, bd.uncapped(BASE), STOCKS2, dart, tmp_path / "w2", cash=bd.UNCAPPED_CASH)
    by = {r["ticker"]: r for r in pnl}
    for r in rows[:2]:
        p = by[r["ticker"]]
        assert r["reward"] == pytest.approx(float(p["net"] / p["buy_value"]))
        assert set(r["features"]) == set(FEATURES)
        assert r["features"]["mkt_move"] == pytest.approx(0.01)
        assert r["features"]["pre_entry_move"] == pytest.approx(0.01)
        assert r["day"] == "2026-09-15"
    assert rows[1]["reward"] < 0  # 000002 flat at 10,100: costs and the exit tick only
    assert rows[2]["reward"] is None
    assert rows[2]["features"]["ret_5m"] == pytest.approx(0.0)  # 10:25-10:29 closes flat
    assert json.loads(json.dumps(rows, default=str))  # serialisable
    assert not list((tmp_path / "work").glob(f"{DAY}-*"))


def test_summary_counts_trainable_rows_and_constant_features() -> None:
    feats = dict.fromkeys(FEATURES, 0.0)
    rows = [
        {"day": "d1", "reward": 0.01, "status": "filled", "features": {**feats, "score": 1.0}},
        {"day": "d1", "reward": -0.01, "status": "filled", "features": {**feats, "score": 2.0}},
        {"day": "d1", "reward": None, "status": "skipped:x", "features": feats},
        {"day": "d1", "reward": 0.0, "status": "market_missing", "features": None},
    ]
    (s,) = bd.summarize(rows)
    assert s["rows"] == 4 and s["trainable"] == 2
    assert s["reward_mean"] == pytest.approx(0.0) and s["reward_std"] == pytest.approx(0.01)
    assert s["constant_features"] == pytest.approx((len(FEATURES) - 1) / len(FEATURES))
    assert bd.trainable(rows) == rows[:2]
