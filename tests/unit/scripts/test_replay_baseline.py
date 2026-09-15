import datetime as dt
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from daytrade_it.infrastructure.brokers.base import MinuteBar

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from replay_baseline import last_regular_close, trade_pnl  # noqa: E402


def test_trade_pnl_nets_costs_per_ticker() -> None:
    fills = [
        {"ticker": "A", "side": "buy", "qty": 1, "price": Decimal(10000)},
        {"ticker": "A", "side": "buy", "qty": 1, "price": Decimal(10100)},
        {"ticker": "A", "side": "sell", "qty": 2, "price": Decimal(10200)},
    ]
    [row] = trade_pnl(fills, positions={}, last_close={})
    assert row["buy_qty"] == 2
    expected = Decimal(20400) * (1 - Decimal("0.00215")) - Decimal(20100) * Decimal("1.00015")
    assert row["net"] == expected


def test_leftover_shares_are_valued_at_the_last_regular_close() -> None:
    fills = [
        {"ticker": "A", "side": "buy", "qty": 1, "price": Decimal(10000)},
        {"ticker": "A", "side": "buy", "qty": 1, "price": Decimal(10100)},
        {"ticker": "A", "side": "sell", "qty": 1, "price": Decimal(10200)},
    ]
    [row] = trade_pnl(fills, positions={"A": 1}, last_close={"A": Decimal(9900)})
    assert row["leftover_qty"] == 1
    sold = Decimal(10200) + Decimal(9900)
    assert row["net"] == sold * (1 - Decimal("0.00215")) - Decimal(20100) * Decimal("1.00015")


def test_unreconciled_quantities_raise() -> None:
    fills = [{"ticker": "A", "side": "buy", "qty": 2, "price": Decimal(10000)}]
    with pytest.raises(ValueError, match="A"):
        trade_pnl(fills, positions={"A": 1}, last_close={"A": Decimal(9900)})


def test_last_regular_close_ignores_after_hours_bars() -> None:
    bars = {
        "A": [
            MinuteBar(ts=dt.datetime(2026, 9, 15, h, m), open=p, high=p, low=p, close=p, volume=1)
            for h, m, p in ((15, 29, Decimal(1)), (15, 30, Decimal(2)), (15, 31, Decimal(3)))
        ]
    }
    assert last_regular_close(bars) == {"A": Decimal(2)}
