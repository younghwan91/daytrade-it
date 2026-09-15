import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from replay_baseline import trade_pnl  # noqa: E402


def test_trade_pnl_nets_costs_per_ticker() -> None:
    fills = [
        {"ticker": "A", "side": "buy", "qty": 1, "price": Decimal(10000)},
        {"ticker": "A", "side": "buy", "qty": 1, "price": Decimal(10100)},
        {"ticker": "A", "side": "sell", "qty": 2, "price": Decimal(10200)},
    ]
    [row] = trade_pnl(fills)
    assert row["buy_qty"] == 2
    expected = Decimal(20400) * (1 - Decimal("0.00215")) - Decimal(20100) * Decimal("1.00015")
    assert row["net"] == expected
