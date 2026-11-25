"""Keyless demo: how much of a trade does the Korean cost model eat? (no DB, no API key)

Usage: uv run python scripts/demo_cost_wall.py [price]
Costs: model defaults in krx-quant-core (commission + sell tax + slippage, tick-rounded fills).
"""

from __future__ import annotations

import sys
from decimal import Decimal

from daytrade_it.domain.services.cost_model import KoreanCostModel


def main() -> None:
    price = Decimal(sys.argv[1]) if len(sys.argv) > 1 else Decimal("10000")
    model = KoreanCostModel()
    print(f"1 share bought and sold flat at {price} KRW (no price move)")
    for market in ("KOSPI", "KOSDAQ"):
        buy = model.cost_of_trade(price, 1, "buy", market)
        sell = model.cost_of_trade(price, 1, "sell", market)
        paid = buy.fill_price * 1 + buy.commission
        got = sell.fill_price * 1 - sell.commission - sell.tax
        bp = (paid - got) / price * 10000
        print(f"  {market:7s} round-trip cost = {bp:6.1f} bp")


if __name__ == "__main__":
    main()
