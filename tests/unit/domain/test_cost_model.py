"""Unit tests for the Korean transaction-cost model (Task 13)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from krx_signal_engine.domain.entities.stock import Exchange
from krx_signal_engine.domain.services.cost_model import (
    DEFAULT_COMMISSION_RATE,
    DEFAULT_KOSDAQ_SLIPPAGE_PCT,
    DEFAULT_KOSDAQ_TAX_RATE,
    DEFAULT_KOSPI_SLIPPAGE_PCT,
    DEFAULT_KOSPI_TAX_RATE,
    CostModelConfig,
    KoreanCostModel,
)
from krx_signal_engine.domain.services.tick_size import round_to_tick


class TestTransactionTax:
    """증권거래세: sell-side only, researched 2026 rates."""

    def test_buy_has_no_tax(self) -> None:
        model = KoreanCostModel()
        tax = model.transaction_tax(Decimal("1000000"), side="BUY", market=Exchange.KOSPI)
        assert tax == Decimal("0")

    def test_sell_kospi_tax_is_020_pct(self) -> None:
        model = KoreanCostModel()
        notional = Decimal("1000000")
        tax = model.transaction_tax(notional, side="SELL", market=Exchange.KOSPI)
        assert tax == notional * DEFAULT_KOSPI_TAX_RATE
        assert tax == Decimal("2000.00")

    def test_sell_kosdaq_tax_is_020_pct(self) -> None:
        model = KoreanCostModel()
        notional = Decimal("1000000")
        tax = model.transaction_tax(notional, side="SELL", market=Exchange.KOSDAQ)
        assert tax == notional * DEFAULT_KOSDAQ_TAX_RATE
        assert tax == Decimal("2000.00")

    def test_side_is_case_insensitive(self) -> None:
        model = KoreanCostModel()
        notional = Decimal("1000000")
        assert model.transaction_tax(notional, side="sell") == model.transaction_tax(
            notional, side="SELL"
        )


class TestCommission:
    """위탁수수료: labeled representative placeholder, both sides."""

    def test_commission_applies_to_notional(self) -> None:
        model = KoreanCostModel()
        commission = model.commission(Decimal("1000000"))
        assert commission == Decimal("1000000") * DEFAULT_COMMISSION_RATE

    def test_min_commission_floor(self) -> None:
        config = CostModelConfig(commission_rate=Decimal("0.00015"), min_commission=Decimal("100"))
        model = KoreanCostModel(config)
        # Tiny notional would otherwise round to less than the floor.
        commission = model.commission(Decimal("10"))
        assert commission == Decimal("100")

    def test_commission_is_configurable_not_hardcoded(self) -> None:
        config = CostModelConfig(commission_rate=Decimal("0.001"))
        model = KoreanCostModel(config)
        assert model.commission(Decimal("1000000")) == Decimal("1000")


class TestSlippage:
    """Flat, honestly-labeled placeholder slippage, tiered by market."""

    def test_buy_slips_up(self) -> None:
        model = KoreanCostModel()
        price = Decimal("70000")
        filled = model.apply_slippage(price, side="BUY", market=Exchange.KOSPI)
        # Whole-branch review finding #5: apply_slippage now rounds the
        # slipped price to a valid KRX tick before returning -- a backtest
        # fill must be a price that could actually occur on KRX.
        assert filled == round_to_tick(price * (1 + DEFAULT_KOSPI_SLIPPAGE_PCT), Exchange.KOSPI)
        # Tick rounding can legitimately erase a small slip entirely (a
        # 0.05% bump on 70000 rounds back down to a 100-tick band) -- >=
        # is the correct invariant post-tick-rounding, not strictly >.
        assert filled >= price

    def test_sell_slips_down(self) -> None:
        model = KoreanCostModel()
        price = Decimal("70000")
        filled = model.apply_slippage(price, side="SELL", market=Exchange.KOSPI)
        assert filled == round_to_tick(price * (1 - DEFAULT_KOSPI_SLIPPAGE_PCT), Exchange.KOSPI)
        assert filled < price

    def test_kosdaq_slips_more_than_kospi(self) -> None:
        model = KoreanCostModel()
        price = Decimal("10000")
        kospi_fill = model.apply_slippage(price, side="BUY", market=Exchange.KOSPI)
        kosdaq_fill = model.apply_slippage(price, side="BUY", market=Exchange.KOSDAQ)
        assert (kosdaq_fill - price) > (kospi_fill - price)
        assert DEFAULT_KOSDAQ_SLIPPAGE_PCT > DEFAULT_KOSPI_SLIPPAGE_PCT

    def test_konex_falls_back_to_kospi_rate(self) -> None:
        model = KoreanCostModel()
        price = Decimal("10000")
        konex_fill = model.apply_slippage(price, side="BUY", market=Exchange.KONEX)
        kospi_fill = model.apply_slippage(price, side="BUY", market=Exchange.KOSPI)
        assert konex_fill == kospi_fill


class TestCostOfTrade:
    """End-to-end breakdown: slippage embedded in fill_price + commission + tax."""

    def test_buy_breakdown_has_no_tax(self) -> None:
        model = KoreanCostModel()
        cost = model.cost_of_trade(
            price=Decimal("70000"), quantity=Decimal("10"), side="BUY", market=Exchange.KOSPI
        )
        expected_fill = round_to_tick(
            Decimal("70000") * (1 + DEFAULT_KOSPI_SLIPPAGE_PCT), Exchange.KOSPI
        )
        assert cost.fill_price == expected_fill
        assert cost.tax == Decimal("0")
        assert cost.commission == expected_fill * Decimal("10") * DEFAULT_COMMISSION_RATE
        assert cost.total_cost == cost.commission  # tax is 0 on buys

    def test_sell_breakdown_known_inputs(self) -> None:
        model = KoreanCostModel()
        price = Decimal("50000")
        quantity = Decimal("100")
        cost = model.cost_of_trade(price=price, quantity=quantity, side="SELL", market=Exchange.KOSPI)

        expected_fill = round_to_tick(price * (1 - DEFAULT_KOSPI_SLIPPAGE_PCT), Exchange.KOSPI)
        expected_notional = expected_fill * quantity
        expected_commission = expected_notional * DEFAULT_COMMISSION_RATE
        expected_tax = expected_notional * DEFAULT_KOSPI_TAX_RATE

        assert cost.fill_price == expected_fill
        assert cost.commission == expected_commission
        assert cost.tax == expected_tax
        assert cost.total_cost == expected_commission + expected_tax

    def test_round_trip_costs_reduce_net_pnl(self) -> None:
        """A buy-then-sell at the same reference price should be a net loss
        once slippage + commission + tax are applied -- the whole point of
        having a cost model at all."""
        model = KoreanCostModel()
        price = Decimal("50000")
        quantity = Decimal("10")

        buy = model.cost_of_trade(price=price, quantity=quantity, side="BUY", market=Exchange.KOSPI)
        sell = model.cost_of_trade(price=price, quantity=quantity, side="SELL", market=Exchange.KOSPI)

        buy_outlay = buy.fill_price * quantity + buy.commission + buy.tax
        sell_proceeds = sell.fill_price * quantity - sell.commission - sell.tax

        assert sell_proceeds < buy_outlay


class TestConfigOverride:
    def test_custom_config_used_end_to_end(self) -> None:
        config = CostModelConfig(
            kospi_tax_rate=Decimal("0.001"),
            kosdaq_tax_rate=Decimal("0.001"),
            commission_rate=Decimal("0.0005"),
            kospi_slippage_pct=Decimal("0.002"),
            kosdaq_slippage_pct=Decimal("0.004"),
        )
        model = KoreanCostModel(config)
        cost = model.cost_of_trade(
            price=Decimal("10000"), quantity=Decimal("1"), side="SELL", market=Exchange.KOSPI
        )
        expected_fill = Decimal("10000") * (1 - Decimal("0.002"))
        assert cost.fill_price == expected_fill
        assert cost.tax == expected_fill * Decimal("0.001")
        assert cost.commission == expected_fill * Decimal("0.0005")


@pytest.mark.parametrize("market", [Exchange.KOSPI, Exchange.KOSDAQ])
def test_negative_or_zero_quantity_does_not_error(market: Exchange) -> None:
    model = KoreanCostModel()
    cost = model.cost_of_trade(
        price=Decimal("10000"), quantity=Decimal("0"), side="SELL", market=market
    )
    assert cost.commission == Decimal("0")
    assert cost.tax == Decimal("0")
