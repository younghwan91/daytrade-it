"""Unit tests for KRX tick-size (호가가격단위) rules (Task 9)."""

from decimal import Decimal

import pytest

from daytrade_it.domain.entities import Exchange
from daytrade_it.domain.services.tick_size import KRX_LOT_SIZE, round_to_tick, tick_size_for


class TestTickSizeFor:
    """Boundary tests against KRX's 2023-01-25 unified tick-size table."""

    @pytest.mark.parametrize(
        ("price", "expected_tick"),
        [
            # < 2,000 -> 1
            (Decimal("500"), Decimal("1")),
            (Decimal("1999"), Decimal("1")),
            # 2,000 <= P < 5,000 -> 5
            (Decimal("2000"), Decimal("5")),
            (Decimal("4999"), Decimal("5")),
            # 5,000 <= P < 20,000 -> 10
            (Decimal("5000"), Decimal("10")),
            (Decimal("19999"), Decimal("10")),
            # 20,000 <= P < 50,000 -> 50
            (Decimal("20000"), Decimal("50")),
            (Decimal("49999"), Decimal("50")),
            # 50,000 <= P < 200,000 -> 100
            (Decimal("50000"), Decimal("100")),
            (Decimal("199999"), Decimal("100")),
            # 200,000 <= P < 500,000 -> 500
            (Decimal("200000"), Decimal("500")),
            (Decimal("499999"), Decimal("500")),
            # >= 500,000 -> 1,000
            (Decimal("500000"), Decimal("1000")),
            (Decimal("1000000"), Decimal("1000")),
        ],
    )
    def test_boundaries_kospi(self, price: Decimal, expected_tick: Decimal) -> None:
        assert tick_size_for(price, Exchange.KOSPI) == expected_tick

    @pytest.mark.parametrize(
        ("price", "expected_tick"),
        [
            (Decimal("500"), Decimal("1")),
            (Decimal("100000"), Decimal("100")),
            (Decimal("500000"), Decimal("1000")),
        ],
    )
    def test_same_table_for_kosdaq(self, price: Decimal, expected_tick: Decimal) -> None:
        """Table has been unified across KOSPI/KOSDAQ/KONEX since 2023-01-25."""
        assert tick_size_for(price, Exchange.KOSDAQ) == tick_size_for(price, Exchange.KOSPI)
        assert tick_size_for(price, Exchange.KOSDAQ) == expected_tick

    def test_konex_uses_same_table(self) -> None:
        assert tick_size_for(Decimal("75000"), Exchange.KONEX) == Decimal("100")

    def test_rejects_non_positive_price(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            tick_size_for(Decimal("0"), Exchange.KOSPI)
        with pytest.raises(ValueError, match="positive"):
            tick_size_for(Decimal("-100"), Exchange.KOSPI)

    def test_rejects_unrecognized_market(self) -> None:
        with pytest.raises(ValueError, match="Unrecognized market"):
            tick_size_for(Decimal("1000"), "NASDAQ")  # type: ignore[arg-type]


class TestRoundToTick:
    def test_rounds_down_to_nearest_tick(self) -> None:
        # 73,456 in the [50,000, 200,000) band -> tick 100 -> 73,400
        assert round_to_tick(Decimal("73456"), Exchange.KOSPI) == Decimal("73400")

    def test_already_on_tick_is_unchanged(self) -> None:
        assert round_to_tick(Decimal("73400"), Exchange.KOSPI) == Decimal("73400")

    def test_low_price_band(self) -> None:
        # 1,777 in the [<2,000) band -> tick 1 -> unchanged
        assert round_to_tick(Decimal("1777"), Exchange.KOSDAQ) == Decimal("1777")


def test_krx_lot_size_is_one_share() -> None:
    """KRX board lot for regular equity orders is a single share."""
    assert KRX_LOT_SIZE == 1
