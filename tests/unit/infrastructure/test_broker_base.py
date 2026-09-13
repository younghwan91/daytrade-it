"""Unit tests for broker base data models (Task 9: currency/timezone/tick-size)."""

from datetime import datetime
from decimal import Decimal

import pytest

from daytrade_it.infrastructure.brokers.base import (
    KST,
    Account,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderStatusEnum,
    OrderType,
    Quote,
)


class TestAccountCurrencyDefault:
    def test_default_currency_is_krw(self) -> None:
        account = Account(
            account_id="A1",
            cash=Decimal("1000000"),
            buying_power=Decimal("1000000"),
            equity=Decimal("1000000"),
        )
        assert account.currency == "KRW"

    def test_created_at_defaults_to_kst_aware(self) -> None:
        account = Account(
            account_id="A1",
            cash=Decimal("1000000"),
            buying_power=Decimal("1000000"),
            equity=Decimal("1000000"),
        )
        assert account.created_at.tzinfo is not None
        assert account.created_at.astimezone(KST).tzinfo == KST


class TestOrderLotAndTickSize:
    def test_defaults(self) -> None:
        order = Order(ticker="005930", quantity=Decimal("10"), side=OrderSide.BUY)
        assert order.lot_size == 1
        assert order.tick_size is None

    def test_explicit_tick_size_coerced_to_decimal(self) -> None:
        order = Order(
            ticker="005930",
            quantity=Decimal("10"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("70000"),
            tick_size="100",  # type: ignore[arg-type]
        )
        assert order.tick_size == Decimal("100")

    def test_rejects_non_positive_lot_size(self) -> None:
        with pytest.raises(ValueError, match="lot_size"):
            Order(ticker="005930", quantity=Decimal("10"), side=OrderSide.BUY, lot_size=0)


class TestQuoteLotAndTickSize:
    def test_defaults(self) -> None:
        quote = Quote(ticker="005930", bid=Decimal("69900"), ask=Decimal("70000"))
        assert quote.lot_size == 1
        assert quote.tick_size is None

    def test_timestamp_defaults_to_kst_aware(self) -> None:
        quote = Quote(ticker="005930", bid=Decimal("69900"), ask=Decimal("70000"))
        assert quote.timestamp.tzinfo is not None
        assert quote.timestamp.astimezone(KST).tzinfo == KST


def test_order_result_submitted_at_defaults_to_kst_aware() -> None:
    result = OrderResult(
        order_id="O1",
        client_order_id="C1",
        ticker="005930",
        quantity=Decimal("10"),
        filled_quantity=Decimal("10"),
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        status=OrderStatusEnum.FILLED,
    )
    assert result.submitted_at.tzinfo is not None


def test_order_status_updated_at_defaults_to_kst_aware() -> None:
    status = OrderStatus(
        order_id="O1",
        status=OrderStatusEnum.FILLED,
        filled_quantity=Decimal("10"),
        remaining_quantity=Decimal("0"),
    )
    assert status.updated_at.tzinfo is not None


def test_kst_is_asia_seoul() -> None:
    now_kst = datetime.now(KST)
    assert now_kst.tzinfo is not None
    # UTC offset for Asia/Seoul is a fixed +9:00 (no DST observed).
    assert now_kst.utcoffset().total_seconds() == 9 * 3600


def test_quote_mid_price_two_sided_book() -> None:
    quote = Quote(ticker="005930", bid=Decimal("70000"), ask=Decimal("70100"))
    assert quote.mid_price == Decimal("70050")


def test_quote_mid_price_upper_limit_empty_ask_uses_bid() -> None:
    """A stock locked at 상한가 has no sellers (Kiwoom reports ask=0).
    Averaging with the empty side used to halve the price, making a
    limit-up stock read as a -50% crash (seen live 2026-09-13, 000922)."""
    quote = Quote(ticker="000922", bid=Decimal("15130"), ask=Decimal("0"))
    assert quote.mid_price == Decimal("15130")


def test_quote_mid_price_lower_limit_empty_bid_uses_ask() -> None:
    quote = Quote(ticker="005930", bid=Decimal("0"), ask=Decimal("49000"))
    assert quote.mid_price == Decimal("49000")


def test_quote_mid_price_empty_book_is_zero() -> None:
    quote = Quote(ticker="005930", bid=Decimal("0"), ask=Decimal("0"))
    assert quote.mid_price == Decimal("0")
