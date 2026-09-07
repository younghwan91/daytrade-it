"""Unit tests for PaperBroker (Task 9: currency/timezone sweep)."""

from decimal import Decimal

from gpt_quant_v2.infrastructure.brokers.base import KST, Order, OrderSide, OrderStatusEnum
from gpt_quant_v2.infrastructure.brokers.paper_broker import PaperBroker, PaperBrokerConfig


def _connected_broker() -> PaperBroker:
    broker = PaperBroker(PaperBrokerConfig(initial_cash=Decimal("10000000")))
    broker.connect()
    return broker


class TestPaperBrokerCurrency:
    def test_account_currency_is_krw(self) -> None:
        broker = _connected_broker()
        account = broker.get_account()
        assert account.currency == "KRW"


class TestPaperBrokerTimezone:
    def test_filled_order_timestamps_are_kst_aware(self) -> None:
        broker = _connected_broker()
        broker.set_price("005930", Decimal("70000"))

        result = broker.place_order(
            Order(ticker="005930", quantity=Decimal("1"), side=OrderSide.BUY)
        )

        assert result.status == OrderStatusEnum.FILLED
        assert result.submitted_at.tzinfo is not None
        assert result.filled_at is not None
        assert result.filled_at.tzinfo is not None
        assert result.submitted_at.astimezone(KST).tzinfo == KST
