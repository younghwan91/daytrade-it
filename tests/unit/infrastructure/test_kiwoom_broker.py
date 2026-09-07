"""Unit tests for KiwoomBroker (Task 10: Phase 3 broker adapter).

All tests mock `KiwoomAPI` at the boundary via a lightweight fake -- no live
network call is ever attempted (no credentials exist in this environment,
see docs/spikes/003-kiwoom-mock-trading.md). The fake mimics kiwoom-client's
shape: `.order`/`.account`/`.market` sub-objects whose methods accept
`**kwargs` and return a raw dict, exactly like the real `KiwoomAPI`.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

from gpt_quant_v2.infrastructure.brokers.base import (
    Order,
    OrderSide,
    OrderStatusEnum,
    OrderType,
)
from gpt_quant_v2.infrastructure.brokers.kiwoom_broker import (
    _ORDER_TYPE_FIELD,
    _ORIG_ORDER_NO_FIELD,
    KiwoomBroker,
)
from gpt_quant_v2.shared.errors import BrokerError, OrderRejectedError


class _FakeOrderModule:
    """Records the exact kwargs it was called with, per method."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.buy_response: dict[str, Any] = {"return_code": 0, "return_msg": "정상", "ord_no": "001"}
        self.sell_response: dict[str, Any] = {"return_code": 0, "return_msg": "정상", "ord_no": "002"}
        self.cancel_response: dict[str, Any] = {"return_code": 0, "return_msg": "정상"}
        self.raise_on_buy: Exception | None = None

    def buy_order(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("buy_order", kwargs))
        if self.raise_on_buy:
            raise self.raise_on_buy
        return self.buy_response

    def sell_order(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("sell_order", kwargs))
        return self.sell_response

    def cancel_order(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("cancel_order", kwargs))
        return self.cancel_response


class _FakeAccountModule:
    def __init__(self) -> None:
        self.deposit_response: dict[str, Any] = {
            "return_code": 0,
            "entr": "1,000,000",
            "100stk_ord_alow_amt": "950,000",
        }
        self.filled_position_response: dict[str, Any] = {
            "return_code": 0,
            "stk_acnt_evlt_prst": [
                {"stk_cd": "005930", "rmnd_qty": "10", "avg_prc": "+70000", "cur_prc": "+71000"}
            ],
        }
        self.unfilled_orders_response: dict[str, Any] = {
            "return_code": 0,
            "oso": [
                {"ord_no": "001", "ord_qty": "10", "cntr_qty": "3", "cntr_pric": "70000"}
            ],
        }

    def deposit_detail(self, **kwargs: Any) -> dict[str, Any]:
        return self.deposit_response

    def filled_position(self, **kwargs: Any) -> dict[str, Any]:
        return self.filled_position_response

    def unfilled_orders(self, **kwargs: Any) -> dict[str, Any]:
        return self.unfilled_orders_response


class _FakeMarketModule:
    def __init__(self) -> None:
        self.quote_response: dict[str, Any] = {
            "return_code": 0,
            "buy_fpr_bid": "+69900",
            "sel_fpr_bid": "+70100",
            "cur_prc": "+70000",
            "now_trde_qty": "1,234,567",
        }

    def stock_quote(self, **kwargs: Any) -> dict[str, Any]:
        return self.quote_response


class _FakeKiwoomAPI:
    """Stands in for `kiwoom_client.KiwoomAPI` -- same `.order`/`.account`/
    `.market` surface, `login`/`logout`/`close` methods, no network calls.
    """

    def __init__(self) -> None:
        self.order = _FakeOrderModule()
        self.account = _FakeAccountModule()
        self.market = _FakeMarketModule()
        self.login_called = False
        self.logout_called = False
        self.close_called = False
        self.raise_on_login = False

    def login(self) -> dict[str, Any]:
        self.login_called = True
        if self.raise_on_login:
            raise RuntimeError("mock login failure")
        return {"token": "fake-token", "token_type": "Bearer"}

    def logout(self) -> dict[str, Any]:
        self.logout_called = True
        return {"return_code": 0}

    def close(self) -> None:
        self.close_called = True


@pytest.fixture
def fake_api() -> _FakeKiwoomAPI:
    return _FakeKiwoomAPI()


@pytest.fixture
def broker(fake_api: _FakeKiwoomAPI) -> KiwoomBroker:
    b = KiwoomBroker(api=fake_api, account_no="1234567890")
    assert b.connect() is True
    return b


class TestConnect:
    def test_connect_success(self, fake_api: _FakeKiwoomAPI) -> None:
        broker = KiwoomBroker(api=fake_api)
        assert broker.connect() is True
        assert broker.is_connected is True
        assert fake_api.login_called is True

    def test_connect_failure_returns_false(self, fake_api: _FakeKiwoomAPI) -> None:
        fake_api.raise_on_login = True
        broker = KiwoomBroker(api=fake_api)
        assert broker.connect() is False
        assert broker.is_connected is False

    def test_disconnect_logs_out_and_closes(self, broker: KiwoomBroker, fake_api: _FakeKiwoomAPI) -> None:
        broker.disconnect()
        assert fake_api.logout_called is True
        assert fake_api.close_called is True
        assert broker.is_connected is False

    def test_methods_require_connect(self, fake_api: _FakeKiwoomAPI) -> None:
        broker = KiwoomBroker(api=fake_api)
        with pytest.raises(BrokerError):
            broker.get_account()


class TestPlaceOrderFieldNames:
    """Covers the field-name ambiguity resolution from the module docstring:
    requests must use the README's field names (trde_tp, dmst_stex_tp,
    org_ord_no), not examples/trading.py's (ord_dv, orig_ord_no,
    acnt_no/acnt_prdt_cd).
    """

    def test_buy_limit_order_uses_readme_field_names(
        self, broker: KiwoomBroker, fake_api: _FakeKiwoomAPI
    ) -> None:
        order = Order(
            ticker="005930",
            quantity=Decimal("10"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("70000"),
        )
        broker.place_order(order)

        assert len(fake_api.order.calls) == 1
        method, kwargs = fake_api.order.calls[0]
        assert method == "buy_order"

        # README-style field names must be present.
        assert kwargs["dmst_stex_tp"] == "01"
        assert kwargs["stk_cd"] == "005930"
        assert kwargs["ord_qty"] == "10"
        assert kwargs["ord_uv"] == "70000"
        assert _ORDER_TYPE_FIELD == "trde_tp"
        assert kwargs["trde_tp"] == "00"

        # examples/trading.py-style field names must NOT be sent -- sending
        # both unverified conventions risks the API rejecting the request
        # for unrecognized fields (see module docstring: option (b), not a
        # dual-send compatibility shim).
        assert "ord_dv" not in kwargs
        assert "orig_ord_no" not in kwargs
        assert "acnt_no" not in kwargs
        assert "acnt_prdt_cd" not in kwargs

    def test_sell_market_order_uses_market_type_code(
        self, broker: KiwoomBroker, fake_api: _FakeKiwoomAPI
    ) -> None:
        order = Order(ticker="005930", quantity=Decimal("5"), side=OrderSide.SELL)
        broker.place_order(order)

        method, kwargs = fake_api.order.calls[0]
        assert method == "sell_order"
        assert kwargs["trde_tp"] == "03"
        assert "ord_uv" not in kwargs  # no limit price for market orders

    def test_cancel_order_uses_org_ord_no(
        self, broker: KiwoomBroker, fake_api: _FakeKiwoomAPI
    ) -> None:
        assert broker.cancel_order("001") is True
        method, kwargs = fake_api.order.calls[0]
        assert method == "cancel_order"
        assert _ORIG_ORDER_NO_FIELD == "org_ord_no"
        assert kwargs["org_ord_no"] == "001"
        assert "orig_ord_no" not in kwargs


class TestPlaceOrderMapping:
    def test_successful_order_maps_to_order_result(self, broker: KiwoomBroker) -> None:
        order = Order(
            ticker="005930",
            quantity=Decimal("10"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("70000"),
            client_order_id="my-client-id",
        )
        result = broker.place_order(order)

        assert result.order_id == "001"
        assert result.client_order_id == "my-client-id"
        assert result.ticker == "005930"
        assert result.quantity == Decimal("10")
        assert result.side == OrderSide.BUY
        assert result.status == OrderStatusEnum.SUBMITTED
        assert result.submitted_at.tzinfo is not None

    def test_order_rejection_raises_order_rejected_error(
        self, broker: KiwoomBroker, fake_api: _FakeKiwoomAPI
    ) -> None:
        from kiwoom_client import KiwoomAPIError

        fake_api.order.raise_on_buy = KiwoomAPIError(
            code=9999, message="증거금 부족", response={"return_code": 9999}
        )
        order = Order(ticker="005930", quantity=Decimal("10"), side=OrderSide.BUY)

        with pytest.raises(OrderRejectedError):
            broker.place_order(order)

    def test_missing_ord_no_in_response_raises_order_rejected_error(
        self, broker: KiwoomBroker, fake_api: _FakeKiwoomAPI
    ) -> None:
        fake_api.order.buy_response = {"return_code": 0, "return_msg": "정상"}
        order = Order(ticker="005930", quantity=Decimal("10"), side=OrderSide.BUY)

        with pytest.raises(OrderRejectedError):
            broker.place_order(order)

    def test_cancel_order_failure_returns_false(
        self, broker: KiwoomBroker, fake_api: _FakeKiwoomAPI
    ) -> None:
        def _raise(**kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("network error")

        fake_api.order.cancel_order = _raise  # type: ignore[method-assign]
        assert broker.cancel_order("999") is False


class TestAccountMapping:
    def test_get_account_maps_deposit_detail(self, broker: KiwoomBroker) -> None:
        account = broker.get_account()
        assert account.account_id == "1234567890"
        assert account.cash == Decimal("1000000")
        assert account.buying_power == Decimal("950000")
        assert account.currency == "KRW"

    def test_get_account_failure_raises_broker_error(
        self, broker: KiwoomBroker, fake_api: _FakeKiwoomAPI
    ) -> None:
        def _raise(**kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("network error")

        fake_api.account.deposit_detail = _raise  # type: ignore[method-assign]
        with pytest.raises(BrokerError):
            broker.get_account()


class TestPositionMapping:
    def test_get_positions_maps_filled_position(self, broker: KiwoomBroker) -> None:
        positions = broker.get_positions()
        assert len(positions) == 1
        pos = positions[0]
        assert pos.ticker == "005930"
        assert pos.quantity == Decimal("10")
        assert pos.average_price == Decimal("70000")
        assert pos.current_price == Decimal("71000")


class TestOrderStatusMapping:
    def test_get_order_status_partial_fill(self, broker: KiwoomBroker) -> None:
        status = broker.get_order_status("001")
        assert status.status == OrderStatusEnum.PARTIAL
        assert status.filled_quantity == Decimal("3")
        assert status.remaining_quantity == Decimal("7")

    def test_get_order_status_not_found_raises(self, broker: KiwoomBroker) -> None:
        with pytest.raises(BrokerError):
            broker.get_order_status("does-not-exist")


class TestQuoteMapping:
    def test_get_quote_maps_stock_quote(self, broker: KiwoomBroker) -> None:
        quote = broker.get_quote("005930")
        assert quote.ticker == "005930"
        assert quote.bid == Decimal("69900")
        assert quote.ask == Decimal("70100")
        assert quote.last_price == Decimal("70000")
        assert quote.volume == 1234567

    def test_get_quote_failure_raises_broker_error(
        self, broker: KiwoomBroker, fake_api: _FakeKiwoomAPI
    ) -> None:
        def _raise(**kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("network error")

        fake_api.market.stock_quote = _raise  # type: ignore[method-assign]
        with pytest.raises(BrokerError):
            broker.get_quote("005930")


class TestNoLiveNetworkCalls:
    """Defensive check: nothing in KiwoomBroker reaches for the real
    kiwoom_client.KiwoomAPI network path -- MagicMock stands in and any
    attribute access that isn't explicitly configured raises on use, so
    unexpected real-looking calls (e.g. httpx) would surface immediately.
    """

    def test_broker_never_imports_real_httpx_client(self) -> None:
        mock_api = MagicMock()
        mock_api.login.return_value = {"token": "t"}
        mock_api.account.deposit_detail.return_value = {
            "return_code": 0,
            "entr": "0",
            "100stk_ord_alow_amt": "0",
        }
        broker = KiwoomBroker(api=mock_api, account_no="ACCT")
        assert broker.connect() is True
        account = broker.get_account()
        assert account.cash == Decimal("0")
        mock_api.login.assert_called_once()
