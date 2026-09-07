"""KiwoomBroker: `BrokerInterface` adapter backed by `kiwoom_client.KiwoomAPI`.

Phase 3 broker adapter for Korean equities (KRX), replacing the Alpaca/IBKR
stubs from ADR-005 with a Kiwoom Securities REST API adapter -- see
ADR-006 for the full rationale.

kiwoom-client has **no schema/validation layer**: every endpoint method is a
thin ``**kwargs -> dict`` wrapper around a raw JSON POST, and the response is
returned as a plain ``dict[str, Any]`` straight from Kiwoom's servers (see
Task 3's spike, `docs/spikes/003-kiwoom-mock-trading.md`). This module is
where that raw-dict boundary gets turned into typed request/response mapping
-- request builders below construct the exact ``**kwargs`` payload,
and response parsing pulls known-ish fields out defensively (``.get`` with a
default, never a bare ``[...]`` index) since no field is contractually
guaranteed to be present.

FIELD-NAME AMBIGUITY -- READ BEFORE TOUCHING ORDER REQUESTS
=============================================================
kiwoom-client's own README and its ``examples/trading.py`` disagree on the
order-endpoint (``kt10000``-``kt10003``) request field names, and neither is
a verified live-call result:

- ``examples/trading.py`` (stale -- last touched only by an unrelated
  repo-rename commit, 2026-09-05): ``acnt_no``, ``acnt_prdt_cd``, ``stk_cd``,
  ``ord_qty``, ``ord_uv``, ``ord_dv`` (order-type field), ``orig_ord_no``
  (original order number for modify/cancel).
- ``README.md`` (actively maintained -- 3+ commits touching order examples
  in the days leading up to this task, most recently 2026-09-07):
  ``dmst_stex_tp`` (exchange, "01"=KRX), ``stk_cd``, ``ord_qty``, ``trde_tp``
  (order-type field, NOT ``ord_dv``), ``ord_uv``, and ``org_ord_no`` (NOT
  ``orig_ord_no``) for modify/cancel. No ``acnt_no``/``acnt_prdt_cd`` --
  account is presumably implicit from the authenticated app_key/app_secret
  session, not a per-call field.

This was re-verified directly against `kiwoom_client.domestic.order.Order`
source (not just Task 3's spike doc summary) while building this file: the
library itself carries no field-name information at all -- `buy_order`,
`sell_order`, `modify_order`, `cancel_order` are pure `**kwargs -> dict`
pass-throughs (see `Order.buy_order` etc. in kiwoom-client's
`domestic/order.py`). The library's own integration-smoke-test parameter
table (`tests/integration_api_smoke.py`) does not cover the order TR codes
(`kt10000`-`kt10003`) either, so there is no third source to break the tie.
The ambiguity is therefore **unresolved by source** -- exactly as Task 3
found.

DECISION (this task, option (b) from the brief): build requests using the
**README's field names** (`dmst_stex_tp`, `stk_cd`, `ord_qty`, `trde_tp`,
`ord_uv`, `org_ord_no`), because the README is the actively-edited surface
(3 recent commits touching this exact section, vs. the example file only
touched by an unrelated rename) and is what a fresh integrator/maintainer
would trust today. Account identity (`acnt_no`/`acnt_prdt_cd`) is *not* sent
per-order, matching the README's usage.

    TODO(verify-before-live-capital): this field-name choice is NOT verified
    against a live/mock Kiwoom call or official Kiwoom TR documentation --
    no credentials exist in this environment (see
    docs/spikes/003-kiwoom-mock-trading.md). Before `KiwoomBroker` is used
    against any live or mock-mode account with real capital, run an actual
    `buy_order`/`cancel_order` call against `is_mock=True` and confirm the
    request is accepted (`return_code == 0`) with these field names. If it
    is rejected, `_ORDER_TYPE_FIELD`/`_ORIG_ORDER_NO_FIELD` below are the
    two constants to flip to the `examples/trading.py` names.

The two "order type"/"original order number" field names are isolated into
module-level constants below specifically so that a future fix (if the
README turns out wrong) is a one-line change plus a test update, not a
larger refactor.
"""

from __future__ import annotations

import logging
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from gpt_quant_v2.infrastructure.brokers.base import (
    Account,
    BrokerInterface,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderStatusEnum,
    OrderType,
    Position,
    Quote,
    now_kst,
)
from gpt_quant_v2.shared.errors import BrokerError, OrderRejectedError

if TYPE_CHECKING:
    from kiwoom_client import KiwoomAPI

logger = logging.getLogger(__name__)

#: KRX exchange code for `dmst_stex_tp` -- "01" means KRX per the README's
#: own inline comment (거래소 구분, 01: KRX).
_KRX_EXCHANGE_CODE = "01"

#: 지정가 (limit) / 시장가 (market) order-type codes, per the README's
#: `trde_tp="00"` example ("00: 지정가"). "03" for market orders is the
#: commonly documented Kiwoom TR convention; like the field name itself,
#: this specific code value is NOT verified against a live call -- see the
#: module docstring's TODO.
_ORDER_TYPE_LIMIT = "00"
_ORDER_TYPE_MARKET = "03"

#: See module docstring: "order type" field name is ambiguous between
#: kiwoom-client's README (`trde_tp`) and its examples (`ord_dv`). README
#: wins as the actively-maintained source -- flip this constant if a live
#: call proves it wrong.
_ORDER_TYPE_FIELD = "trde_tp"

#: Same ambiguity for "original order number" on modify/cancel: README uses
#: `org_ord_no`, examples/trading.py uses `orig_ord_no`.
_ORIG_ORDER_NO_FIELD = "org_ord_no"

#: return_code values kiwoom-client (via `_check_return_code`) treats as
#: success. Anything else raises `KiwoomAPIError` inside kiwoom-client
#: itself before we ever see the dict, so in practice a dict reaching this
#: module's response mappers already has `return_code == 0` -- kept here
#: only for the rare case a caller passes us a raw dict directly (tests).
_SUCCESS_RETURN_CODE = 0


def _to_decimal(value: Any, default: str = "0") -> Decimal:
    """Best-effort conversion of a raw Kiwoom response field to Decimal.

    Kiwoom's REST responses represent numeric fields as strings, sometimes
    with a leading sign (`"+70000"`) or thousands separators (`"1,234,567"`)
    per the README's explicit note. Never raises -- falls back to `default`
    for missing/unparseable values, since kiwoom-client does no validation
    and a missing field should not crash the whole account/quote mapping.
    """
    if value is None or value == "":
        return Decimal(default)
    if isinstance(value, Decimal):
        return value
    text = str(value).strip().replace(",", "")
    if text.startswith("+"):
        text = text[1:]
    try:
        return Decimal(text)
    except InvalidOperation:
        logger.warning("kiwoom_broker: could not parse numeric field %r, using default", value)
        return Decimal(default)


class KiwoomBroker(BrokerInterface):
    """`BrokerInterface` adapter for Kiwoom Securities' REST API.

    Wraps a `kiwoom_client.KiwoomAPI` instance (sync facade -- this adapter
    implements `BrokerInterface`'s sync methods; an async variant would wrap
    `AsyncKiwoomAPI` instead but is out of scope here). All Kiwoom-specific
    request/response shape handling lives in this class's private helpers so
    that callers only ever see typed `Account`/`Position`/`Order`/etc.
    objects, never raw kiwoom-client dicts.

    Args:
        api: A `KiwoomAPI` instance (or any object exposing the same
            `.order`/`.account`/`.market` surface -- injected so tests can
            supply a mock without touching `kiwoom_client` at all).
        account_no: Kiwoom account number, used only for `Account.account_id`
            in results returned to callers (not sent per-order -- see the
            module docstring's field-name decision).
    """

    def __init__(self, api: KiwoomAPI, account_no: str | None = None) -> None:
        self._api = api
        self._account_no = account_no or "UNKNOWN"
        self._connected = False

    # -- BrokerInterface: connection lifecycle -------------------------------

    def connect(self) -> bool:
        """Log in to Kiwoom (issues an access token) and mark connected."""
        try:
            self._api.login()
        except Exception as exc:  # kiwoom_client raises KiwoomAPIError or httpx errors
            logger.error("kiwoom_broker: connect failed: %s", exc)
            self._connected = False
            return False
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Log out and close the underlying HTTP client."""
        try:
            self._api.logout()
        except Exception as exc:  # noqa: BLE001 - logout is best-effort on disconnect
            logger.warning("kiwoom_broker: logout failed during disconnect: %s", exc)
        finally:
            try:
                self._api.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("kiwoom_broker: close() failed during disconnect: %s", exc)
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # -- BrokerInterface: account / positions --------------------------------

    def get_account(self) -> Account:
        self._ensure_connected()
        try:
            deposit = self._api.account.deposit_detail(qry_tp="3")
        except Exception as exc:
            raise BrokerError(
                f"Kiwoom get_account failed: {exc}", broker_name="kiwoom"
            ) from exc
        return self._map_account(deposit)

    def get_positions(self) -> list[Position]:
        self._ensure_connected()
        try:
            raw = self._api.account.filled_position(dmst_stex_tp="KRX")
        except Exception as exc:
            raise BrokerError(
                f"Kiwoom get_positions failed: {exc}", broker_name="kiwoom"
            ) from exc
        rows = self._extract_rows(raw)
        return [self._map_position(row) for row in rows]

    # -- BrokerInterface: orders ----------------------------------------------

    def place_order(self, order: Order) -> OrderResult:
        self._ensure_connected()
        kwargs = self._build_order_kwargs(order)

        try:
            if order.side == OrderSide.BUY:
                raw = self._api.order.buy_order(**kwargs)
            else:
                raw = self._api.order.sell_order(**kwargs)
        except Exception as exc:
            # kiwoom-client raises KiwoomAPIError for non-zero return_code
            # (rejections included) -- surface that as an OrderRejectedError
            # so callers can distinguish "broker down" from "order rejected".
            raise OrderRejectedError(
                message=f"Kiwoom order rejected: {exc}",
                broker_name="kiwoom",
                symbol=order.ticker,
            ) from exc

        return self._map_order_result(order, raw)

    def cancel_order(self, order_id: str) -> bool:
        self._ensure_connected()
        kwargs = {
            _ORIG_ORDER_NO_FIELD: order_id,
            "dmst_stex_tp": _KRX_EXCHANGE_CODE,
        }
        try:
            raw = self._api.order.cancel_order(**kwargs)
        except Exception as exc:
            logger.error("kiwoom_broker: cancel_order(%s) failed: %s", order_id, exc)
            return False
        return_code = raw.get("return_code", _SUCCESS_RETURN_CODE) if isinstance(raw, dict) else 0
        return return_code == _SUCCESS_RETURN_CODE

    def get_order_status(self, order_id: str) -> OrderStatus:
        self._ensure_connected()
        try:
            raw = self._api.account.unfilled_orders(
                all_stk_tp="0", trde_tp="0", stex_tp="0"
            )
        except Exception as exc:
            raise BrokerError(
                f"Kiwoom get_order_status failed: {exc}", broker_name="kiwoom"
            ) from exc

        for row in self._extract_rows(raw):
            if str(row.get("ord_no", "")) == str(order_id):
                return self._map_order_status(order_id, row)

        raise BrokerError(
            f"Order not found: {order_id}", broker_name="kiwoom"
        )

    def get_quote(self, ticker: str) -> Quote:
        self._ensure_connected()
        try:
            raw = self._api.market.stock_quote(stk_cd=ticker)
        except Exception as exc:
            raise BrokerError(
                f"Kiwoom get_quote failed: {exc}", broker_name="kiwoom"
            ) from exc
        return self._map_quote(ticker, raw)

    # -- Request builders ------------------------------------------------------

    def _build_order_kwargs(self, order: Order) -> dict[str, Any]:
        """Build the `**kwargs` payload for `buy_order`/`sell_order`.

        See the module docstring for the field-name ambiguity this resolves
        (README-style field names, `_ORDER_TYPE_FIELD`/exchange code
        isolated into module constants for easy correction later).
        """
        order_type_code = (
            _ORDER_TYPE_LIMIT if order.order_type == OrderType.LIMIT else _ORDER_TYPE_MARKET
        )
        kwargs: dict[str, Any] = {
            "dmst_stex_tp": _KRX_EXCHANGE_CODE,
            "stk_cd": order.ticker,
            "ord_qty": str(int(order.quantity)),
            _ORDER_TYPE_FIELD: order_type_code,
        }
        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                # Order already validates this in __post_init__; guarded
                # again here since we're about to format it for the wire.
                raise OrderRejectedError(
                    message="limit_price is required for limit orders",
                    broker_name="kiwoom",
                    symbol=order.ticker,
                )
            kwargs["ord_uv"] = str(int(order.limit_price))
        return kwargs

    # -- Response mappers --------------------------------------------------

    def _map_account(self, raw: dict[str, Any]) -> Account:
        """Map a `deposit_detail` (kt00001) response to `Account`.

        Field names below (`entr`/`profa_100tp_ori_ord_alow_amt`-style
        Kiwoom deposit fields) are not documented anywhere in kiwoom-client
        (no captured example response exists per Task 3's spike) -- `.get`
        with fallbacks to `"0"` is deliberate defensiveness, not confidence
        these are the exact right keys. `entr` ("예수금", deposit/cash) and
        `100stk_ord_alow_amt` ("주문가능금액", orderable amount) are the
        Kiwoom TR-doc field names for `kt00001`'s primary cash figures.
        """
        cash = _to_decimal(raw.get("entr"))
        buying_power = _to_decimal(raw.get("100stk_ord_alow_amt", raw.get("ord_alow_amt")))
        if buying_power == 0:
            buying_power = cash
        equity = _to_decimal(raw.get("tot_est_amt", raw.get("entr")))
        if equity == 0:
            equity = cash
        return Account(
            account_id=self._account_no,
            cash=cash,
            buying_power=buying_power,
            equity=equity,
            currency="KRW",
            created_at=now_kst(),
        )

    def _map_position(self, row: dict[str, Any]) -> Position:
        """Map one row of a `filled_position` (kt00005) response to `Position`."""
        quantity = _to_decimal(row.get("rmnd_qty", row.get("qty")))
        avg_price = _to_decimal(row.get("avg_prc", row.get("pur_pric")))
        current_price = _to_decimal(row.get("cur_prc")) if row.get("cur_prc") else None
        return Position(
            ticker=str(row.get("stk_cd", "")),
            quantity=quantity,
            average_price=avg_price,
            current_price=current_price,
        )

    def _map_order_result(self, order: Order, raw: dict[str, Any]) -> OrderResult:
        """Map a `buy_order`/`sell_order` response to `OrderResult`.

        `ord_no` is the one field both of kiwoom-client's disagreeing
        sources (README and examples/trading.py) agree the response
        carries -- see Task 3's spike doc.
        """
        order_id = str(raw.get("ord_no", ""))
        if not order_id:
            raise OrderRejectedError(
                message=f"Kiwoom order response missing ord_no: {raw!r}",
                broker_name="kiwoom",
                symbol=order.ticker,
            )
        return OrderResult(
            order_id=order_id,
            client_order_id=order.client_order_id,
            ticker=order.ticker,
            quantity=order.quantity,
            filled_quantity=Decimal("0"),
            side=order.side,
            order_type=order.order_type,
            status=OrderStatusEnum.SUBMITTED,
            limit_price=order.limit_price,
            submitted_at=now_kst(),
            message=str(raw.get("return_msg", "")) or None,
        )

    def _map_order_status(self, order_id: str, row: dict[str, Any]) -> OrderStatus:
        """Map an `unfilled_orders` (ka10075) row to `OrderStatus`.

        `unfilled_orders` only lists orders that are not yet fully filled;
        an order absent from it is treated as filled or cancelled by
        callers of `get_order_status` -- see `place_order`'s TODO in the
        module docstring about the unverified response schema generally.
        """
        filled_qty = _to_decimal(row.get("cntr_qty", row.get("exec_qty")))
        total_qty = _to_decimal(row.get("ord_qty"))
        remaining = total_qty - filled_qty
        status = OrderStatusEnum.PARTIAL if filled_qty > 0 else OrderStatusEnum.SUBMITTED
        if remaining <= 0 and total_qty > 0:
            status = OrderStatusEnum.FILLED
        return OrderStatus(
            order_id=order_id,
            status=status,
            filled_quantity=filled_qty,
            remaining_quantity=max(remaining, Decimal("0")),
            filled_avg_price=_to_decimal(row.get("cntr_pric")) if row.get("cntr_pric") else None,
            updated_at=now_kst(),
        )

    def _map_quote(self, ticker: str, raw: dict[str, Any]) -> Quote:
        """Map a `stock_quote` (ka10004) response to `Quote`.

        README explicitly documents response values as strings (prices with
        a leading sign, volumes with thousands separators) -- `_to_decimal`
        handles both. `sel_fpr_bid`/`buy_fpr_bid` (best ask/bid) are the
        Kiwoom 호가 TR field-name convention (매도/매수 1차 호가).
        """
        bid = _to_decimal(raw.get("buy_fpr_bid", raw.get("bid")))
        ask = _to_decimal(raw.get("sel_fpr_bid", raw.get("ask")))
        last_price_raw = raw.get("cur_prc")
        last_price = _to_decimal(last_price_raw) if last_price_raw not in (None, "") else None
        volume_raw = raw.get("now_trde_qty", raw.get("trde_qty", "0"))
        try:
            volume = int(_to_decimal(volume_raw))
        except (InvalidOperation, ValueError):
            volume = 0
        return Quote(
            ticker=ticker,
            bid=bid,
            ask=ask,
            last_price=last_price,
            volume=volume,
            timestamp=now_kst(),
        )

    @staticmethod
    def _extract_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
        """Kiwoom list-shaped responses nest rows under a TR-specific key
        (e.g. `stk_acnt_evlt_prst` for filled_position). Since the exact key
        varies per TR and isn't documented for any of the TRs this adapter
        calls, this scans the response for the first list-of-dicts value
        rather than hardcoding a key that might be wrong for a given TR.
        """
        for value in raw.values():
            if isinstance(value, list) and (not value or isinstance(value[0], dict)):
                return value
        return []

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise BrokerError(
                "Kiwoom broker not connected. Call connect() first.", broker_name="kiwoom"
            )
