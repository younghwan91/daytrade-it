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

RESOLVED (2026-09-13): the field *names* were right, but two field *values*
were wrong guesses. The sibling repo `scalp-it` places real live limit
orders through this exact TR family (`kt10000`/`kt10001`/`kt10003`) on a
production Kiwoom account since 2026-08-28 -- `scalp-it`'s
`data/pair_orders.log` has 22 live submissions, all `return_code == 0` with
an assigned `ord_no`. Its request bodies (`adapters/kiwoom_order.py`)
confirm:

- `dmst_stex_tp`: the literal string `"KRX"`, not `"01"` -- this module's
  own `get_positions` (line ~220, `filled_position(dmst_stex_tp="KRX")`)
  already used the correct value; only the order-request path had the wrong
  one.
- `trde_tp` (order-type) for a limit order: `"0"`, not `"00"`.
- `stk_cd`, `ord_qty`, `ord_uv` field names match exactly for buy/sell.

CORRECTION (2026-11-14): "22 live submissions, all `return_code == 0`" above
is wrong. Only 8 (1-share buy 4, sell 4) were `return_code == 0`; all 14
cancels (`kt10003`, body `org_ord_no` + `ord_qty`) were `return_code == 2`.
Kiwoom's official spec JSON names the cancel fields `orig_ord_no` and
`cncl_qty` -- see `_ORIG_ORDER_NO_FIELD`.

`_ORDER_TYPE_MARKET` ("03") remains **unverified** -- `scalp-it` places
limit orders only and has never sent a market order through this endpoint.
Confirm it against a live/mock call before relying on it.

The order type / exchange code constants stay isolated at module level
below so a future correction (if `_ORDER_TYPE_MARKET` turns out wrong too)
is a one-line change plus a test update, not a larger refactor.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from krx_quant_core.execution.kiwoom_spec import (
    BALANCE_ROWS_KEY,
    DEFAULT_EXCHANGE,
    TRDE_TP_LIMIT,
    TRDE_TP_MARKET,
    UNFILLED_ROWS_KEY,
)
from krx_quant_core.market.codes import normalize_code

from daytrade_it.domain.entities.stock import Exchange
from daytrade_it.domain.services.tick_size import round_to_tick
from daytrade_it.infrastructure.brokers.base import (
    Account,
    BrokerInterface,
    MinuteBar,
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
from daytrade_it.shared.errors import BrokerError, OrderRejectedError

if TYPE_CHECKING:
    from kiwoom_client import KiwoomAPI

logger = logging.getLogger(__name__)

#: Kiwoom REST protocol values shared with scalp-it live in
#: `krx_quant_core.execution.kiwoom_spec`, which records how each was
#: confirmed. `dmst_stex_tp="KRX"` (not "01") and limit `trde_tp="0"` (not
#: "00") were verified 2026-09-13 against scalp-it's 22 live order fills
#: (see module docstring). The market order code "03" remains UNVERIFIED
#: (`TRDE_TP_MARKET_VERIFIED is False`) -- `execute_trade` refuses MARKET
#: orders for that reason.
_KRX_EXCHANGE_CODE = DEFAULT_EXCHANGE
_ORDER_TYPE_LIMIT = TRDE_TP_LIMIT
_ORDER_TYPE_MARKET = TRDE_TP_MARKET

#: See module docstring: "order type" field name is ambiguous between
#: kiwoom-client's README (`trde_tp`) and its examples (`ord_dv`). README
#: wins as the actively-maintained source -- flip this constant if a live
#: call proves it wrong.
_ORDER_TYPE_FIELD = "trde_tp"

#: Cancel (`kt10003`) body field names, from Kiwoom's official spec JSON
#: (Kiwoom-Securities/Kiwoom-REST-API `kiwoom/_data/kiwoom_api_spec.json`,
#: c7a3d25, checked 2026-11-14): `dmst_stex_tp`, `orig_ord_no`, `stk_cd`,
#: `cncl_qty` ("0" = cancel all remaining). The old guess `org_ord_no` +
#: `ord_qty` (from kiwoom-client's README) got `return_code == 2` on all 14
#: live cancels in scalp-it's `data/pair_orders.log` -- no live cancel has
#: succeeded yet, so these names are still UNVERIFIED by a live call
#: (scalp-it `scripts/smoke_cancel.py`).
_ORIG_ORDER_NO_FIELD = "orig_ord_no"
_CANCEL_QTY_FIELD = "cncl_qty"

#: return_code values kiwoom-client (via `_check_return_code`) treats as
#: success. Anything else raises `KiwoomAPIError` inside kiwoom-client
#: itself before we ever see the dict, so in practice a dict reaching this
#: module's response mappers already has `return_code == 0` -- kept here
#: only for the rare case a caller passes us a raw dict directly (tests).
_SUCCESS_RETURN_CODE = 0

#: Rows keys (kt00018 / ka10075), as read on the live account by scalp-it
#: and confirmed present in real 2026-09-13 responses.
_BALANCE_ROWS_KEY = BALANCE_ROWS_KEY
_UNFILLED_ROWS_KEY = UNFILLED_ROWS_KEY

#: `normalize_code` ("A041830" -> "041830", keeps mid-code letters like
#: "0155E0") is `krx_quant_core.market.codes.normalize_code`, imported above
#: and re-exported from this module.


def _to_decimal(value: Any, default: str = "0") -> Decimal:
    """Best-effort conversion of a raw Kiwoom response field to Decimal.

    Kiwoom's REST responses represent numeric fields as strings, sometimes
    with a leading `+`/`-` sign or thousands separators (`"1,234,567"`) per
    the README's explicit note. **The sign encodes direction vs. the
    reference/previous-close price (등락 방향), not an actual negative
    price or quantity** -- verified live 2026-09-13 against a real
    `get_quote("005930")` call, which returned `buy_fpr_bid`/`sel_fpr_bid`
    both negative-prefixed for a price below the reference point (a real
    KRX price/quantity is never negative). `scalp-it`'s
    `adapters/kiwoom.py.strip_sign()` handles this the same way
    (`.lstrip("+-")`) across every signed field it reads. Never raises --
    falls back to `default` for missing/unparseable values, since
    kiwoom-client does no validation and a missing field should not crash
    the whole account/quote mapping.
    """
    if value is None or value == "":
        return Decimal(default)
    if isinstance(value, Decimal):
        return value
    text = str(value).strip().replace(",", "").lstrip("+-")
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
        """Close the HTTP client. Deliberately does NOT log out (revoke the token).

        Kiwoom hands every process the *same* access token for an appkey
        while it is valid (verified 2026-09-13: issuing twice returned an
        identical token, and the first kept working). scalp-it trades live
        on the same appkey, and it only refreshes on HTTP 401 while Kiwoom
        reports a bad token as HTTP 200 + 8005, with no retry at all on
        orders. Revoking here -- at preopen check exit, every crash-restart,
        and the 15:20 stop -- would kill scalp-it's live session. The token
        simply expires on its own.
        """
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
            raise BrokerError(f"Kiwoom get_account failed: {exc}", broker_name="kiwoom") from exc
        return self._map_account(deposit)

    def get_positions(self) -> list[Position]:
        """Holdings via `kt00018` (계좌평가잔고내역).

        Same TR, parameters, rows key and fields scalp-it reads holdings with
        on the live account (`adapters/kiwoom_order.py::holdings`). This used
        `kt00005` (체결잔고), which was never verified and whose rows live
        under a different key.
        """
        self._ensure_connected()
        try:
            raw = self._api.account.evaluation_balance_detail(qry_tp="1", dmst_stex_tp="KRX")
        except Exception as exc:
            raise BrokerError(f"Kiwoom get_positions failed: {exc}", broker_name="kiwoom") from exc
        positions = [self._map_position(row) for row in raw.get(_BALANCE_ROWS_KEY) or []]
        return [p for p in positions if p.ticker and p.quantity > 0]

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
        """Cancel an open order.

        Kiwoom's cancel TR (`kt10003`) body is `dmst_stex_tp`, `orig_ord_no`,
        `stk_cd`, `cncl_qty` (official spec JSON; see `_ORIG_ORDER_NO_FIELD`).
        Not yet confirmed by a live cancel -- every live cancel so far used the
        old `org_ord_no`/`ord_qty` body and got `return_code == 2`.
        `BrokerInterface.cancel_order` only takes `order_id`, so `stk_cd` and the
        remaining quantity -- the two
        extra fields are looked up here via `unfilled_orders` (the same TR
        `get_order_status` already uses) rather than widening the interface.
        """
        self._ensure_connected()
        try:
            unfilled = self._api.account.unfilled_orders(all_stk_tp="0", trde_tp="0", stex_tp="0")
        except Exception as exc:
            logger.error("kiwoom_broker: cancel_order(%s) lookup failed: %s", order_id, exc)
            return False

        row = self._find_unfilled(unfilled, order_id)
        if row is None:
            logger.error(
                "kiwoom_broker: cancel_order(%s): order not found in unfilled_orders", order_id
            )
            return False

        kwargs = {
            "dmst_stex_tp": _KRX_EXCHANGE_CODE,
            _ORIG_ORDER_NO_FIELD: order_id,
            "stk_cd": normalize_code(row.get("stk_cd")),
            _CANCEL_QTY_FIELD: str(int(self._map_order_status(order_id, row).remaining_quantity)),
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
            raw = self._api.account.unfilled_orders(all_stk_tp="0", trde_tp="0", stex_tp="0")
        except Exception as exc:
            raise BrokerError(
                f"Kiwoom get_order_status failed: {exc}", broker_name="kiwoom"
            ) from exc

        row = self._find_unfilled(raw, order_id)
        if row is not None:
            return self._map_order_status(order_id, row)

        raise BrokerError(f"Order not found: {order_id}", broker_name="kiwoom")

    @staticmethod
    def _find_unfilled(raw: dict[str, Any], order_id: str) -> dict[str, Any] | None:
        want = str(order_id).strip().lstrip("0")
        for row in raw.get(_UNFILLED_ROWS_KEY) or []:
            if str(row.get("ord_no", "")).strip().lstrip("0") == want:
                return row
        return None

    def get_quote(self, ticker: str) -> Quote:
        self._ensure_connected()
        try:
            raw = self._api.market.stock_quote(stk_cd=ticker)
        except Exception as exc:
            raise BrokerError(f"Kiwoom get_quote failed: {exc}", broker_name="kiwoom") from exc
        return self._map_quote(ticker, raw)

    def get_minute_bars(self, ticker: str, day: date | None = None) -> list[MinuteBar]:
        """ka10080 1-minute bars for `day` (default today, KST), oldest first.

        One page is ~900 bars (≈2.3 sessions), newest first, so the first page
        always covers the current session. Fields and sign handling follow
        scalp-it `adapters/kiwoom.py` (verified live): prices carry a
        direction sign that is not a negative price; `cntr_tm` is
        YYYYMMDDHHMMSS KST.
        """
        self._ensure_connected()
        day = day or now_kst().date()
        try:
            raw = self._api.chart.stock_minute_chart(stk_cd=ticker, tic_scope="1", upd_stkpc_tp="1")
        except Exception as exc:
            raise BrokerError(f"Kiwoom minute bars failed: {exc}", broker_name="kiwoom") from exc
        bars = []
        for row in raw.get("stk_min_pole_chart_qry") or []:
            text = str(row.get("cntr_tm") or "")
            if len(text) != 14 or not text.isdigit():
                continue
            ts = datetime.strptime(text, "%Y%m%d%H%M%S")
            if ts.date() != day:
                continue
            bars.append(
                MinuteBar(
                    ts=ts,
                    open=_to_decimal(row.get("open_pric")),
                    high=_to_decimal(row.get("high_pric")),
                    low=_to_decimal(row.get("low_pric")),
                    close=_to_decimal(row.get("cur_prc")),
                    volume=int(_to_decimal(row.get("trde_qty"))),
                )
            )
        return sorted(bars, key=lambda b: b.ts)

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
            # Round to the correct KRX price-band tick (호가가격단위) before
            # sending -- `int(order.limit_price)` alone truncates and
            # ignores tick bands entirely, which can produce an
            # invalid/rejected order price (whole-branch review finding
            # #5; `domain/services/tick_size.py`, Task 9, previously had
            # zero production callers). The unified table has applied
            # identically to KOSPI/KOSDAQ/KONEX since 2023-01-25 (see that
            # module's docstring), so any `Exchange` member yields the
            # same tick size today; `Order` carries no per-ticker market
            # field, so `Exchange.KOSPI` is passed as a neutral default.
            tick_price = round_to_tick(order.limit_price, Exchange.KOSPI)
            kwargs["ord_uv"] = str(int(tick_price))
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
        """Map one `kt00018` holdings row to `Position`.

        `stk_cd` comes back prefixed ("A005930"); left as-is, no holding
        ever matched a 6-digit ticker, so duplicate-entry checks and exits
        silently saw nothing.
        """
        quantity = _to_decimal(row.get("rmnd_qty"))
        avg_price = _to_decimal(row.get("pur_pric", row.get("avg_prc")))
        current_price = _to_decimal(row.get("cur_prc")) if row.get("cur_prc") else None
        return Position(
            ticker=normalize_code(row.get("stk_cd")),
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
        total_qty = _to_decimal(row.get("ord_qty"))
        remaining_raw = row.get("oso_qty") or row.get("ord_remnq")
        if remaining_raw not in (None, ""):
            # 미체결수량 -- the field scalp-it reads on the live account.
            remaining = _to_decimal(remaining_raw)
            filled_qty = max(total_qty - remaining, Decimal("0"))
        else:
            filled_qty = _to_decimal(row.get("cntr_qty", row.get("exec_qty")))
            remaining = total_qty - filled_qty
        if remaining <= 0:
            # The row is in the unfilled list, so something is still open.
            # Reporting 0 here made callers treat a live order as done and
            # skip cancelling it.
            remaining = total_qty if total_qty > 0 else Decimal("1")
            filled_qty = Decimal("0")
        status = OrderStatusEnum.PARTIAL if filled_qty > 0 else OrderStatusEnum.SUBMITTED
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

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise BrokerError(
                "Kiwoom broker not connected. Call connect() first.", broker_name="kiwoom"
            )
