# Spike 003: kiwoom-client mock-trading (is_mock=True) flow

**Task:** Task 3 of the Phase 0 spike plan. Goal: confirm `kiwoom-client`'s
mock-trading flow works end-to-end and its request/response shapes are
understood, before Phase 3 designs `KiwoomBroker` (this repo's
`BrokerInterface` implementation) against assumed shapes.

**Live mock call made: NO.**

## Credential check (step 1)

Checked, in order:

- `KIWOOM_APP_KEY` / `KIWOOM_APP_SECRET` / `KIWOOM_ACCOUNT_NO` in this
  machine's environment — not set (`env | grep -i kiwoom` returned nothing).
- `~/Documents/git/kiwoom-client/.env` — does not exist. Only
  `.env.example` is present, containing placeholder values
  (`KIWOOM_APP_KEY="your_app_key"`, etc.).
- `~/Documents/git/gpt-quant-v2` and this repo — only `.env.example` files
  exist (and a `docker/.env.example`); no real `.env` with secrets.
- `~/Documents/git/kiwoom-client/tests/integration_report.json` — the
  output file kiwoom-client's own `tests/integration_api_smoke.py` writes
  after a real run against the mock server. It does not exist, so no prior
  session in this environment ever ran a live call either.

**Finding: no mock-mode credentials are available anywhere on this machine.**
Per the plan's Global Constraints (no live trading, no live broker
credentials in Phase 0) and the task brief's explicit instruction, no
credentials were requested from the user. `scripts/spikes/kiwoom_mock_spike.py`
implements this discovery logic and exits with a `SKIPPED` message rather
than fabricating a call — running it (`python scripts/spikes/kiwoom_mock_spike.py`)
reproduces the same conclusion.

This falls to step 3 of the brief: document the request/response field
names by reading `kiwoom-client`'s source directly.

## What was read

`kiwoom-client` v0.3.0 (source checkout at `~/Documents/git/kiwoom-client`,
not installed via pip in this environment):

- `src/kiwoom_client/__init__.py` — `KiwoomAPI` / `AsyncKiwoomAPI` facades
- `src/kiwoom_client/base.py` — `BaseClient`/`AsyncBaseClient`, the actual
  HTTP request/response handling
- `src/kiwoom_client/auth.py` — OAuth2 token issuance
- `src/kiwoom_client/domestic/order.py`, `account.py`, `chart.py`,
  `market.py`
- `README.md`, `examples/basic_usage.py`, `examples/trading.py`,
  `examples/market_data.py`
- `tests/integration_api_smoke.py` (a real-server smoke-test harness the
  library ships, with a hand-curated parameter table per TR code — useful
  as a second source for request field names, though it was not run here
  since it also requires live credentials)

## Key surprise relative to what the plan assumed

**The brief's step 3 asked to document "the request/response Pydantic
models' actual field names." kiwoom-client has no Pydantic models at all.**
Every endpoint method (`buy_order`, `stock_quote`, `deposit_detail`, etc.)
is a thin wrapper that POSTs a TR (`api_id`) with a raw `**kwargs` dict as
the JSON body and returns the parsed JSON response as a plain
`dict[str, Any]` (see `ClientProtocol.request` in `base.py`). There is no
request or response schema/model layer in this library — field names are
whatever Kiwoom's own REST API defines per TR code, and kiwoom-client does
no validation of them. This means:

- Phase 3's `KiwoomBroker` will need to build its own typed request/response
  models (or a validation layer) if `BrokerInterface` wants typed shapes —
  kiwoom-client will not provide that structure for free.
- Field names below come from example code and the library's own smoke-test
  parameter table, not from a schema — they should still be cross-checked
  against Kiwoom's official TR documentation before Phase 3 relies on them,
  since (see next finding) the library's own sources disagree with each
  other on the order endpoints.

**Second surprise: the library's own example and README disagree on the
buy/sell/modify/cancel order field names.**

- `examples/trading.py` (last touched only by a repo-rename commit,
  `169947d`, i.e. stale content) uses:
  `acnt_no`, `acnt_prdt_cd`, `stk_cd`, `ord_qty`, `ord_uv`, `ord_dv`
  (order type field), `orig_ord_no` (original order number for
  modify/cancel).
- `README.md` (actively edited — 3 recent commits touching it) uses:
  `dmst_stex_tp` (exchange, e.g. `"01"` = KRX), `stk_cd`, `ord_qty`,
  `trde_tp` (order type field, not `ord_dv`), `ord_uv`, and
  `org_ord_no` (not `orig_ord_no`) for modify/cancel. No `acnt_no` /
  `acnt_prdt_cd` appear — account is presumably implicit from the
  authenticated session/token.

These cannot both be exactly right for the same live TR. The README is the
more recently maintained source, but neither is a verified live-call
result. **Phase 3 must verify the actual field names against Kiwoom's
official TR documentation (or a real mock-mode call) before finalizing
`KiwoomBroker`'s request builders** — do not trust either source blindly.

## Confirmed API shapes (from source, not a live call)

### Client / auth

- `KiwoomAPI(app_key, app_secret, is_mock=True)` — `is_mock=True` sets
  `base_url` to `https://mockapi.kiwoom.com` (vs. `https://api.kiwoom.com`
  for prod). See `_ClientCore.MOCK_URL` / `PROD_URL` in `base.py`.
- `api.login()` returns a `dict` with at least `token` (or `access_token`)
  and `token_type`, plus an expiry as either `expires_dt`
  (`yyyyMMddHHmmss` local time string) or `expires_in` (seconds) —
  `auth.py`'s `_TokenCache._store` / `_parse_expiry` handle both shapes.
  Login is optional; the token is issued lazily on first API call and
  auto-refreshed on expiry or a `return_code == 3` "token invalid" payload
  (which arrives as **HTTP 200**, not 401 — see the `TOKEN_ERROR_RETURN_CODES`
  comment in `base.py`, verified by the library's maintainer against the
  real prod server on 2026-08-15).
- Every response envelope carries `return_code` (`0` = success) and
  `return_msg`. `KiwoomAPIError` is raised for non-zero `return_code`
  (`base.py::_check_return_code`).
- Pagination: response carries `cont_yn` ("Y"/"N") and `next_key`; pass
  these back in as request params to continue.

### Account (module `api.account`, resource `/api/dostk/acnt`)

All methods are thin `**kwargs -> dict` wrappers; no fixed request/response
model. Examples of TR codes and their (from `tests/integration_api_smoke.py`'s
parameter table, used against the mock server historically) request params:

- `deposit_detail` → TR `kt00001`, e.g. `{"qry_tp": "3"}` — "예수금상세현황요청"
  (deposit/cash detail). Called with no args in the README (`api.account.deposit_detail()`).
- `account_evaluation` → TR `kt00004`, e.g.
  `{"qry_tp": "0", "dmst_stex_tp": "KRX"}` — "계좌평가현황요청". The
  `basic_usage.py` example instead calls it with `acnt_no=...` and
  `qry_tp="1"`, another example/README mismatch of the same kind noted
  above.
- `filled_position` → TR `kt00005`, `{"dmst_stex_tp": "KRX"}` — 체결잔고요청
  (position balance).
- `unfilled_orders` → TR `ka10075`, e.g.
  `{"all_stk_tp": "0", "trde_tp": "0", "stk_cd": "005930", "stex_tp": "0"}`.

No fixed set of response field names could be confirmed without a live
call — the library returns whatever Kiwoom's raw JSON payload contains
under whatever the TR's spec defines, and no test fixture with a captured
example response exists in this checkout.

### Quote / market data (module `api.market`, resource `/api/dostk/mrkcond`)

- `stock_quote(stk_cd="005930")` → TR `ka10004` ("주식호가요청" — stock
  order-book/quote request).
- Related: `order_book_info` → `ka10007`; `stock_minute_price` → `ka10006`;
  `stock_daily_weekly_monthly` → `ka10005`.
- Response values are **all strings** per the README's explicit note (price
  like `"+70000"`, change rate like `"-1.23"`, volume like `"1,234,567"`
  with thousands separators). kiwoom-client ships `to_number()` /
  `to_dataframe()` / `normalize()` helpers (`parsing.py`) to convert these;
  numeric-looking identifier fields like `stk_cd` and date fields like
  `base_dt` are deliberately left as strings by `to_dataframe()` (leading
  zeros matter for ticker codes).

### Order (module `api.order`, resource `/api/dostk/ordr`)

TR codes (fixed, from `order.py` source — reliable):

| Method | TR (api_id) | Purpose |
|---|---|---|
| `buy_order` | `kt10000` | 주식매수주문 |
| `sell_order` | `kt10001` | 주식매도주문 |
| `modify_order` | `kt10002` | 주식정정주문 |
| `cancel_order` | `kt10003` | 주식취소주문 |

Request field names: see the "surprise" discrepancy above — README's
version (`dmst_stex_tp`, `stk_cd`, `ord_qty`, `trde_tp`, `ord_uv`,
`org_ord_no`) is the actively-maintained source, but neither it nor
`examples/trading.py`'s version (`acnt_no`, `acnt_prdt_cd`, `ord_dv`,
`orig_ord_no`) is verified against a live call. `examples/trading.py`
does show the response is expected to carry `ord_no` (order number) —
both sources agree on that one field
(`ord_no = buy_result.get("ord_no", "")`).

## Files changed

- `scripts/spikes/kiwoom_mock_spike.py` (new) — standalone script that
  discovers credentials (env vars, then `.env` in kiwoom-client's repo or
  this repo, ignoring `.env.example`-style placeholder values), and either
  runs the real mock-mode flow (login → `account.deposit_detail` →
  `market.stock_quote` → `order.buy_order` → `order.cancel_order`,
  printing the exact response keys/shapes) or exits with a `SKIPPED`
  message pointing here. Verified by running it in this environment: it
  correctly reports no credentials found and exits 0 without any network
  call.
- `docs/spikes/003-kiwoom-mock-trading.md` (this file).

## Self-review

- Documented the credential absence honestly; did not fabricate a "live
  call" result. `scripts/spikes/kiwoom_mock_spike.py` was actually executed
  and its `SKIPPED` output is reproduced verbatim in the verification note
  above, not invented.
- All field names above are traced to specific files in
  `~/Documents/git/kiwoom-client` (source, README, examples, smoke-test
  harness) — none are invented. Where the brief's assumption (Pydantic
  models) turned out to be wrong, that is called out explicitly rather than
  silently substituted.
- No production code under `src/gpt_quant_v2/` was touched. Only
  `scripts/spikes/kiwoom_mock_spike.py` and this doc were added.

## Concerns for Phase 3

1. **kiwoom-client has no schema layer** — `KiwoomBroker` will need to own
   its own request/response typing (e.g. Pydantic models wrapping the raw
   dicts) rather than reuse anything from kiwoom-client.
2. **The order-endpoint field names are unverified and internally
   inconsistent** in kiwoom-client's own docs/examples (`ord_dv` vs.
   `trde_tp`, `orig_ord_no` vs. `org_ord_no`, presence/absence of
   `acnt_no`/`acnt_prdt_cd`). Phase 3 must resolve this against Kiwoom's
   official TR documentation or an actual mock-mode call — do not port
   either source's field names into `KiwoomBroker` without verification.
3. **No live mock-mode call has ever been made in this environment** (no
   `tests/integration_report.json` from kiwoom-client's own smoke harness
   either). The first real verification of any of these shapes will happen
   whenever credentials become available — that should happen before or
   during Phase 3, not be assumed away.
