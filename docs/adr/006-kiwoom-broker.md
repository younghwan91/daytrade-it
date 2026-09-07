# ADR-006: KiwoomBroker as the Korean-Equity Broker Adapter

## Status

Accepted. Supersedes the concrete-implementation list in
[ADR-005](005-abstract-broker.md) (the abstract `BrokerInterface` itself is
kept unchanged).

## Context

ADR-005 defined `BrokerInterface` for a US-market system and named
`PaperBroker`, `AlpacaBroker` (future), and `IBKRBroker` (future) as its
concrete implementations. This repo's redevelopment (see the SDD plan) has
since repointed the whole system at KRX-listed Korean equities, so Alpaca
and IBKR no longer apply -- neither broker trades KRX stocks.

The replacement is [`kiwoom-client`](https://pypi.org/project/kiwoom-client/),
a third-party Python wrapper around Kiwoom Securities' REST API, already
used elsewhere in this system's stack (`quant-airflow`'s ingestion DAGs).
Task 3's Phase-0 spike
([`docs/spikes/003-kiwoom-mock-trading.md`](../spikes/003-kiwoom-mock-trading.md))
read `kiwoom-client`'s source directly (no live credentials were available
then, and none are available now either -- see that doc's credential-check
section) and found two things that shape this ADR:

1. **No schema/validation layer.** Every `kiwoom-client` endpoint method
   (`buy_order`, `stock_quote`, `deposit_detail`, ...) is a thin
   `**kwargs -> dict` wrapper; there is no Pydantic model or equivalent on
   either the request or response side. `KiwoomBroker` therefore has to own
   all request/response typing itself -- `kiwoom-client` provides none of it
   for free.
2. **The library's own docs disagree with its own examples** on the
   order-endpoint (`buy_order`/`sell_order`/`modify_order`/`cancel_order`)
   request field names: `README.md` uses `dmst_stex_tp`/`trde_tp`/
   `org_ord_no`; `examples/trading.py` uses `acnt_no`/`acnt_prdt_cd`/
   `ord_dv`/`orig_ord_no`. This task re-verified the ambiguity directly
   against `kiwoom_client.domestic.order.Order`'s source (not just the
   spike doc's summary) while implementing `KiwoomBroker`: the source itself
   carries zero field-name information (pure `**kwargs` pass-through), so
   there is no third, more-authoritative source inside the library to break
   the tie, and the library's own integration-smoke-test parameter table
   does not cover the order TR codes either.

There is also a historical irony worth recording: the pre-v2 iteration of
this system (before the "GPT Quant v2" redevelopment that produced ADR-005's
US-market pivot) was built on KIS (Korean Investment Securities)'s broker
API. This ADR is, in effect, a return to a Korean broker -- just a different
one (Kiwoom, not KIS), for a different reason (KRX-first redevelopment, not
a continuation of the original KIS integration).

## Decision

### Keep the abstraction, replace the implementations

`BrokerInterface` (`infrastructure/brokers/base.py`) is unchanged.
`KiwoomBroker` (`infrastructure/brokers/kiwoom_broker.py`) is added as a new
concrete implementation, alongside `PaperBroker`. `AlpacaBroker`/`IBKRBroker`
are not implemented (the factory still raises `BrokerFactoryError` for
`alpaca`/`ibkr`, unchanged from before this task) -- they are dead code paths
now that this system trades exclusively on KRX, but removing the `BrokerType`
enum members and factory branches is out of this task's scope and left to a
future cleanup task if the team decides to formally drop non-Korean-broker
support.

`BrokerType` (`shared/config/settings.py`) gains a `KIWOOM = "kiwoom"`
member. `BrokerSettings` gains `kiwoom_app_key`/`kiwoom_app_secret`
(`SecretStr`, required when `type == KIWOOM`, sourced only from
`GPTQ_BROKER__KIWOOM_APP_KEY`/`GPTQ_BROKER__KIWOOM_APP_SECRET` env vars --
never hardcoded), `kiwoom_account_no` (reporting only), and
`kiwoom_is_mock` (defaults `True`, so a missing env var never silently
targets live capital). The broker factory
(`infrastructure/brokers/factory.py`) constructs a `kiwoom_client.KiwoomAPI`
from these settings and wraps it in `KiwoomBroker` for `broker_type ==
"kiwoom"`.

### Own request/response typing at the boundary

Since `kiwoom-client` does no validation, `KiwoomBroker` builds explicit
request-kwargs dicts (`_build_order_kwargs`) and explicit response mappers
(`_map_account`, `_map_position`, `_map_order_result`, `_map_order_status`,
`_map_quote`) rather than passing raw dicts through to callers. Response
field access is defensive (`.get(key, default)`, never a bare index) because
no field is contractually guaranteed present -- there is no captured example
response for any of the account/quote/order-status endpoints anywhere in
`kiwoom-client`'s own repo (per Task 3's spike), so several response field
names used here (e.g. `entr`, `rmnd_qty`, `cntr_qty`) are the standard
Kiwoom TR-documentation names, chosen for lack of a better source, and share
the same "not verified against a live call" caveat as the order-request
field names below.

### Field-name ambiguity: resolved as option (b) -- pick README, flag loudly

`KiwoomBroker` builds order requests using the **README's** field-name
convention (`dmst_stex_tp`, `stk_cd`, `ord_qty`, `trde_tp`, `ord_uv`,
`org_ord_no`), not `examples/trading.py`'s. Reasoning:

- The README is the actively-maintained surface (3+ commits touching this
  exact order-example section in the days before this task, most recently
  the day this task was done); `examples/trading.py` has been touched only
  by an unrelated repo-rename commit since March.
- A fresh integrator reading `kiwoom-client` today would trust the README
  over a possibly-stale example file.
- Sending *both* field-name conventions at once (the alternative,
  option (a) from the task brief) risks the live API rejecting the request
  for unrecognized/conflicting fields -- an unverified guess either way, but
  a single clean guess is easier to correct than a dual-send shim once real
  verification happens.

This is a **guess, not a verified fact**, and is flagged as such loudly and
repeatedly: a large comment block at the top of `kiwoom_broker.py`, a
`TODO(verify-before-live-capital)` in the same docstring, and the two
ambiguous field names isolated into module-level constants
(`_ORDER_TYPE_FIELD`, `_ORIG_ORDER_NO_FIELD`) specifically so a future
correction (if the README turns out wrong) is a one-constant change plus a
test update, not a broader refactor.

**This must be verified against a real Kiwoom mock-mode call (or official
Kiwoom TR documentation, if obtained) before `KiwoomBroker` is used against
any account with real capital.** No live credentials exist in this
environment (confirmed independently by Task 3, Task 6, and this task) --
that gap is tracked, not resolved, by this ADR.

## Consequences

### Positive

- `BrokerInterface`'s contract proves itself: `KiwoomBroker` slots in next
  to `PaperBroker` with no interface changes needed, validating ADR-005's
  original abstraction even under a market/broker pivot it wasn't designed
  for.
- All Kiwoom-specific shape assumptions are isolated in one file
  (`kiwoom_broker.py`), with the two genuinely ambiguous field names further
  isolated into named constants -- a future fix is small and localized.
- Credentials are never hardcoded; `is_mock` defaults to the safe (mock)
  value.

### Negative

- The order-request field-name choice, and most of the response-field
  names, are unverified against any live or mock Kiwoom call. If the README
  guess is wrong, every order placed against a real (even mock-mode) Kiwoom
  account will fail until corrected.
- `kiwoom-client`'s lack of a schema layer means `KiwoomBroker`'s mapping
  code has no compiler/type-checker backing for the *Kiwoom-side* field
  names (only for the `BrokerInterface`-side types) -- a silent Kiwoom-side
  API change would not be caught until a live call fails.
- Alpaca/IBKR remain named in `BrokerType` and the factory's error branches
  as not-implemented, rather than being removed -- minor enum/dead-code
  clutter left for a future cleanup.

### Mitigations

- The verification TODO is loud and specific (which two constants to flip,
  and what "success" looks like: `return_code == 0` on a real mock-mode
  call) so it is actionable the moment credentials become available, rather
  than requiring someone to re-derive the ambiguity from scratch.
- All 19 of this task's tests mock `KiwoomAPI` at the boundary and assert
  the exact kwargs sent for buy/sell/cancel, so the moment the field names
  are corrected, the tests that need updating are obvious and few.
