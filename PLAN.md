# gpt-quant-v2 → KRX/Toss-News Redevelopment Plan

> 이 문서는 krx-news-client 세션에서 진행한 딥리서치("뉴스를 매매전략에 붙이는 법")를 바탕으로,
> gpt-quant-v2를 토스 뉴스(krx-news-client) 기반 한국 주식(KOSPI/KOSDAQ) 시스템으로 전면 재개발하기 위해
> 세운 계획이다. 코드는 아직 수정되지 않았다 — 실행 전 사용자 검토용.

## Grounding: what I found in the actual code

**US-market assumptions baked in (not exhaustive, but the load-bearing ones):**

- `domain/value_objects/ticker.py`: `Ticker` validates `^[A-Z]+$`, 1–5 chars — will **reject every Korean 6-digit code** (e.g. `005930`). This is a hard blocker touched by nearly every module (news, signals, brokers, MCP tools).
- `domain/entities/stock.py`: `Exchange` enum is `NASDAQ/NYSE/AMEX/ARCA/BATS/OTC`; `is_large_cap`/`is_mid_cap`/`is_small_cap` thresholds are hardcoded in USD (`$10B`/`$2B`) — wrong currency, wrong buckets for KRX.
- `infrastructure/brokers/base.py`: `Account.currency` defaults `"USD"`; `Quote`/`Order`/`Position` are share-based, no lot-size or price-tick-size concept KRX needs (원 단위, 호가단위).
- `infrastructure/brokers/paper_broker.py`: hardcodes `currency="USD"`, zero commission by default, no 증권거래세/농특세.
- `application/ml/feature_engineer.py`: `MARKET_OPEN=9:30`, `MARKET_CLOSE=16:00` (US Eastern), English keyword list (`earnings`, `fda`, `guidance`...) — irrelevant for Korean-language news features.
- `shared/config/settings.py`: `TradingSettings.window_start/end` docstring says "Eastern Time"; `PolygonSettings`, `TiingoSettings`, `OpenAISettings`, `BrokerSettings` (paper/alpaca/ibkr) are all US-vendor-specific nested settings blocks.
- `infrastructure/external/tiingo_client.py` + `prompts.py`: Tiingo News API client and an **English-only** sentiment prompt (no Korean instructions, no Korean output schema).
- `docs/adr/005-abstract-broker.md`: notably, the **original** system (pre-v2) was built on KIS (Korean Investment Securities) and was deliberately abstracted away *from* Korea toward Alpaca/IBKR. This redevelopment is in effect a partial reversion of that ADR's stated direction, but replacing KIS with kiwoom-client — worth calling out explicitly as a new ADR that supersedes 005.
- `domain/services/signal_generator.py`: `SignalConfig` already has `sentiment_weight/ml_weight/technical_weight` (0.4/0.4/0.2) — confirmed, this exists and should be preserved/extended, not replaced. However `combine_signals()` currently does simple confidence-weighted averaging; there is **no risk-gate concept anywhere** — negative sentiment can currently only produce a SELL signal, never block/override a BUY from another source.
- `domain/services/risk_manager.py`: only has per-position/day-loss/exposure checks against `Decimal` USD limits — no sentiment- or disclosure-driven gate, no regime/macro overlay.
- `application/handlers/backtest.py`: pulls bars from `PolygonClient` (US), `transaction_cost_pct=0.001`/`slippage_pct=0.0005` are flat/approximate — no Korean 증권거래세 (사실상 매도시에만 부과, ~0.18–0.20% 코스피/코스닥 구간별로 다름) or commission-tier modeling. Metrics (`sharpe_ratio`, `sortino_ratio`, `max_drawdown`) are computed here in code — good, this is the place that must remain the single source of truth (see Phase 4 note on MCP/LLM verification discipline).
- `interfaces/mcp/tools.py`: `BacktestResult` and other Pydantic response models are returned to MCP callers (i.e., agent-facing) — this is exactly the boundary research finding #6 warns about: any LLM/agent consuming these tools must be constrained to *read* `BacktestHandler`-computed values, never recompute Sharpe/MDD itself.
- `infrastructure/persistence/timescaledb/init.sql`: `symbol VARCHAR(20)` — already wide enough for 6-digit KRX codes, no schema blocker there. Good.
- Frontend (`SignalsPage.tsx` etc.) is presentational/generic (ticker as free-text, BUY/SELL/ALL) — low risk, cosmetic Korean-market labeling changes only.

**Sibling libraries confirmed and their actual shapes:**
- `krx-news-client` (`~/Documents/git/krx-news-client`): `TossScraper().scrape_news()` → `list[NewsArticle]` (pydantic) with `id, source(dart|toss), category(disclosure|market|stock|economy|analysis|breaking), title, url, content, summary, tickers: list[str] (6-digit, e.g. "005930"), author, published_at, collected_at`. Also `DartScraper(api_key=...).scrape_disclosures()` → `list[Disclosure]` with `company, ticker, disclosure_type`. This is a **pull/pull-on-demand client library, no push/webhook** — ingestion must be a poller (cron/Airflow-style), not an event subscriber.
- `kiwoom-client`: mature, published (PyPI), `KiwoomAPI`/`AsyncKiwoomAPI`, 182 REST endpoints incl. `domestic/order.py`, `domestic/account.py`, `domestic/chart.py`, WebSocket realtime, `is_mock=True/False` (모의투자/실전투자) — maps directly onto `BrokerInterface`.
- `krx-fundamentals-client`: DART+KRX+Naver fundamentals, has a `screening.py` — useful for Phase 2 (Ticker/Stock enrichment) and Phase 4 (fundamentals-aware risk gate), optional/secondary dependency.
- 사용자는 또한 **`quant-airflow`**(kiwoom-client + krx-news-client를 TimescaleDB로 적재)와 **`kr-quant`**(walk-forward/Deflated Sharpe CI 가드레일이 있는 KOSPI/KOSDAQ 알파 리서치)도 운영 중 — 이 계획은 gpt-quant-v2를 이 스택과의 관계 속에서 위치시켜야 한다 (quant-airflow가 이미 하는 수집을 중복하는지, quant-airflow의 TimescaleDB를 그대로 소비할지) — 조용히 겹치게 두면 안 됨.

---

## Proposed name candidates (pick one before Phase 7, not now)

1. **`kr-news-quant`** — plain, mirrors `krx-news-client`/`kr-quant` naming already in the user's stack; signals "Korean news-driven quant system" unambiguously.
2. **`toss-alpha`** — short, brandable, signals the Toss-news-driven thesis specifically; risk: sounds tied to Toss forever even if a second news source is added later (e.g. Naver, Hankyung).
3. **`krx-signal-engine`** (or `krx-quant-engine`) — emphasizes it's the signal-generation/execution engine of the stack, distinguishing it from `krx-news-client` (data) and `kr-quant` (research) — clearest fit if this repo is meant to sit downstream of `quant-airflow`/`kr-quant` as the live trading component.

Recommend deferring the final pick until Phase 0's scope decision (does this replace or sit atop `quant-airflow`?) is resolved, since that shapes which name reads truest.

---

## Phase 0 — Discovery & Validation Spikes (no rebuild yet)

Goal: de-risk the two biggest unknowns before committing engineering time.

1. **Korean sentiment model spike** (research finding #7). Run a small labeled sample (50–100 Toss articles, hand-labeled BULLISH/BEARISH/NEUTRAL) through 2–3 candidate approaches and score agreement/precision:
   - GPT-4o/4o-mini with a *Korean-language* prompt (fastest to ship, keeps existing `openai_client.py` architecture, just needs new `prompts.py`).
   - A Korean FinBERT/RoBERTa-class open model (e.g. `snunlp/KR-FinBert-SC`, `kakaobank` finance models) run locally/via HF inference.
   - Decide: keep LLM-only, add local model as a second signal, or ensemble. **Do not assume accuracy — this spike must produce a number before Phase 4 sentiment-gate work starts.**
2. **Toss ingestion volume/reliability spike.** Run `TossScraper().scrape_news()` on a schedule for a few days; check article volume/ticker coverage/duplicate rate, since it's an unofficial internal API (per krx-news-client's own docs) — confirm it's not going to be rate-limited/blocked mid-build (there's already a `docs/krx-blocked.md`-style risk in a sibling repo — read that, it may describe exactly this risk for a related scraper).
3. **kiwoom-client mock-trading spike.** Stand up `KiwoomAPI(is_mock=True)`, place a test order end-to-end, confirm order/account/quote shapes so `BrokerInterface` implementation isn't guesswork.
4. **Scope decision vs. `quant-airflow`.** Confirm with the user: does this repo scrape/ingest itself, or read from `quant-airflow`'s TimescaleDB (which the user already built for exactly this)? This changes Phase 1 substantially — recommend leaning toward **consuming quant-airflow's DB for price/fundamentals data and doing its own lighter-weight polling only for live signal-triggering news** (Toss is time-sensitive; a nightly Airflow batch is probably too slow for "news → signal → trade" latency), but this is the user's call, not this plan's to decide.
5. Output of Phase 0: a short written decision log (not code) — sentiment model choice, ingestion architecture (self-poll vs. read quant-airflow), broker sandbox confirmed working. Everything downstream depends on this.

---

## Phase 1 — Data-Layer Swap (News + Market Data)

**Changes:**
- Replace `infrastructure/external/tiingo_client.py` with an `infrastructure/external/toss_news_client.py` wrapper around `krx-news-client.TossScraper` (+ `DartScraper` for disclosures) — since krx-news-client is pull-based with no pagination/date-filter API (`scrape_news()` returns "current feed" only), the wrapper needs a **poll loop + dedup-by-id cache** analogous to the old `RateLimiter`/`_seen_ids` pattern in `tiingo_client.py`, not a straight drop-in.
- Add a `to_domain_entity()` mapping from krx-news-client's `NewsArticle` → domain `NewsArticle`, same pattern as `TiingoClient.NewsItem.to_domain_entity()`. Map `category` (disclosure/market/stock/economy/analysis/breaking) into `analysis_result`/metadata since domain entity has no category field today — decide whether to add one.
- Add `infrastructure/external/dart_client.py` wrapping `DartScraper` for the "risk gate" disclosure feed (Phase 4 dependency).
- Replace `infrastructure/external/polygon_client.py` with a KRX market-data client. **Open question for Phase 0/1**: kiwoom-client's `domestic/chart.py`/`domestic/market.py` can serve OHLCV, so this may not need a separate vendor — evaluate whether `polygon_client.py` can simply be retired in favor of the broker client (kiwoom) providing both order execution and market data, simplifying the architecture versus keeping a Polygon-equivalent separate vendor.
- `shared/config/settings.py`: replace `PolygonSettings`/`TiingoSettings` nested blocks with `KiwoomSettings` (app_key/app_secret/is_mock) and drop/repurpose `TiingoSettings`; krx-news-client needs no API key for Toss but DART needs one (`DartSettings`).

**Risky/uncertain:** Toss's `wts-info-api` is unofficial and undocumented by Toss itself — no SLA, could change/break without notice (krx-news-client's own README calls it "비공식 내부 API"). Build a circuit-breaker/fallback path (e.g., degrade gracefully to DART-only) rather than hard-failing.

**Validate before moving on:** ingestion wrapper running against live Toss feed for at least a few days without breaking (from Phase 0 spike), dedup logic verified against duplicate `newsId`s.

---

## Phase 2 — Domain Model Adjustments for KRX

**Changes:**
- `domain/value_objects/ticker.py`: rewrite validator for **6-digit numeric KRX codes** (e.g. `^\d{6}$`), drop the uppercase-letter assumption. Decide whether to also validate ISIN-style prefix or KOSPI/KOSDAQ market flag as a separate value (kiwoom-client responses likely include market classification — check its `stock_info.py`/`market.py` response shape in Phase 0/1).
- `domain/entities/stock.py`: replace `Exchange` enum with `KOSPI/KOSDAQ/KONEX`; rewrite `is_large_cap`/`is_mid_cap`/`is_small_cap` thresholds in KRW (order-of-magnitude different numbers, not just currency-converted — Korean market-cap bands are commonly discussed in 조/억원 units) and confirm actual bucket cutoffs with the user or a public KRX classification rather than guessing.
- `infrastructure/brokers/base.py`: `Account.currency` default → `"KRW"`; consider adding `lot_size`/`tick_size` fields to `Order`/`Quote` since KRX has price-band tick-size rules (호가단위) that vary by price level — this affects both live orders and backtest realism (Phase 5).
- Sweep every other US-assumption: `shared/config/settings.py` `TradingSettings.window_start/end` docstrings/defaults (KRX trading hours 09:00–15:30 KST, no pre/post market in the same sense), timezone handling throughout (`datetime.now()` calls in `paper_broker.py` are naive/local — must become explicit `Asia/Seoul` or UTC-with-KST-display consistently).
- `Sentiment`/`Confidence`/`Price`/`Volume` value objects appear currency/locale-agnostic already — verify each still holds (quick read, low risk) but no changes expected beyond `Price` display formatting (원 vs $).

**Risky/uncertain:** market-cap bucket thresholds and tick-size table need a real source of truth (KRX rules), not invented numbers — flag as a task requiring the user's domain knowledge or a lookup against `krx-fundamentals-client`.

**Validate before moving on:** unit tests for `Ticker` against real KRX codes (005930, 000660, 035720, etc.) plus edge cases (preferred shares like `005935` — still 6 digits, fine); confirm no other module regex-matches on the old A-Z ticker format (`grep -rn "A-Z" src/` sweep before declaring this phase done).

---

## Phase 3 — Broker Swap to kiwoom-client

**Changes:**
- Add `infrastructure/brokers/kiwoom_broker.py` implementing `BrokerInterface` (connect/get_account/get_positions/place_order/cancel_order/get_order_status/get_quote) via `kiwoom_client.KiwoomAPI`/`AsyncKiwoomAPI` — the abstract interface in `infrastructure/brokers/base.py` maps reasonably cleanly onto kiwoom-client's `domestic/order.py`+`domestic/account.py`.
- `shared/config/settings.py`: replace `BrokerType.{PAPER,ALPACA,IBKR}` with `{PAPER,KIWOOM}`; replace `alpaca_api_key/secret/base_url` fields with kiwoom app_key/app_secret + `is_mock` toggle (this maps 1:1 onto kiwoom-client's own `is_mock` param, convenient).
- `infrastructure/brokers/paper_broker.py`: update `currency="USD"` → `"KRW"`, and rework commission/slippage defaults to approximate real KRX costs as a *simulation* default (see Phase 5 for the authoritative cost model — paper broker should reuse the same cost-model module, not duplicate it).
- New ADR: **ADR-006 (supersedes parts of ADR-005)** — explicitly document that the abstraction is being kept but the concrete adapters are now `PaperBroker` + `KiwoomBroker`, replacing Alpaca/IBKR; note the historical irony that the pre-v2 system was KIS-based and this is a return to a Korean broker via a different (kiwoom) vendor.
- `application/container.py`/DI wiring and `infrastructure/brokers/factory.py`: update factory switch statement for the new `BrokerType` values.

**Risky/uncertain:** kiwoom-client requires real app_key/app_secret even for mock trading (per its README) — need credentials provisioned before this phase's tests can run against a live sandbox; until then, unit tests should mock `KiwoomAPI` at the boundary.

**Validate before moving on:** an integration test placing a mock-mode order end-to-end through `KiwoomBroker`, confirming `OrderResult` mapping is correct (kiwoom order status/field names will differ from Alpaca-style responses — verify no silent field-mapping bugs).

---

## Phase 4 — Signal/Risk-Gate Redesign (the core research-driven work)

This is where the five research findings actually change behavior, not just data source.

**4a. Preserve and extend the existing multi-signal blend (finding #1).**
- Keep `SignalGenerator`'s `sentiment_weight/ml_weight/technical_weight` (0.4/0.4/0.2) pattern — do not regress to `NewsBasedStrategy`'s sentiment-only path (that class currently generates a full BUY/SELL signal from sentiment alone with no ML/technical corroboration — decide whether to retire `NewsBasedStrategy` in favor of always routing through `SignalGenerator.combine_signals()`, since the sentiment-only path is exactly the "not reliable standalone" pattern research finding #1 warns against).
- `feature_engineer.py`: replace English `DEFAULT_KEYWORDS` with Korean financial keywords (실적, 매출, 영업이익, 상한가, 하한가, 유상증자, 무상증자, 자사주매입, 감사의견, etc.) and replace `MARKET_OPEN/CLOSE` with KRX hours; retrain XGBoost model on Korean price/feature data from scratch (old models trained on US patterns are not transferable).

**4b. Add sentiment as a risk gate, independent of the signal generator (finding #3 — the most defensible new pattern).**
- New domain service, e.g. `domain/services/risk_gate.py`, sitting *before* order placement, not inside `SignalGenerator`: given the latest sentiment for a ticker (and, when available, a DART disclosure flag), it can force `REJECT`/`HALT_ENTRY`/`FORCE_EXIT` independent of whatever `TradingSignal` was produced by ML/technical/composite sources. This should hook into `RiskManager` (`domain/services/risk_manager.py`) as an additional check alongside `check_position_limit`/`check_daily_loss_limit`, and specifically into `application/handlers/trade_execution.py` before `place_order` is called.
- DART disclosure severity: `Disclosure.disclosure_type` from krx-news-client needs a classification (e.g. 관리종목지정, 상장폐지, 감사의견거절 = hard halt; routine filings = no gate) — this taxonomy doesn't exist yet anywhere in the codebase and needs to be built, likely informed by `krx-fundamentals-client`.

**4c. Portfolio-level stress overlay (finding #2, #4).**
- New service, e.g. `domain/services/regime_detector.py` / `application/handlers/portfolio_risk.py`: compute a volatility proxy (realized vol of KOSPI/KOSDAQ index or portfolio) and a credit-spread-like proxy (harder in KR — no direct equivalent of US HY-OAS; candidates: KRW CDS spread, or a simpler VKOSPI-based proxy) to detect "crisis/high-uncertainty" regimes, then upweight `sentiment_weight` in `SignalConfig` dynamically during those regimes (finding #4). This is the most research-novel and least-specified piece — treat as its own spike/prototype (backtest-only, not wired into live trading) before committing to a specific overlay formula, since the research finding is about *combining* macro stress + LLM sentiment, and the exact proxy construction for the Korean market isn't validated anywhere yet.

**4d. LLM/agent computation discipline (finding #6).**
- Audit every MCP tool in `interfaces/mcp/tools.py` (esp. `BacktestResult`, `Signal`, `TradeResult` models) and confirm each numeric risk metric (sharpe_ratio, max_drawdown_pct, etc.) is populated **only** from `BacktestHandler`/`RiskManager`/`SignalGenerator` code paths, never left for an agent to "fill in" or recompute from raw trade lists via its own arithmetic. Add an explicit code comment/docstring convention on `interfaces/mcp/tools.py` response models stating this, and consider a test that asserts MCP tool outputs match directly-computed values from the same handler (regression guard against a future refactor accidentally exposing raw data + a "compute this yourself" tool description).

**Risky/uncertain:** 4c (macro overlay) is the piece most likely to be research-grade rather than production-grade; 4b's disclosure taxonomy needs domain research the user is better positioned to do than to have invented here.

**Validate before moving on:** backtest the with/without risk-gate and with/without regime-overlay variants against the same period and confirm risk-gate at minimum doesn't hurt returns (it should reduce drawdown even if it slightly reduces raw return — that's the point) before enabling in Phase 5's default config.

---

## Phase 5 — Backtest & Cost-Model Rework

**Changes:**
- `application/handlers/backtest.py`: replace `PolygonClient` dependency with the KRX market-data source decided in Phase 1; replace flat `transaction_cost_pct=0.001`/`slippage_pct=0.0005` with a proper Korean cost model module (new `domain/services/cost_model.py` or similar): 매도 시 증권거래세 (現 코스피 약 0.18%/코스닥 약 0.18%~0.20% 구간, rates change periodically — confirm current rate rather than hardcoding an assumed one), 위탁수수료 (broker-specific, kiwoom's actual commission schedule), and 슬리피지 calibrated to KRX liquidity (small/mid-cap KOSDAQ names slip far more than large-cap KOSPI — a flat slippage % is unrealistic across the KRX cap spectrum, more so than it was for US large-caps).
- Reuse this cost model in `PaperBroker` (Phase 3) so paper trading and backtesting share one source of truth for costs, rather than two hand-maintained numbers drifting apart.
- `BacktestMetrics` structure (sharpe/sortino/max_drawdown/win_rate/profit_factor) stays as-is — the calculation *inputs* change (KRW P&L, KRX cost model) but the metrics themselves are currency-agnostic, low risk there.

**Risky/uncertain:** getting the current KRX transaction-tax rate and any recent changes right; KOSDAQ liquidity/slippage assumptions need calibration against real historical spreads, not guessed constants — this is exactly the kind of number research finding #5 warns must be realistic, not carried over from a near-zero-cost US backtest assumption.

**Validate before moving on:** run the Phase 4 signal/risk-gate variants through this realistic KR cost model and sanity-check that a previously-"profitable"-looking (US-cost-model) strategy doesn't evaporate once realistic KRX costs are applied — if it does, that's exactly the kind of thing that must be caught before any live capital is put behind it.

---

## Phase 6 — Interfaces & Frontend Rework

**Changes:**
- `interfaces/api/routers/news.py`, `signals.py`, `trades.py`, `portfolio.py`: update field validators (`ticker: str = Field(min_length=1, max_length=5)` in `NewsAnalyzeRequest` → 6, fixed length) and any USD-formatted response fields.
- `interfaces/mcp/tools.py`: update tool descriptions/docstrings that reference US concepts (tickers, "shares", USD) to KRX-equivalent language; re-verify the discipline from 4d holds after these edits.
- `interfaces/websocket/*`: likely low-risk, mostly pass-through — confirm no US-specific assumptions (didn't inspect deeply; quick pass needed).
- Frontend (`SignalsPage.tsx`, `DashboardPage.tsx`, `PortfolioPage.tsx`, `BacktestPage.tsx`): mostly cosmetic — ticker input becomes 6-digit numeric, currency formatting $ → ₩, direction/labels stay the same (BUY/SELL generic enough). Low risk, do last.
- `docker/timescaledb/init.sql`: `symbol VARCHAR(20)` already accommodates KRX codes — no schema migration needed there, but `currency`-adjacent columns (if any added during Phase 3/5) need an Alembic migration.

**Validate before moving on:** full API + MCP smoke test against the new broker/data stack in mock mode; frontend manual click-through against a running backend.

---

## Phase 7 — Rename + Release

- Pick final name from the 2–3 candidates above (or another) once Phase 0's scope question (standalone vs. quant-airflow-integrated) is resolved.
- Rename package (`gpt_quant_v2` → new slug), update `pyproject.toml`, Docker container names/network name (`gpt_quant_*` throughout `docker-compose.yml`), CI workflow names, README, ADR index.
- Tag this as a major version bump / new repo history checkpoint (discuss with user whether to keep git history or start fresh given the scope of change — given this is described as "전면 재개발", a fresh `v0.1.0` under the new name with a link back to the old repo as prior art may be cleaner than pretending it's a patch release).

---

## Cross-cutting risks to flag explicitly to the user before approving this plan

1. **Korean sentiment accuracy is unvalidated** (Phase 0/4a) — the whole system's edge depends on this and it's the single biggest unknown; recommend not writing a line of Phase 4 risk-gate code until Phase 0's spike produces a number.
2. **Toss's news API is unofficial/undocumented** — architecture must tolerate it disappearing or changing shape without notice.
3. **Overlap with `quant-airflow`/`kr-quant`** — building this in isolation risks duplicating ingestion/research work the user has already built elsewhere in the same GitHub account; Phase 0 must resolve this before Phase 1 starts, or Phase 1's data-layer work may need to be redone.
4. **KRX transaction-cost and tick-size numbers must come from an authoritative current source**, not assumptions baked into this plan — this was not verified and neither should the implementation team without checking.
5. **kiwoom-client requires live credentials even for mock mode** — provisioning lead time should be accounted for before Phase 3 starts.

### Critical Files for Implementation

- `src/gpt_quant_v2/domain/value_objects/ticker.py`
- `src/gpt_quant_v2/domain/services/signal_generator.py`
- `src/gpt_quant_v2/domain/services/risk_manager.py`
- `src/gpt_quant_v2/infrastructure/brokers/base.py`
- `src/gpt_quant_v2/infrastructure/external/tiingo_client.py`
- `src/gpt_quant_v2/application/handlers/backtest.py`
- `src/gpt_quant_v2/shared/config/settings.py`
- `~/Documents/git/krx-news-client/src/krx_news_client/scrapers/toss.py`
- `~/Documents/git/kiwoom-client/src/kiwoom_client/domestic/order.py`

---

*krx-news-client 세션의 딥리서치 오케스트레이션 결과로 작성됨 (2026-09-07).*
