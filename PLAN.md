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

---

## Global Constraints (all tasks below)

- Work happens only in this worktree, on branch `feat/krx-toss-redevelopment`. Never touch `main`.
- No live trading, no live broker credentials are used in Phase 0. All spikes run in mock/sandbox mode or against read-only public data.
- Every numeric claim about model accuracy, API reliability, or broker behavior must come from an actual run recorded in the task's output — never asserted from memory or documentation alone.
- Commit after each task; do not squash across tasks. Author/committer identity must remain the human git identity already configured in this repo (`chyohw97@gmail.com`) — per `.github/workflows/commit-identity.yml`, agent contribution goes in a `Co-Authored-By:` trailer only, never as author/committer email.
- These are Phase 0 tasks from the plan above ("Discovery & Validation Spikes"). Their job is to produce a decision, backed by evidence, not to build production code. Phase 1+ tasks will be written after Phase 0's outputs are known — do not start Phase 1+ work in these tasks.

## Task 1: Korean news sentiment model validation spike

**Goal:** produce a real accuracy/agreement number for Korean-language financial-news sentiment classification, so Phase 4's risk-gate design has evidence instead of an assumption (deep-research finding: Korean sentiment models are unvalidated).

**Steps:**
1. Using `krx-news-client`'s `TossScraper` (installed from `~/Documents/git/krx-news-client`, or `pip install krx-news-client` if published), pull a sample of at least 50 recent Toss news articles across a mix of categories (market/stock/economy/breaking).
2. Hand-label each article's sentiment as BULLISH/BEARISH/NEUTRAL yourself (best-effort, single-rater — note this limitation explicitly in the report) based on the article text.
3. Score the same articles with two approaches:
   - A Korean-language prompt against an LLM (reuse the existing `openai_client.py` structured-output pattern from this repo, but with a Korean system prompt written for this spike — do not modify the production `prompts.py` file, write a standalone script under `scripts/spikes/sentiment_spike.py`).
   - A Korean FinBERT/RoBERTa-class open model if one can be loaded within a reasonable time/dependency budget (e.g. via `transformers` + a public Korean finance sentiment checkpoint); if none can be loaded within ~30 minutes of effort, note that as a finding rather than blocking on it.
4. Compute agreement (accuracy against your hand labels) for each approach. Save the labeled dataset, scores, and a short written verdict (which approach to use, or whether neither clears a usable bar) to `docs/spikes/001-sentiment-validation.md`.
5. Do not wire this into any production code path — this is a standalone spike script + report only.

**Report:** status (DONE/BLOCKED/NEEDS_CONTEXT), the accuracy numbers for each approach, the recommendation, and the path to `docs/spikes/001-sentiment-validation.md`.

## Task 2: Toss ingestion reliability spike

**Goal:** confirm `krx-news-client`'s `TossScraper` is stable enough to build on before committing further engineering time (deep-research + plan flag: it's an unofficial/undocumented API with no SLA).

**Steps:**
1. Write a standalone script `scripts/spikes/toss_reliability_spike.py` that polls `TossScraper().scrape_news()` on an interval (e.g. every few minutes) and logs: timestamp, article count returned, any errors/exceptions, and duplicate-id rate against previously seen articles.
2. Run it for as long as is practical in this session (at minimum, run it enough times back-to-back to see whether repeated calls succeed, return consistent shapes, and don't get rate-limited/blocked) — note in the report that a longer multi-day run is recommended before Phase 1 truly commits, since this single session can't observe multi-day stability.
3. Record findings — article volume per poll, error rate, duplicate rate, any observed blocking/rate-limiting — in `docs/spikes/002-toss-reliability.md`.
4. Check whether a sibling repo already documents Toss/KRX-scraper blocking risk (the plan mentions this may exist) — search `~/Documents/git/*/docs/` for anything about scraper blocking and summarize what it says if found.

**Report:** status, observed reliability numbers, whether there's a known-blocking risk documented elsewhere, and the path to `docs/spikes/002-toss-reliability.md`.

## Task 3: kiwoom-client mock-trading spike

**Goal:** confirm `kiwoom-client`'s mock-trading (`is_mock=True`) flow actually works end-to-end and its request/response shapes are understood, before Phase 3 designs `KiwoomBroker` against assumed shapes.

**Steps:**
1. Check whether mock-mode credentials (app_key/app_secret) are already available in this machine's environment or a `.env`/config file anywhere in `~/Documents/git/kiwoom-client` or `~/Documents/git/gpt-quant-v2` (e.g. `.env.example` shows the variable names; check for a non-example `.env`). Do NOT ask the user for credentials in this task — if none are available, that is itself the finding.
2. If credentials are available: using `kiwoom-client` (installed from `~/Documents/git/kiwoom-client` or via pip), write a standalone script `scripts/spikes/kiwoom_mock_spike.py` that connects with `is_mock=True`, fetches account info, fetches a quote for a real KRX ticker (e.g. 005930), and places+immediately cancels one mock order. Record the exact response shapes (field names) for account/quote/order.
3. If credentials are NOT available: read `kiwoom-client`'s README and its `domestic/order.py`, `domestic/account.py`, `domestic/chart.py` source directly to document the request/response Pydantic models' actual field names (no live call needed) — this still de-risks Phase 3's `KiwoomBroker` design even without a live call.
4. Write findings to `docs/spikes/003-kiwoom-mock-trading.md`: whether a live mock call was made, the confirmed field shapes for account/quote/order, and anything that surprised you relative to what the plan assumed.

**Report:** status, whether live credentials were available, the confirmed API shapes, and the path to `docs/spikes/003-kiwoom-mock-trading.md`.

## Task 4: quant-airflow / kr-quant overlap analysis

**Goal:** answer the plan's open scope question — should gpt-quant-v2 (renamed) ingest Toss/DART/market data itself, or read from `quant-airflow`'s existing TimescaleDB pipeline? — with an actual comparison, not a guess, so Phase 1's data-layer design doesn't get redone.

**Steps:**
1. Read `~/Documents/git/quant-airflow`'s README and DAG definitions to determine: does it already ingest krx-news-client (Toss/DART) data? At what frequency? Into what schema/table shape? Does it ingest kiwoom-client price/quote data too?
2. Read `~/Documents/git/kr-quant`'s README to understand what it consumes and whether it already depends on quant-airflow's output.
3. Write a short recommendation to `docs/spikes/004-data-layer-scope.md`: (a) what quant-airflow already provides today, with evidence (file/DAG names, schedule), (b) what gpt-quant-v2 would still need to poll itself regardless (the plan's working hypothesis is that nightly/batch data can come from quant-airflow, but time-sensitive news-to-signal latency needs its own lighter poller — confirm or refute this against what quant-airflow's DAGs actually schedule), (c) a concrete recommendation for Phase 1.
4. This is a research/analysis task — no code changes to quant-airflow or kr-quant, read-only.

**Report:** status, the recommendation (self-ingest vs. consume quant-airflow vs. hybrid), and the path to `docs/spikes/004-data-layer-scope.md`.

---

## Phase 0 Results (informs Phase 1 tasks below)

- **Task 1 (sentiment):** FinBERT (`snunlp/KR-FinBert-SC`) scored 60.5% accuracy / 18.2% BEARISH recall on 76 hand-labeled Toss articles — not usable for a risk-gate as-is. OpenAI comparison leg parked pending a user-supplied `OPENAI_API_KEY` (Ruling: do not block Phase 1 on this — Phase 4's risk-gate work will need to revisit this before it ships).
- **Task 2 (Toss reliability):** stable across 23 live calls this session (0 errors, consistent shape, no rate-limiting observed), but Toss ToS-for-automation is unverified — a policy risk, not something a technical spike resolves.
- **Task 3 (kiwoom shapes):** no live credentials available; source-reading only. kiwoom-client has no schema/validation layer (raw dict in/out), and its own README and examples **disagree** on order-request field names (`ord_dv`/`orig_ord_no` vs `trde_tp`/`org_ord_no`) — Phase 3's `KiwoomBroker` must verify against a real mock call or Kiwoom's official TR docs before trusting either.
- **Task 4 (data-layer scope):** confirmed hybrid — consume `quant-airflow`'s TimescaleDB read-only (LAN, `psycopg2`, same pattern as `kr-quant`'s `storage.py`) for all batch/EOD data (`daily_bars_adjusted`, `supply_demand`, `sector_index`, `earnings`, `consensus`, historical `news_articles`/`disclosures`/`news_judgments`); build gpt-quant-v2's own lightweight Toss/DART poller only for latency-sensitive news-to-signal, since quant-airflow's own design spec explicitly excludes sub-minute reaction. Do not write into quant-airflow's tables/upsert keys — read-only consumer only. `news_judgments.judged_at` is stored in UTC — convert before joining KST-naive data.

**Ruling (sequencing):** Phase 1's data-mapping code (both the quant-airflow DB reader and the Toss/DART poller) needs to construct domain `Ticker` value objects for 6-digit KRX codes, but `Ticker` currently validates `^[A-Z]+$` (US-only) — this was originally scoped as Phase 2 work. Pulling the `Ticker`/`Exchange` fix forward to the start of Phase 1 (Task 5 below) rather than building Phase 1's data clients against a value object that will reject every real input. This is a scope-order change, not a scope change — Phase 2's other domain-model work (market-cap buckets, tick-size, broker currency) still happens later, on schedule.

## Task 5: Ticker/Exchange domain fix for KRX codes

**Goal:** unblock every later task that needs to construct a domain `Ticker` for a Korean stock — currently impossible.

**Steps:**
1. In `src/gpt_quant_v2/domain/value_objects/ticker.py`, replace the `^[A-Z]+$` (1-5 char) validator with one accepting 6-digit numeric KRX codes (`^\d{6}$`, e.g. `005930`). Confirm this also accepts preferred-share codes (still 6 digits, e.g. `005935`).
2. In `src/gpt_quant_v2/domain/entities/stock.py`, replace the `Exchange` enum (`NASDAQ/NYSE/AMEX/ARCA/BATS/OTC`) with `KOSPI/KOSDAQ/KONEX`.
3. Do **not** attempt to fix market-cap bucket thresholds (`is_large_cap`/`is_mid_cap`/`is_small_cap`) in this task — those need a real KRW threshold source per the plan's Phase 2 section and are explicitly out of scope here. If leaving them as unconverted USD thresholds would silently produce wrong classifications when called with KRW prices, add a `NotImplementedError` or an explicit TODO with a comment pointing at this plan's Phase 2 section — do not guess at KRW numbers.
4. Sweep the codebase (`grep -rn "A-Z" src/`) for any other module that regex-matches on the old US ticker format and would break silently on a 6-digit input — list what you find in your report even if you don't fix it (flag as a concern for the controller to triage into a later task).
5. Write/update unit tests for `Ticker` against real KRX codes (005930 삼성전자, 000660 SK하이닉스, 035720 카카오) plus the preferred-share edge case, and against invalid inputs (should still reject, e.g. `"AAPL"`, `"12345"` (5 digits), `"1234567"` (7 digits)).
6. Do not touch broker/settings/backtest code in this task — scope is `ticker.py` + `stock.py` (Exchange enum) + their direct tests only. Downstream currency/broker changes are separate later tasks.

**Report:** status, what the grep sweep in step 4 found (list of affected files even if unfixed), test results, and the report file path.

## Task 6: quant-airflow read-only TimescaleDB market-data client

**Goal:** replace `infrastructure/external/polygon_client.py` (US market data) with a read-only client over `quant-airflow`'s TimescaleDB, per Task 4's confirmed recommendation — no new ingestion, no writes.

**Steps:**
1. First, verify LAN reachability to the TimescaleDB host from this machine (the connection details are in `quant-airflow/README.md:36-46` — read them, don't guess) — a plain `psycopg2.connect(...)` + `SELECT 1` is enough to confirm. If unreachable, stop and report BLOCKED/NEEDS_CONTEXT rather than building a client you can't test — this was flagged in spike 004 as unverified and needs a real check now.
2. If reachable: write `src/gpt_quant_v2/infrastructure/external/quant_airflow_db_client.py`, a read-only client (reuse `kr-quant`'s `storage.py` connection pattern — read it first at `~/Documents/git/kr-quant/` for the exact pattern, don't reinvent) exposing at minimum: fetch adjusted daily bars for a ticker/date-range from `daily_bars_adjusted`, fetch latest `supply_demand`/`sector_index` rows, fetch `earnings`/`consensus` for a ticker. Map rows to this repo's existing domain entities (check `domain/entities/` for an existing `Bar`/`PriceBar`-shaped entity to map into, or note if one doesn't exist and a new lightweight DTO is needed instead — don't invent a domain entity without checking first).
3. Enforce read-only at the connection level (e.g. `psycopg2` connection opened with a read-only role/transaction if quant-airflow exposes one, or at minimum never issue INSERT/UPDATE/DELETE in this client's code) — this client must never write into quant-airflow's schema.
4. Handle the `news_judgments.judged_at` UTC-storage gotcha explicitly if this client touches that table (convert to KST before any date-boundary comparison) — cite spike 004's note on this.
5. Write tests: with the LAN reachable, an integration-style test that reads a known real row (pick a well-known large-cap ticker like 005930) and confirms shape; a unit test for the KST/UTC conversion logic using fixed inputs (no DB needed).
6. Do not modify `quant-airflow` or `kr-quant` themselves — this repo's client only, read-only.

**Report:** status, whether LAN reachability was confirmed, what tables/queries were implemented, test results, and the report file path. If blocked on reachability, say so clearly — this is a legitimate stop condition, not a task failure to route around by guessing at data shapes.

## Task 7: Toss/DART lightweight news poller

**Goal:** replace `infrastructure/external/tiingo_client.py` (US news) with a poller over `krx-news-client`, scoped as the low-latency path quant-airflow's own design explicitly excludes (spike 004).

**Steps:**
1. Write `src/gpt_quant_v2/infrastructure/external/toss_news_client.py` wrapping `krx_news_client.TossScraper` — since `scrape_news()` returns only the current snapshot with no pagination (confirmed in Task 2's spike), implement a poll loop + dedup-by-id cache, following the pattern Task 2's `scripts/spikes/toss_reliability_spike.py` already exercised (read it first for the working call pattern — don't reinvent the wheel it already validated). Recommended poll interval: something in the few-minutes range, not seconds — Task 2 found no rate-limiting at faster intervals in-session, but this is a production poller, not a spike, so default conservatively and make the interval configurable.
2. Write `src/gpt_quant_v2/infrastructure/external/dart_client.py` wrapping `krx_news_client.DartScraper` for disclosures — this feeds Phase 4's risk-gate work later, so at minimum surface `disclosure_type` on whatever it returns; don't build the risk-gate classification logic itself in this task, that's Phase 4.
3. Map both to the domain `NewsArticle` entity (`to_domain_entity()`, same pattern `TiingoClient.NewsItem.to_domain_entity()` used) — construct `Ticker` from the 6-digit codes krx-news-client already returns (this now works after Task 5). The domain `NewsArticle` entity has no `category`/`disclosure_type` field today (checked in the plan's grounding section) — add one if it's a clean, additive Pydantic field change; if it requires restructuring the entity, stop and report DONE_WITH_CONCERNS rather than improvising a bigger change.
4. Store this poller's own output separately from anything quant-airflow owns — per spike 004, do not write into quant-airflow's `news_articles`/`news_judgments` tables or reuse its upsert keys. A new table/store is out of scope for this task unless the plan's persistence layer already has an obvious place for it — if unsure, keep it in-memory/ephemeral for now and flag persistence as a follow-up, don't invent new TimescaleDB schema without checking with the controller first.
5. Build a circuit-breaker/fallback path per the plan's original Phase 1 note: if Toss polling fails repeatedly, degrade to DART-only rather than hard-failing the whole poller.
6. Write tests: dedup logic (duplicate article IDs across polls don't re-emit), circuit-breaker behavior (simulated Toss failures fall back to DART), domain mapping correctness.

**Report:** status, what was implemented, test results, any concerns about the `NewsArticle` entity change (if made), and the report file path.

## Task 8: settings.py rewire for Phase 1

**Goal:** wire Tasks 6 and 7's new clients into configuration, retiring the now-dead US-vendor settings.

**Steps:**
1. In `src/gpt_quant_v2/shared/config/settings.py`, remove `PolygonSettings` and `TiingoSettings` (or repurpose `TiingoSettings` if Task 7's implementer left useful structure — check Task 7's actual diff before deciding remove-vs-repurpose).
2. Add `QuantAirflowDBSettings` (read-only DSN/connection params for Task 6's client — host/port/db/user/password/sslmode as applicable, sourced from env vars, never hardcoded) and `TossNewsSettings`/`DartSettings` (poll interval, DART API key env var name) for Task 7's clients.
3. Update dependency-injection wiring (wherever `PolygonClient`/`TiingoClient` were constructed and injected — check `application/` for a container/factory) to construct and inject the new clients instead.
4. Do not touch broker settings (`BrokerSettings`/`AlpacaSettings`/etc.) — that's Phase 3, separate task, not this one.
5. Run the existing test suite for `settings.py` and the DI wiring to confirm nothing else silently depended on the removed settings classes.

**Report:** status, what was removed/added/rewired, test results, and the report file path.

---

## Phase 1 Results (informs Phase 2 tasks below)

Tasks 5-8 all complete, reviewed clean. Ticker/Exchange are KRX-native (pulled forward from Phase 2 into Task 5). `quant_airflow_db_client.py` (read-only TimescaleDB reader) and `toss_news_client.py`/`dart_client.py`/`news_poller.py` (lightweight Toss/DART poller, circuit-breaker fallback) both exist and are tested, but neither is wired into any application-layer entry point yet — that's later-phase work (Phase 5 backtest rework, Phase 6 interfaces). `PolygonSettings`/`TiingoSettings` are retired; `PolygonClient`/`TiingoClient` classes and their handler consumers (`backtest.py`, `signal_generation.py`, `news_analysis.py`, MCP `tools.py`) still exist as dead code, not yet rewired — flagged for Phase 5/6, not fixed here.

**Carried-forward debt (do not re-discover, just be aware):**
- `interfaces/api/routers/portfolio.py:318` and `interfaces/api/routers/news.py:194` still gate on `ticker.isalpha()` + 1-5 chars (US format) — will reject real KRX codes. Phase 6 (interfaces rework) territory.
- 40 pre-existing test failures (stale `AAPL`/`NASDAQ` fixtures in `test_entities.py`, `test_api_signals.py`, and possibly `conftest.py`) — a fixture-migration task, not yet scheduled. Any task touching these files should fix fixtures it directly needs, not the whole set at once.
- kiwoom-client's own README vs. examples disagree on order-request field names (`ord_dv`/`orig_ord_no` vs `trde_tp`/`org_ord_no`) — must be resolved with a real mock call or Kiwoom's official TR docs before Phase 3 (`KiwoomBroker`) trusts either.
- Task 1's OpenAI-vs-FinBERT sentiment comparison is parked pending a user-supplied `OPENAI_API_KEY` — Phase 4 (risk-gate design) must not assume FinBERT's 60.5%/18.2% numbers are the final word; re-run Task 1's spike script once a key is available.

## Task 9: Market-cap buckets, tick-size, currency/timezone sweep (Phase 2 completion)

**Goal:** finish Phase 2's domain-model KRX adjustments that Task 5 explicitly deferred (market-cap thresholds, tick-size) plus the remaining currency/timezone sweep the original plan scoped for Phase 2.

**Steps:**
1. **Market-cap buckets** (`domain/entities/stock.py`, `is_large_cap`/`is_mid_cap`/`is_small_cap`, currently `NotImplementedError` per Task 5): research KRX's actual, current market-cap classification convention (this is public regulatory/market-convention information, not something requiring the user's personal knowledge — look it up rather than guessing or asking; if using web search, cite the source in code comments and the report). Implement thresholds in KRW. If authoritative current thresholds can't be confirmed with reasonable confidence, leave the `NotImplementedError` in place rather than guessing, and say so clearly in the report — don't silently ship invented numbers.
2. **Tick-size table** (호가단위): KRX price-band tick sizes vary by price level and differ between KOSPI/KOSDAQ in places — research the actual current table (again, public market-convention information) and implement it as a pure function (e.g. `tick_size_for(price: Decimal, market: Exchange) -> Decimal`) in an appropriate domain/shared location — check if `Price`/`Order` value objects are the natural home, or if a new small module is warranted. Add `lot_size`/`tick_size` awareness to `Order`/`Quote` in `infrastructure/brokers/base.py` per the original plan's Phase 2 section, if that's a clean additive change; if it requires restructuring, stop and report DONE_WITH_CONCERNS rather than improvising.
3. **Currency default**: `infrastructure/brokers/base.py` `Account.currency` default → `"KRW"`. `infrastructure/brokers/paper_broker.py` currency literal → `"KRW"` (commission/slippage defaults can stay as placeholders for now — Phase 5 owns the real KRX cost model; just fix the currency string, don't invent a cost model here).
4. **Timezone sweep**: `shared/config/settings.py` `TradingSettings.window_start/end` docstrings/defaults currently reference Eastern Time — update to KRX hours (09:00–15:30 KST, no separate pre/post-market session in the same sense as US markets). Sweep `infrastructure/brokers/paper_broker.py` (and anywhere else a grep for `datetime.now()` without explicit timezone turns up within `src/gpt_quant_v2/`) for naive/local-time calls that should be explicit `Asia/Seoul` or UTC-with-KST-display — fix what's directly related to trading-hours logic; if the sweep turns up something ambiguous or far outside this task's scope, list it in the report rather than fixing everything found.
5. Do not touch broker credential/order-placement code (that's Phase 3), backtest cost-model code (Phase 5), or interfaces (Phase 6).
6. Write/update tests: market-cap bucket boundaries (if implemented) with real KRW figures, tick-size table against known price points, currency defaults, and timezone-sensitive logic with fixed inputs.

**Report:** status, the market-cap/tick-size sources used (or why they were left as `NotImplementedError`), what currency/timezone changes were made, test results, and the report file path.

---

## Phase 2 Results

Complete (Task 5 + Task 9). Domain model is KRX-native: `Ticker` (6-digit codes), `Exchange` (KOSPI/KOSDAQ/KONEX), tick-size table (`domain/services/tick_size.py`, cited 2023-01-25 unified table, reviewer-verified plausible), `Account`/`PaperBroker` currency = KRW, trading-window settings = KRX hours (KST). Market-cap buckets remain `NotImplementedError` — genuinely a cross-sectional/rank-based classification, not a per-stock KRW-threshold problem; out of scope until something needs it.

**No live credentials of any kind are available in this environment** (confirmed independently by Task 3 and Task 6's searches: no `.env` beyond `.env.example` anywhere in `gpt-quant-v2`, `gpt-quant-v2-krx`, or `kiwoom-client`). Per the operating directive for this whole plan, this is the one category of gap that stays open pending the user — everything buildable without live credentials should still get built, tested against mocks, and clearly marked where a real credential would need to close the loop.

## Task 10: KiwoomBroker implementation (Phase 3)

**Goal:** implement the broker adapter for Korean equities, per the plan's original Phase 3 section, working entirely from Task 3's source-reading spike (no live credentials available) — mocked at the `KiwoomAPI` boundary.

**Steps:**
1. Read Task 3's spike doc (`docs/spikes/003-kiwoom-mock-trading.md`) first — it already documents `kiwoom-client`'s actual shapes and flags a real ambiguity: its own README and examples **disagree** on order-request field names (`ord_dv`/`orig_ord_no` vs `trde_tp`/`org_ord_no`). Do not silently pick one and hope — either (a) support both via a small compatibility shim with a clear comment explaining why, or (b) pick the one that looks more authoritative (the actual `domestic/order.py` source, if it settles the question — check again, don't just trust the spike doc's summary) and add a loud `TODO`/comment + a report note that this must be verified against a real mock call before Phase 3 code goes anywhere near live capital. Do not guess silently.
2. Implement `src/gpt_quant_v2/infrastructure/brokers/kiwoom_broker.py`, a `KiwoomBroker` implementing whatever abstract interface `infrastructure/brokers/base.py` defines (`connect`/`get_account`/`get_positions`/`place_order`/`cancel_order`/`get_order_status`/`get_quote` or however it's actually named — read `base.py` first, don't assume method names). Map `kiwoom_client.KiwoomAPI`/`AsyncKiwoomAPI` calls (from `domestic/order.py`, `domestic/account.py`) into this interface. Since `kiwoom-client` has no schema/validation layer (raw dict in/out, per Task 3's finding), add explicit mapping/validation at this boundary — don't pass raw dicts through untyped.
3. Add `BrokerType.KIWOOM` to whatever enum currently has `PAPER`/`ALPACA`/`IBKR` (check Task 8's settings.py changes first — it may already be partially prepared), with `kiwoom_app_key`/`kiwoom_app_secret`/`is_mock` settings sourced from env vars, never hardcoded. Wire the broker factory (wherever `BrokerType` is switched on to construct a broker instance) to construct `KiwoomBroker` for `BrokerType.KIWOOM`.
4. Since there are no live credentials, all tests must mock `KiwoomAPI` at the boundary (don't attempt a live call — Task 3 already confirmed none is possible here). Write tests for: successful order placement mapping, the field-name ambiguity handling from step 1, error/failure response mapping, account/position/quote mapping.
5. Write a new ADR: `docs/adr/006-kiwoom-broker.md` — per the original plan's note, explicitly supersedes parts of ADR-005 (the abstraction is kept, but concrete adapters are now `PaperBroker` + `KiwoomBroker`, replacing Alpaca/IBKR); note the historical irony that the pre-v2 system was KIS-based and this is a return to a Korean broker via a different vendor. Check `docs/adr/005-abstract-broker.md` and the existing ADR index/README first for this repo's ADR format/conventions.
6. Do not touch `PaperBroker`'s cost-model/commission logic (that's Phase 5), and do not touch interfaces/API routers (Phase 6).

**Report:** status, how the field-name ambiguity was resolved (and the explicit flag that it needs real-credential verification before live use), what was implemented, test results, and the report file path.

---

## Phase 3 Results

Complete (Task 10, approved). `KiwoomBroker` implements `BrokerInterface`'s 8 methods, defensive raw-dict mapping, field-name ambiguity resolved (README variant, loud TODO), ADR-006 written. **Tracked follow-up (not yet fixed anywhere):** `BrokerSettings.type`'s `alias="broker_type"` bypasses `env_prefix`, so `GPTQ_BROKER_TYPE` cannot select any broker type via env var (only unprefixed `BROKER_TYPE` works) — dormant today because nothing yet calls `create_broker(settings.broker.type, ...)`, but whichever task first wires that call (likely Phase 6) must fix this alias bug first or broker selection will silently default to `paper` regardless of configuration.

## Task 11: Risk-gate service + signal-generator discipline (Phase 4a/4b)

**Goal:** the core research-driven deliverable — add sentiment/disclosure as a risk gate independent of the ML/technical signal path, and stop generating trading signals from sentiment alone.

**Steps:**
1. Read `domain/services/signal_generator.py`'s `SignalGenerator`/`SignalConfig`/`combine_signals()` and whatever `NewsBasedStrategy`-equivalent class currently generates signals from sentiment alone (grep for it — the plan's grounding section names it, confirm it still exists and where). Confirm the `sentiment_weight`/`ml_weight`/`technical_weight` (0.4/0.4/0.2) blend still exists post-Phases-1-3 changes.
2. Retire (or clearly deprecate — your call, document the reasoning) the sentiment-only signal path: any code path that produces a full BUY/SELL `TradingSignal` from sentiment alone, with no ML/technical corroboration, should either be removed or refactored to always route through `SignalGenerator.combine_signals()`. Deep-research finding #1 (folded into this plan's grounding) is explicit: sentiment alone doesn't reliably beat a benchmark, only combined signals do — this is the concrete code consequence of that finding.
3. New domain service, `domain/services/risk_gate.py`: given the latest sentiment for a ticker (from Task 7's `toss_news_client`/`dart_client` output) and, when available, a DART disclosure severity classification, it can force `REJECT`/`HALT_ENTRY`/`FORCE_EXIT` **independent of** whatever `TradingSignal` the ML/technical/composite path produced. Build a small disclosure-severity taxonomy for `dart_client.py`'s `disclosure_type` field (e.g. 관리종목지정/상장폐지/감사의견거절 = hard halt; routine filings = no gate) — this is genuinely new domain knowledge, not something to invent casually; research it (public regulatory/market-convention information, same as Task 9's tick-size research) and cite sources, or leave specific categories unclassified (default: no gate) rather than guessing severity.
4. Wire this gate into `domain/services/risk_manager.py` (`RiskManager`) as an additional check alongside whatever position/daily-loss checks already exist, and note in your report where `application/handlers/trade_execution.py` (or equivalent — check it exists and its actual name) would need to call it before `place_order` — you do NOT need to wire the actual call site into a live order-placement flow if none is currently active end-to-end (Phase 3's `KiwoomBroker` isn't wired into any live trading loop yet); wiring the `RiskManager` method itself and documenting the intended call site is sufficient for this task.
5. **Do not implement the sentiment-below-threshold auto-liquidate-existing-positions pattern uncritically.** A sibling project (scalp-it) independently pre-registered and offline-tested this exact pattern for its own strategy and it was REJECTED by real backtesting (net returns worsened, win rate barely moved, the triggering sample was too rare to matter) — that result is specific to scalp-it's own strategy shape (intraday pairs trading on surging small-caps) and doesn't mechanically transfer to gpt-quant-v2's different strategy shape, so don't skip this feature because of it, but DO scope this gate primarily as an **entry-blocker** (halt new entries on bad sentiment/disclosure) rather than a **forced-liquidator of existing positions** by default — reserve forced-exit for the hard-severity disclosure categories from step 3 (delisting, trading halt, embezzlement-class events), not generic negative sentiment scores. Document this design choice and its rationale in the report.
6. Do NOT retrain the XGBoost model or touch `feature_engineer.py`'s Korean-keyword swap in this task — that's Task 12/13 below, sequenced after this so the risk-gate architecture is settled first. Do NOT touch backtest code (Phase 5) or interfaces (Phase 6).
7. Write tests: risk-gate REJECT/HALT_ENTRY/FORCE_EXIT decisions for each severity category, the retired/refactored sentiment-only path no longer producing standalone signals, `RiskManager` integration.

**Report:** status, what happened to the sentiment-only signal path, the disclosure-severity taxonomy and its sources (or what was left unclassified and why), the entry-blocker-vs-liquidator design decision and rationale, test results, and the report file path.

## Task 12: Korean feature keywords + MCP/LLM computation discipline audit (Phase 4a cont'd / 4d)

**Goal:** finish the smaller, mechanical parts of Phase 4 — Korean-language feature keywords (English ones are currently useless) and an audit ensuring LLM/agent-facing tools never get to "compute" risk metrics themselves.

**Steps:**
1. In `application/ml/feature_engineer.py`, replace `DEFAULT_KEYWORDS` (English: `earnings`, `fda`, `guidance`, etc.) with Korean financial keywords relevant to KRX news/disclosures — 실적, 매출, 영업이익, 상한가, 하한가, 유상증자, 무상증자, 자사주매입, 감사의견, 관리종목, 거래정지, etc. (this list can reasonably overlap with Task 11's disclosure-severity taxonomy if one was built — check before duplicating). Replace `MARKET_OPEN`/`MARKET_CLOSE` constants with KRX hours if they're still hardcoded to US Eastern time here specifically (Task 9 already fixed `TradingSettings`, but `feature_engineer.py` may have its own separate constants — check, don't assume Task 9 covered this file).
2. **Do not retrain the XGBoost model in this task.** Retraining requires real KRX historical price/feature data at scale and a working backtest harness to validate against — that's Phase 5 territory (backtest rework) once it's done. Swapping keywords/constants here is a prerequisite, not the retraining itself; say so explicitly in the report so nobody mistakes "keywords swapped" for "model retrained."
3. Audit every MCP tool in `interfaces/mcp/tools.py` (`BacktestResult`, `Signal`, `TradeResult` and any other response models carrying numeric risk metrics — sharpe_ratio, max_drawdown_pct, sortino_ratio, win_rate, etc.): confirm each is populated ONLY from `BacktestHandler`/`RiskManager`/`SignalGenerator` code paths (or whatever this repo's actual handler names are — verify, don't assume), never left for an agent/LLM caller to "fill in" or recompute from raw trade lists via its own arithmetic. Add an explicit docstring/comment convention on these response models stating this constraint, for future editors.
4. Add a regression-guard test: for at least one MCP tool that returns a numeric risk metric, assert the returned value matches a directly-computed value from the same underlying handler (proving the tool isn't silently exposing raw data + a "compute this yourself" surface).
5. If the audit finds an existing violation (an MCP tool that DOES let the metric be agent-computed), fix it if it's a small, contained change; if fixing it requires larger restructuring, report DONE_WITH_CONCERNS with specifics rather than a large unplanned rewrite.

**Report:** status, the keyword/constant changes made, the MCP audit findings (clean, or what was fixed/flagged), test results, and the report file path.

---

## Phase 4 Results

Complete (Tasks 11-12). `domain/services/risk_gate.py` is the flagship deliverable — entry-blocker-primary, structurally verified (single `FORCE_EXIT` construction site, gated strictly on hard-severity disclosure taxonomy, never on sentiment alone) via an adversarial test. Sentiment-only signal path retired. Korean ML feature keywords in place; **XGBoost retraining itself is still not done** — it needs the backtest harness (this phase) to validate against, and real historical KRX price data (Task 6's `quant_airflow_db_client` can supply this). MCP computation-discipline audit complete; `BacktestResult`/`TrainingResult` now carry a caller-visible `is_placeholder` field since neither has a real engine behind it yet — which is exactly what this phase (5) builds.

## Task 13: KRX cost model + backtest market-data swap (Phase 5)

**Goal:** replace `application/handlers/backtest.py`'s `PolygonClient` dependency and flat US-style cost assumptions with a real KRX cost model and Task 6's read-only market-data client — this is the task that turns `BacktestResult` from a hardcoded placeholder into a genuine computation.

**Steps:**
1. Read `application/handlers/backtest.py` in full first. Replace its `PolygonClient` dependency with Task 6's `quant_airflow_db_client.py` (adjusted daily bars) for historical price data — this was flagged as dead/unwired in Task 8's report, so this is the first task that actually wires it into a live handler path.
2. Build `domain/services/cost_model.py` (or similar — check for a natural existing home first): a Korean transaction-cost model covering (a) 증권거래세 (securities transaction tax, sell-side only — research the actual current KOSPI/KOSDAQ rate; this is public regulatory information, same research approach as Tasks 9/11, cite your source, and if you can't confirm current rates confidently, use a clearly-labeled placeholder rate with a prominent comment rather than a silently-wrong number), (b) 위탁수수료 (brokerage commission — use a clearly-labeled representative default since Task 10's `KiwoomBroker` has no live fee schedule to read from; make it configurable, not hardcoded deep in logic), (c) slippage calibrated by market-cap/liquidity tier if reasonably implementable (KOSDAQ small-caps slip more than KOSPI large-caps) — if a full liquidity-tier model is too much for this task, a simpler placeholder (e.g. a flat rate with a comment flagging it needs calibration against real historical spreads later) is acceptable, documented honestly rather than presented as calibrated.
3. Reuse this cost model in `infrastructure/brokers/paper_broker.py` (currently has its own separate, unconfirmed commission/slippage defaults per Task 9's report) so paper trading and backtesting share one source of truth rather than two hand-maintained numbers.
4. Wire `BacktestHandler` (or whatever the actual class/function is — verify the name) to actually compute `sharpe_ratio`/`sortino_ratio`/`max_drawdown_pct`/`win_rate`/`profit_factor` from real simulated trades using the new price source and cost model, and set `BacktestResult.is_placeholder = False` once real computation replaces the hardcoded stub in `interfaces/mcp/tools.py`'s `backtest()` MCP tool function.
5. Do NOT retrain the XGBoost model in this task either — that's a follow-up once this backtest harness exists and can validate a retrained model's output; say so explicitly if you get close to that boundary and stop.
6. Do NOT touch interfaces/frontend beyond the one `BacktestResult.is_placeholder` flag flip described in step 4 — broader interface work is Phase 6.
7. Write tests: cost-model unit tests against known price/quantity inputs (tax + commission + slippage arithmetic), a backtest run against real historical KRX bars (e.g. 005930) producing genuinely computed (not hardcoded) metrics, and a regression test confirming `BacktestResult.is_placeholder` is `False` when the real engine runs.

**Report:** status, the cost-model rates/sources used (or what was left as an honestly-labeled placeholder and why), what got wired into `backtest.py`/`paper_broker.py`, test results (including whether a real backtest run against real KRX data was actually executed), and the report file path.

---

## Phase 5 Results

Complete (Task 13, approved). `BacktestResult` is no longer a placeholder: `backtest.py` uses Task 6's `QuantAirflowDBClient` for real historical bars, `domain/services/cost_model.py`'s `KoreanCostModel` (0.20% 증권거래세, cited; commission/slippage honestly labeled unconfirmed placeholders) is shared by both the backtest engine and `PaperBroker`, and a real backtest run against 005930 was independently reproduced byte-for-byte by the reviewer. Full-suite baseline corrected and confirmed: 40 failed/293 passed/5 skipped, zero regressions across the whole plan so far. Three genuinely pre-existing broken handler imports (`news_analysis.py`, `signal_generation.py`, `trade_execution.py` — reference nonexistent classes, confirmed broken before this plan started) are isolated (not fixed) behind a loud, narrowly-scoped `try/except ImportError` in `application/handlers/__init__.py` — tracked as follow-up debt, not blocking.

## Task 14: Interfaces cleanup — API validators, broker wiring, MCP language (Phase 6, backend half)

**Goal:** close out the accumulated interface-layer debt from Tasks 5, 8, and 10, and finish the MCP-facing language cleanup Task 12 didn't fully cover.

**Steps:**
1. **Fix the two US-ticker `isalpha()` gates** flagged in Task 5's review: `interfaces/api/routers/portfolio.py:318` and `interfaces/api/routers/news.py:194` (line numbers may have drifted across 8 subsequent tasks — search fresh, don't trust stale line numbers) currently reject any real 6-digit KRX ticker with a "Must be 1-5 uppercase letters" error. Update both to accept KRX's `^\d{6}$` format (reuse the domain `Ticker` value object's validation logic if these routers can cleanly import it, rather than re-deriving a second regex).
2. **Fix the `BrokerSettings.type`/`alias="broker_type"` bug** flagged in Task 10's review: pydantic-settings' `alias` bypasses `env_prefix` entirely, so `GPTQ_BROKER_TYPE` currently cannot select any broker type (only the unprefixed `BROKER_TYPE` works). This was dormant because nothing called `create_broker(settings.broker.type, ...)` yet — check whether anything now does (Task 13 didn't touch broker wiring, so this may still be unwired; if truly nothing calls it yet, fix the alias bug anyway since it's cheap and this is exactly the phase where interface-layer config bugs get closed out) and confirm `GPTQ_BROKER_TYPE=kiwoom` genuinely selects `KiwoomBroker` after the fix, with a test.
3. **Correct ADR-006's env var documentation** (flagged as a Minor in Task 10's review): it currently documents `GPTQ_BROKER__KIWOOM_APP_KEY`/`GPTQ_BROKER__KIWOOM_APP_SECRET` (nested double-underscore) but the actual implementation uses flat `GPTQ_KIWOOM_APP_KEY`/`GPTQ_KIWOOM_APP_SECRET` — a one-line fix in `docs/adr/006-kiwoom-broker.md`.
4. **Update MCP tool docstrings/descriptions in `interfaces/mcp/tools.py`** that still reference US concepts (tickers as letters, "shares", USD amounts) to KRX-equivalent language (6-digit codes, 주, KRW/원) — Task 12 already fixed the computation-discipline issue here; this is the remaining cosmetic-but-accuracy-relevant language pass. Re-verify the `is_placeholder` discipline from Task 12 still holds after these edits (don't accidentally regress it while touching descriptions).
5. **Quick pass over `interfaces/websocket/*`** (if it exists — check) for any US-specific assumptions; Task 3's original grounding notes flagged this as "likely low-risk, mostly pass-through, quick pass needed" — confirm or fix, don't assume clean without checking.
6. Do NOT touch the three broken handlers (`news_analysis.py`/`signal_generation.py`/`trade_execution.py`) — that's separate, already-tracked debt, not this task's job. Do NOT touch the frontend (that's Task 15, next).

**Report:** status, what was fixed in each of the 5 areas above, test results (including the `GPTQ_BROKER_TYPE=kiwoom` selection test), and the report file path.

## Task 15: Frontend cosmetic pass (Phase 6, frontend half)

**Goal:** the low-risk, presentational remainder of Phase 6 — the plan's original grounding assessed the frontend as "presentational/generic (ticker as free-text, BUY/SELL/ALL), low risk, cosmetic Korean-market labeling changes only."

**Steps:**
1. In `frontend/src/pages/` (`SignalsPage.tsx`, `DashboardPage.tsx`, `PortfolioPage.tsx`, `BacktestPage.tsx`, or whatever the actual current file names are — check first, they may have changed), update currency formatting from `$`/USD conventions to `₩`/KRW where hardcoded.
2. Confirm ticker input fields don't have a client-side validation pattern hardcoded to the old US format (1-5 uppercase letters) that would reject real KRX codes — fix if found, matching Task 14's backend validator fix.
3. This is explicitly a low-risk, minimal task per the plan's own grounding assessment — do not restructure components, add new pages, or redesign layout. If the frontend build/type-check step exists in this repo (check `frontend/package.json` for a `build`/`typecheck` script), run it to confirm nothing broke.
4. If no currency/ticker-format hardcoding is actually found (i.e. the frontend was already generic enough), say so plainly in the report rather than inventing changes to justify the task.

**Report:** status, what was changed (or confirmation nothing needed changing, with evidence), build/typecheck results if applicable, and the report file path.

---

## Phase 6 Results

Complete (Tasks 14-15). All tracked interface-layer debt closed: both `isalpha()` gates + 4 sibling length constraints fixed in `portfolio.py`/`news.py` (reusing domain `Ticker`), `BrokerSettings.type` env-prefix bug fixed and end-to-end tested (`GPTQ_BROKER_TYPE=kiwoom` → `KiwoomBroker` via unmocked `create_broker()`), ADR-006 corrected, MCP tool language KRX-ified without touching Task 12's `is_placeholder` discipline, frontend currency `$`→`₩`. **Remaining known, deliberately-untouched debt**: `signals.py`/`trades.py` still carry the same `min_length`/`max_length` US-format constraint pattern (same bug class, different files, never in any task's named scope) — track for whenever those files are next touched, not urgent since no live trading loop calls them yet.

## Task 16: Local rename mechanics (Phase 7, prep only — no GitHub rename)

**Goal:** do all the local, worktree-contained rename work so the repo is ready to become its new identity, WITHOUT executing the actual GitHub repository rename — that's a real, semi-irreversible action affecting a public repo's URL/identity and needs the user's explicit go-ahead, tracked as an open item alongside the OpenAI API key gap from Task 1.

**Chosen name (provisional, pending user confirmation): `krx-signal-engine`.** Rationale, from Task 4's own findings: this repo is confirmed to sit downstream of `quant-airflow` (batch data) and alongside `kr-quant` (research) as the live trading/signal-generation component — `krx-signal-engine` is the plan's own candidate that "emphasizes it's the signal-generation/execution engine of the stack, distinguishing it from krx-news-client (data) and kr-quant (research) — clearest fit if this repo is meant to sit downstream of quant-airflow/kr-quant," which Task 4 confirmed is exactly the architecture. The other two candidates (`kr-news-quant`, `toss-alpha`) remain on record in this plan's earlier section if the user prefers a different name — this task's renames are mechanical and can be re-run with a different slug if the user picks otherwise.

**Steps:**
1. Rename the Python package: `src/gpt_quant_v2/` → `src/krx_signal_engine/` (or the confirmed final slug), updating every internal import (`from gpt_quant_v2...` → `from krx_signal_engine...`) across `src/`, `tests/`, `scripts/`. This is a large mechanical rename — a careful find-and-replace plus a full test-suite run to confirm nothing broke is the right approach, not a manual file-by-file edit.
2. Update `pyproject.toml`: `name = "gpt-quant-v2"` → the new slug, `[project.scripts]`/entry points if any reference the old package name, `packages`/`include` paths if explicitly listed.
3. Update `docker/docker-compose.yml` and `Dockerfile`: container names, network name, any `gpt_quant`/`gpt-quant-v2` string throughout.
4. Update `.github/workflows/*.yml`: workflow names/job names referencing the old name (cosmetic, doesn't need to be exhaustive — CI will still function either way, this is about not leaving a confusing old name in workflow UI).
5. Update `README.md` and `docs/adr/README.md`/ADR index for the new name; add a note (in README, one paragraph) that this repo was redeveloped from `gpt-quant-v2` (link back for provenance) per the plan's own suggestion.
6. Do NOT rename the actual GitHub repository (no `gh repo rename`, no remote URL change). Do NOT touch `origin`'s remote config. Do NOT push anything that asserts the new identity is final/live — this task's commits stay on the `feat/krx-toss-redevelopment` branch like everything else in this plan.
7. Run the full test suite after the package rename to confirm the mechanical rewrite didn't silently break imports (expect the same 40-failure baseline, zero new regressions — the rename itself shouldn't touch test logic, just import paths).

**Report:** status, confirmation of the chosen provisional name and rationale, what was renamed, test results (full suite, confirming the established 40-failure baseline held), and an explicit reminder in the report that the actual GitHub rename is NOT done and needs a separate, explicit go-ahead — same category of open item as Task 1's pending API key.

