# KRX Signal Engine

**Korean-Equity (KOSPI/KOSDAQ) Trading System with a News/Disclosure Risk Gate**

A clean-architecture trading system for Korean equities: a read-only market-data
reader over a sibling `quant-airflow` TimescaleDB, a Kiwoom Securities broker
adapter, and a risk gate that blocks entries (and force-exits on hard-severity
DART disclosures) independent of the ML/technical signal path.

> **Provenance note:** this repository was redeveloped from
> [`gpt-quant-v2`](https://github.com/younghwan91/gpt-quant-v2), re-platformed
> from a US-equity system into a Korean-equity (KOSPI/KOSDAQ) system. See
> `PLAN.md` for the full 8-phase redevelopment plan and its results, and
> `PLAN.md`'s Phase 7 Results section for the consolidated list of what's
> still open. `krx-signal-engine` is a **provisional** name pending final
> confirmation -- the GitHub repository itself has not yet been renamed.

## 🎯 What's real vs. still a stub (read this before trusting a number)

This system was built in phases, and not everything a caller sees is fully
wired yet. Honesty here matters more than a feature list:

- **Risk gate** (`domain/services/risk_gate.py`): real. Classifies DART
  disclosures against a researched hard-severity taxonomy (delisting,
  관리종목지정, 매매거래정지, 감사의견거절, 횡령/배임, 계속기업불확실성) and
  blocks/force-exits independent of the ML/technical signal. Sentiment alone
  can only ever block new entries, never force an exit -- structurally
  enforced and adversarially tested. Now wired into `execute_trade`'s MCP
  tool via `RiskManager.validate_trade` before any order is placed.
- **Backtest engine**: real data (via `quant_airflow_db_client.py`, read-only
  off `quant-airflow`'s TimescaleDB) and a real Korean cost model (증권거래세,
  위탁수수료, KOSPI/KOSDAQ tick-size-aware slippage) -- but the *strategy*
  being backtested is still a demo volume-momentum stub, not this system's
  actual signal-generation logic. `BacktestResult.strategy_is_placeholder`
  tells you which is which; don't read a Sharpe ratio here as validation of
  the news-driven strategy.
- **News ingestion** (`toss_news_client.py`/`dart_client.py`/`news_poller.py`):
  built and tested (poll+dedup, circuit-breaker fallback to DART), but **not
  yet wired into the MCP tools** -- `get_news`/`get_price_history` raise
  `NotImplementedError` rather than silently serving stale US-market data.
- **Sentiment accuracy**: unvalidated. A Korean FinBERT checkpoint was
  benchmarked and scored only 60.5% accuracy / 18.2% BEARISH recall on a
  76-article hand-labeled sample -- not usable for a risk gate as-is. The
  OpenAI comparison leg is implemented but has never been run (no API key
  available in this environment). See `PLAN.md`'s Phase 0 Results.
- **Broker** (`kiwoom_broker.py`): implemented and unit-tested against a
  mocked `KiwoomAPI`, but never exercised against a live/mock Kiwoom
  connection -- no credentials have ever been available in this environment.
  A field-name ambiguity in `kiwoom-client` itself (its README and its
  examples disagree on order-request field names) is resolved by a documented
  best guess with a `TODO(verify-before-live-capital)` -- do not use for real
  orders until that's confirmed against a real mock call.
- **XGBoost model**: not retrained. Korean feature keywords are in place, but
  retraining needs the backtest harness (done) validated against a real
  strategy (not done) first.

## 🏗️ Architecture

## 🏗️ Architecture

```
src/krx_signal_engine/
├── domain/           # Core business logic (no external dependencies)
├── application/      # Use cases and orchestration
├── infrastructure/   # External integrations (DB, APIs, brokers)
├── interfaces/       # Entry points (API, CLI, MCP tools)
└── shared/           # Cross-cutting concerns
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- UV package manager (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/younghwan91/gpt-quant-v2.git  # placeholder: repo not yet renamed to krx-signal-engine on GitHub, see Task 16 report
cd krx-signal-engine

# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Start infrastructure
docker compose up -d

# Run database migrations
alembic upgrade head

# Start the API server
uvicorn krx_signal_engine.interfaces.api.main:app --reload
```

### Using the CLI

```bash
# Analyze a ticker (6-digit KRX code, e.g. 005930 = 삼성전자)
gptq analyze 005930

# Get current signals
gptq signals --min-confidence 0.7

# Run backtest (strategy is currently a demo stub -- see "What's real" above)
gptq backtest --start 2024-01-01 --end 2024-03-01

# Start real-time trading (paper mode)
gptq trade --paper
```

## 🤖 AI Agent Integration (MCP)

This system exposes MCP tools for AI agents. `get_news`/`get_price_history`
currently raise `NotImplementedError` (the Toss/DART news source isn't wired
into the MCP context yet -- see "What's real" above); the rest are live:

```python
# Available MCP tools:
- analyze_ticker(ticker: str) -> TickerAnalysis   # 6-digit KRX code
- get_signals(min_confidence: float) -> List[Signal]
- execute_trade(signal_id: str) -> TradeResult    # now routed through RiskManager.validate_trade
- backtest(strategy: str, start: str, end: str) -> BacktestResult  # real engine, demo strategy -- check strategy_is_placeholder
- get_portfolio() -> Portfolio
- get_system_status() -> SystemStatus
- get_news(...) -> NotImplementedError            # not yet wired
- get_price_history(...) -> NotImplementedError    # not yet wired
```

## 📊 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/krx_signal_engine --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests
pytest tests/e2e/          # End-to-end tests
```

## 📁 Project Structure

```
krx-signal-engine/
├── src/krx_signal_engine/
│   ├── domain/
│   │   ├── entities/        # Stock, News, Signal, Trade
│   │   ├── value_objects/   # Price, Volume, Ticker
│   │   ├── services/        # SignalGenerator, RiskManager, RiskGate, TickSize, CostModel
│   │   └── events/          # Domain events
│   ├── application/
│   │   ├── commands/        # Write operations
│   │   ├── queries/         # Read operations
│   │   └── handlers/        # Command/Query handlers
│   ├── infrastructure/
│   │   ├── persistence/     # Database repositories
│   │   ├── messaging/       # Redis pub/sub
│   │   ├── external/        # API clients
│   │   └── brokers/         # Trading broker interfaces
│   ├── interfaces/
│   │   ├── api/             # FastAPI endpoints
│   │   ├── cli/             # Typer CLI
│   │   ├── mcp/             # AI agent tools
│   │   └── websocket/       # Real-time updates
│   └── shared/
│       ├── config/          # Settings management
│       ├── logging/         # Structured logging
│       └── errors/          # Exception hierarchy
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── frontend/                # React dashboard
├── docker/                  # Docker configurations
├── docs/                    # Documentation
└── scripts/                 # Utility scripts
```

## ⚙️ Configuration

All configuration is managed via environment variables (`GPTQ_` prefix,
mostly flat -- see `shared/config/settings.py` for the exact field names).
See `.env.example` for the full list.

Key configurations:
- `GPTQ_DATABASE_URL`: TimescaleDB connection string (this repo's own tables)
- `GPTQ_REDIS_URL`: Redis connection string
- `GPTQ_BROKER_TYPE`: `paper` or `kiwoom`
- `GPTQ_KIWOOM_APP_KEY` / `GPTQ_KIWOOM_APP_SECRET` / `GPTQ_KIWOOM_IS_MOCK`: Kiwoom broker credentials (`is_mock` defaults `True`)
- `GPTQ_KRX_NEWS__DART_API_KEY`: DART Open API key, for `dart_client.py`'s disclosure feed
- `GPTQ_QUANT_AIRFLOW_DB__*`: read-only DSN for `quant-airflow`'s TimescaleDB (batch/EOD market data -- see `docs/spikes/004-data-layer-scope.md`)
- `GPTQ_OPENAI_API_KEY`: OpenAI API key, used by the sentiment-analysis prompt (accuracy unvalidated -- see above)
- `GPTQ_POLYGON_API_KEY` / `GPTQ_TIINGO_API_KEY`: dead. These were the original US-market vendors; the settings classes were removed in the redevelopment. Present here only as a note that they no longer do anything.

## 📈 Trading Strategy

**Currently, this is two separate things that are not yet connected** (see
"What's real" above): a real `SignalGenerator`/`RiskGate` pipeline that has
never been backtested end-to-end, and a real backtest engine that only ever
tests a demo momentum stub. The intended design, once wired together:

1. **News Collection**: Toss (via `krx-news-client`) + DART disclosures, polled by `news_poller.py` with circuit-breaker fallback -- built, not yet wired into the MCP-facing tools
2. **Sentiment Analysis**: Korean-language sentiment scoring (accuracy unvalidated -- see above)
3. **Risk Gate**: `domain/services/risk_gate.py` blocks entries on bad sentiment/disclosures, force-exits only on hard-severity disclosures -- real, and now wired into `execute_trade`
4. **Feature Engineering**: Korean financial keywords + KRX trading-hours features (`feature_engineer.py`)
5. **ML Prediction**: XGBoost predicts direction/magnitude -- model itself not yet retrained on Korean data
6. **Signal Generation**: `SignalGenerator.combine_signals()` blends sentiment/ML/technical weights (0.4/0.4/0.2) -- structurally refuses to emit a signal from sentiment alone
7. **Trade Execution**: `execute_trade` MCP tool -> `RiskManager.validate_trade` -> `KiwoomBroker.place_order` (paper or, once credentials exist, real)

## 🛡️ Risk Management

- `domain/services/risk_gate.py`: sentiment/disclosure-driven entry blocks and hard-severity force-exits, independent of the ML/technical signal
- `domain/services/tick_size.py`: KRX price-band tick-size rounding, applied at Kiwoom order construction and backtest/paper-broker fills
- `domain/services/cost_model.py`: KRX transaction costs (증권거래세 -- rate cited but **not confirmed against a primary source**, treat as an educated placeholder like commission/slippage)
- Position sizing based on account risk percentage
- Maximum position limits per ticker
- Trading window restrictions (KRX hours, 09:00-15:30 KST)
- Daily loss limits

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
