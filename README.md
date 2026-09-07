# KRX Signal Engine

**AI-Agent-Friendly News-Driven Algorithmic Trading System**

A modern, clean-architecture trading system designed for AI agent integration, featuring news sentiment analysis, ML-based signal generation, and automated trade execution.

> **Provenance note:** this repository was redeveloped from
> [`gpt-quant-v2`](https://github.com/yourusername/gpt-quant-v2), re-platformed
> from a US-equity system into a Korean-equity (KOSPI/KOSDAQ) system driven by
> Toss news. `krx-signal-engine` is a **provisional** name pending final
> confirmation -- the GitHub repository itself has not yet been renamed (see
> the plan's Task 16 report for the open item).

## 🎯 Features

- **News-Driven Trading**: Analyze financial news sentiment using OpenAI
- **ML Signal Generation**: XGBoost models predict price movements
- **AI-Agent Ready**: MCP tool interface for seamless AI integration
- **Real-Time Processing**: WebSocket updates for live market data
- **Comprehensive Backtesting**: Historical strategy validation
- **Clean Architecture**: Domain-driven design with clear boundaries

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
git clone https://github.com/yourusername/krx-signal-engine.git
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
# Analyze a ticker
gptq analyze AAPL

# Get current signals
gptq signals --min-confidence 0.7

# Run backtest
gptq backtest --start 2024-01-01 --end 2024-03-01

# Start real-time trading (paper mode)
gptq trade --paper
```

## 🤖 AI Agent Integration (MCP)

This system exposes MCP tools for AI agents:

```python
# Available MCP tools:
- analyze_ticker(ticker: str) -> TickerAnalysis
- get_signals(min_confidence: float) -> List[Signal]
- execute_trade(signal_id: str) -> TradeResult
- backtest(strategy: str, start: str, end: str) -> BacktestResult
- get_portfolio() -> Portfolio
- get_system_status() -> SystemStatus
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
│   │   ├── services/        # TradingStrategy, RiskManager
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

All configuration is managed via environment variables. See `.env.example` for available options.

Key configurations:
- `GPTQ_DATABASE_URL`: TimescaleDB connection string
- `GPTQ_REDIS_URL`: Redis connection string
- `GPTQ_POLYGON_API_KEY`: Polygon.io API key
- `GPTQ_OPENAI_API_KEY`: OpenAI API key
- `GPTQ_TIINGO_API_KEY`: Tiingo API key

## 📈 Trading Strategy

The system implements a news-driven momentum strategy:

1. **News Collection**: Fetch recent news from Tiingo API
2. **Sentiment Analysis**: Analyze news sentiment using OpenAI
3. **Feature Engineering**: Combine news + price data features
4. **ML Prediction**: XGBoost predicts max rise, drawdown, and direction
5. **Signal Generation**: Create signals based on risk/reward ratio
6. **Trade Execution**: Execute via broker interface (paper or real)

## 🛡️ Risk Management

- Position sizing based on account risk percentage
- Maximum position limits per ticker
- Stop-loss and take-profit automation
- Trading window restrictions (market hours only)
- Daily loss limits

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
