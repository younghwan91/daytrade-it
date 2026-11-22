<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/brand/wordmark-dark.svg">
  <img alt="daytrade-it" src="docs/images/brand/wordmark-light.svg" height="56">
</picture>

**English** | [한국어](README.ko.md)

**KOSPI/KOSDAQ news day-trading system, and the research that measures whether it has an edge.**

[![CI](https://github.com/younghwan91/daytrade-it/actions/workflows/ci.yml/badge.svg)](https://github.com/younghwan91/daytrade-it/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

The system reads Toss news and DART disclosures, has an LLM (Anthropic Claude Haiku) judge each item,
buys one share at a limit price and exits at 15:00 KST. Orders go through a Kiwoom Securities broker adapter
and a disclosure-based risk gate. Market data is read, read-only, from a sibling
[`quant-airflow`](https://github.com/younghwan91/quant-airflow) TimescaleDB.

> **Status — real-money trading is stopped.** Live entries have been halted since 2026-09-17
> (kill switch `data/live_STOP`); the daemon only records judgments and a paper ledger.
> The research in [`docs/research/`](docs/research/README.md) found **no edge over round-trip costs
> (about 36.7 bp)** for this way of trading news. Closed hypotheses, pre-registrations and the forward
> gate are indexed there. Treat this repository as a research record and a reference
> implementation, not as a profitable strategy.

## What is in here

| Area | What it does |
|---|---|
| **Risk gate** (`domain/services/risk_gate.py`) | Classifies DART disclosures (delisting, trading halt, audit-opinion refusal, embezzlement, ...) and blocks entries or forces exits, independent of the signal. Sentiment alone can only block entries, never force an exit. |
| **News judging** (`infrastructure/external/`) | Toss and DART pollers, plus an Anthropic client that judges news with prompt caching. |
| **Signals and trading** (`application/`, `scripts/news_signal_poller.py`) | Turns judgments into entries and exits; one-share limit orders, price band, per-day and per-stock caps, paper mode. |
| **Broker** (`infrastructure/brokers/`) | Kiwoom limit orders, plus paper and simulated brokers. Market orders are refused (the market-order code is unverified). |
| **Backtest / replay** | Replays the live trading code over historical bars with the Korean cost model; runs on a designated host only (a guard in `krx-quant-core`). |
| **Research** (`scripts/eval/`, `docs/research/`) | Measurement scripts, pre-registrations (`docs/research/frozen/`) and written verdicts. |
| **Interfaces** | `gptq` CLI, FastAPI app, MCP server for AI agents (`docs/mcp/README.md`). |

Market-generic logic (tick sizes, cost model, disclosure risk gate, Kiwoom constants) lives in
[`krx-quant-core`](https://github.com/younghwan91/krx-quant-core); `risk_gate.py`, `cost_model.py` and
`tick_size.py` here are thin re-exports.

## Architecture

```
src/daytrade_it/
├── domain/          # entities, value objects, services (RiskGate, RiskManager, cost, tick size)
├── application/     # commands/queries/handlers, signals, trading, replay, news, ml
├── infrastructure/  # external (Toss, DART, Anthropic, quant-airflow DB), brokers, persistence
├── interfaces/      # api (FastAPI), cli (Typer), mcp, websocket
└── shared/          # config, logging, errors
```

Every order passes `RiskManager.validate_trade()` before it can reach `KiwoomBroker.place_order()`.
On a BUY the risk gate looks at the ticker's most recent hard-severity disclosure of the last 7 days; if its
data file is missing or stale, the entry is refused rather than waved through.

## Quick start

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
git clone https://github.com/younghwan91/daytrade-it.git
cd daytrade-it

uv sync --extra dev
cp .env.example .env        # fill in your own DSNs and API keys

docker compose -f docker/docker-compose.yml up -d   # TimescaleDB / Redis
uv run alembic upgrade head
uv run uvicorn daytrade_it.interfaces.api.main:app --reload   # Swagger UI at /docs
```

```bash
uv run gptq analyze 005930          # 6-digit KRX code
uv run gptq signals --min-confidence 0.7
uv run gptq trade --paper
```

### Configuration

Settings are read from environment variables with the `GPTQ_` prefix; `.env.example` lists all of them
(`shared/config/settings.py` has the exact field names). Credentials (Kiwoom app key/secret, DART key,
Anthropic key, database DSNs) are supplied only through the environment. Never commit a filled `.env`.
`GPTQ_BROKER_TYPE` is `paper` or `kiwoom`, and the Kiwoom mock flag defaults to on.

## Tests

```bash
uv run pytest tests/unit tests/integration --no-cov -q
uv run ruff check src tests
```

`tests/integration/` has a few tests that need a reachable `quant-airflow` database and skip cleanly
without one. `tests/e2e/` needs a running API server (`E2E_BASE_URL`, default `http://localhost:8000`).

## Documentation

- [`docs/research/README.md`](docs/research/README.md) — research index, forward gate, closed hypotheses
- [`docs/adr/`](docs/adr/README.md) — architecture decisions
- [`docs/mcp/README.md`](docs/mcp/README.md) — MCP tools
- [`PLAN.md`](PLAN.md) — the original migration plan (history; may be out of date)

Provenance: redeveloped from `gpt-quant-v2` (a US-equity system, since removed) into a Korean-equity system.

## Sibling repositories

| Repo | Role |
|---|---|
| [`krx-news-client`](https://github.com/younghwan91/krx-news-client) | Toss/DART news scraping client library |
| [`kiwoom-client`](https://github.com/younghwan91/kiwoom-client) | Kiwoom Securities REST API client |
| [`krx-quant-core`](https://github.com/younghwan91/krx-quant-core) | KRX market rules, costs, risk gate, execution constants |
| [`quant-airflow`](https://github.com/younghwan91/quant-airflow) | Batch ingestion into TimescaleDB, read here |
| [`swing-it`](https://github.com/younghwan91/swing-it) | KOSPI/KOSDAQ research, no live trading |

## License

MIT — see [LICENSE](LICENSE).
