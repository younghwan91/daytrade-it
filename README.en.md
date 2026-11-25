<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/brand/wordmark-dark.svg">
  <img alt="daytrade-it" src="docs/images/brand/wordmark-light.svg" height="52">
</picture>

# daytrade-it

**KOSPI/KOSDAQ news day-trading system, and the research that measures whether it has an edge.**

Reads Toss news and DART disclosures, judges each item with Anthropic Claude Haiku, and validates orders through a disclosure risk gate and Kiwoom broker adapter.

[![CI](https://github.com/younghwan91/daytrade-it/actions/workflows/ci.yml/badge.svg)](https://github.com/younghwan91/daytrade-it/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Guide](docs/guide.md) · [Research](docs/research/README.md) · [MCP tools](docs/mcp/README.md) · [한국어](README.md)

</div>

> **Status — real-money trading is stopped.** Live entries have been halted since 2026-09-17
> (kill switch `data/live_STOP`); the daemon only records judgments and a paper ledger.
> The research in [`docs/research/`](docs/research/README.md) found **no edge over round-trip costs
> (about 36.7 bp)** for this way of trading news. Closed hypotheses, pre-registrations and the forward
> gate are indexed there. Treat this repository as a research record and a reference
> implementation, not as a profitable strategy.

## Key results

All from [`docs/research/README.md`](docs/research/README.md); negative results included.

- **Verdict: no edge over the ~36.7 bp round-trip cost.** Selection, entry timing, exits, latency, ranking and pyramiding were each closed at least once.
- **Forward gate: 3 of 7 conditions pass, `promote=false`.** The +38.8 bp/trade mean comes from one trade (+3,022 bp); without the top 1% it is -16.4 bp; Deflated Sharpe 0.005 (55 trades, 2026-10-05).
- **Ranking works, money does not.** Stage-1 confirm IC 0.132 (t=8.3) but the economic curve is negative everywhere; the pre-registered stage 2/3 entry/exit timing failed to beat baseline ([result](docs/research/2026-10-05-stage23-result.md)).
- **Agent-loop search over ~1,000 structures (autoresearch) rejected:** net -2.4 bp, alpha concentrated in 1.4% of trades and one ticker ([doc](docs/research/2026-10-24-autoresearch.md)).
- **One robust non-news fact:** over 11 years overnight +25.7 bp/day, intraday -28.6 bp/day.

## Try it now (no keys, no DB)

```bash
git clone https://github.com/younghwan91/daytrade-it.git && cd daytrade-it && uv sync --extra dev
uv run python scripts/demo_cost_wall.py 10000   # round-trip cost in bp of a flat 1-share trade
uv run pytest tests/unit --no-cov -q            # 771 unit tests, ~10 s
```

Checked 2026-11-26: KOSPI 33.0 bp, KOSDAQ 53.0 bp at 10,000 KRW (model defaults; the 36.7 bp above is the realized average).

## Full setup

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

## Documentation

| Doc | What's inside |
|---|---|
| [Project guide](docs/guide.md) | Architecture, configuration, running tests |
| [`docs/research/README.md`](docs/research/README.md) | Research index, forward gate, closed hypotheses |
| [`docs/adr/`](docs/adr/README.md) | Architecture decisions |
| [`docs/mcp/README.md`](docs/mcp/README.md) | MCP tools |
| [`PLAN.md`](PLAN.md) | The original migration plan (history; may be out of date) |

## Sibling repositories

| Repo | Role |
|---|---|
| [`krx-news-client`](https://github.com/younghwan91/krx-news-client) | Toss/DART news scraping client library |
| [`kiwoom-client`](https://github.com/younghwan91/kiwoom-client) | Kiwoom Securities REST API client |
| [`krx-quant-core`](https://github.com/younghwan91/krx-quant-core) | KRX market rules, costs, risk gate, execution constants |
| [`quant-airflow`](https://github.com/younghwan91/quant-airflow) | Batch ingestion into TimescaleDB, read here |
| [`swing-it`](https://github.com/younghwan91/swing-it) | KOSPI/KOSDAQ research, no live trading |

## Tools by the same author

| Repo | One line |
|---|---|
| [`kiwoom-client`](https://github.com/younghwan91/kiwoom-client) | Python client for the Kiwoom Securities REST API |
| [`fin-checkup`](https://github.com/younghwan91/fin-checkup) | MCP server that shows Korean listed-company financials as traffic lights from DART filings |
| [`scalp-it`](https://github.com/younghwan91/scalp-it) | Scalping research and execution repo sharing the same account |

## License

MIT — see [LICENSE](LICENSE).
