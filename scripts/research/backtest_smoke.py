"""Nightly smoke: the krx-quant-core-backed MCP backtest over the last 20 trading days.

Runs `BacktestHandler` (core `run_replay` underneath) for five liquid KRX
tickers over the last 20 sessions that have bars in quant-airflow's
`daily_bars_adjusted`, and prints the MCP `backtest` response as JSON (same
keys, validated through `interfaces.mcp.tools.BacktestResult`).

It calls the handler directly rather than the `backtest` MCP tool so a
nightly smoke doesn't add a trial to the `research/runs` ledger. Run by
`kqc nightly .` from `research/nightly.toml` on simnode; exits non-zero if
the DB can't be read or no bars come back.

    uv run python scripts/research/backtest_smoke.py
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from daytrade_it.application.handlers.backtest import BacktestCommand, BacktestHandler
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.interfaces.mcp.tools import BacktestResult

TICKERS = ["005930", "000660", "035420", "005380", "051910"]
SESSIONS = 20
STRATEGY = "news_momentum"
INITIAL_CAPITAL = Decimal("10000000")


def db_kwargs(env: dict[str, str] | os._Environ[str]) -> dict[str, Any]:
    """Connection args from ``KR_QUANT_DB`` (the DSN cron exports from quant-airflow/.env).

    Empty when it isn't set -- the client then uses ``GPTQ_QUANT_AIRFLOW_DB_*``.
    """
    dsn = env.get("KR_QUANT_DB")
    if not dsn:
        return {}
    url = urlsplit(dsn.strip("\"'"))
    return {
        "host": url.hostname,
        "port": url.port,
        "dbname": url.path.lstrip("/"),
        "user": unquote(url.username or ""),
        "password": unquote(url.password or ""),
    }


async def run_smoke(handler: BacktestHandler, db: Any, *, today: dt.date) -> dict[str, Any]:
    """Backtest `TICKERS` over the last `SESSIONS` trading days up to `today`."""
    bars = await asyncio.to_thread(
        db.get_adjusted_daily_bars,
        Ticker(value=TICKERS[0]),
        start=today - dt.timedelta(days=SESSIONS * 3),
        end=today,
    )
    days = sorted({b.date for b in bars if b.date <= today})[-SESSIONS:]
    if not days:
        raise SystemExit(f"no daily bars for {TICKERS[0]} up to {today}")
    result = await handler.handle(
        BacktestCommand(
            tickers=TICKERS,
            start_date=dt.datetime.combine(days[0], dt.time(), tzinfo=dt.UTC),
            end_date=dt.datetime.combine(days[-1], dt.time(), tzinfo=dt.UTC),
            initial_capital=INITIAL_CAPITAL,
            strategy=STRATEGY,
        )
    )
    m = result.metrics
    response = BacktestResult(
        strategy=STRATEGY,
        start_date=result.start_date.strftime("%Y-%m-%d"),
        end_date=result.end_date.strftime("%Y-%m-%d"),
        initial_capital=result.initial_capital,
        final_capital=result.final_capital,
        total_return_pct=float(m.total_return_pct),
        sharpe_ratio=float(m.sharpe_ratio),
        max_drawdown_pct=float(m.max_drawdown_pct),
        total_trades=m.total_trades,
        winning_trades=m.winning_trades,
        losing_trades=m.losing_trades,
        win_rate=float(m.win_rate),
        is_placeholder=False,
        strategy_is_placeholder=True,  # _generate_signal is still the demo rule
    )
    return response.model_dump(mode="json")


def main() -> None:
    from daytrade_it.infrastructure.external.quant_airflow_db_client import (  # noqa: PLC0415
        QuantAirflowDBClient,
    )

    db = QuantAirflowDBClient(**db_kwargs(os.environ))
    response = asyncio.run(
        run_smoke(BacktestHandler(market_data_client=db), db, today=dt.date.today())
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
