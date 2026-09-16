"""scripts/research/backtest_smoke.py: last 20 trading days x 5 tickers, MCP response keys."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts" / "research"))

from backtest_smoke import TICKERS, run_smoke  # noqa: E402

from daytrade_it.application.handlers.backtest import BacktestHandler  # noqa: E402
from daytrade_it.infrastructure.external.quant_airflow_db_client import (  # noqa: E402
    AdjustedDailyBar,
)

MCP_KEYS = {
    "strategy",
    "start_date",
    "end_date",
    "initial_capital",
    "final_capital",
    "total_return_pct",
    "sharpe_ratio",
    "max_drawdown_pct",
    "total_trades",
    "winning_trades",
    "losing_trades",
    "win_rate",
    "is_placeholder",
    "strategy_is_placeholder",
}


class _FakeDB:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dt.date, dt.date]] = []
        first = dt.date(2026, 7, 1)
        self.days = [
            first + dt.timedelta(days=i)
            for i in range(60)
            if (first + dt.timedelta(days=i)).weekday() < 5
        ]

    def get_adjusted_daily_bars(self, ticker, *, start, end):  # noqa: ANN001, ANN201
        self.calls.append((ticker.value, start, end))
        return [
            AdjustedDailyBar(
                code=ticker.value,
                date=d,
                open=1000,
                high=1000,
                low=1000,
                close=1000,
                volume=0,
                source="test",
            )
            for d in self.days
            if start <= d <= end
        ]


async def test_smoke_backtests_last_20_trading_days_of_five_tickers() -> None:
    db = _FakeDB()
    today = dt.date(2026, 8, 31)

    response = await run_smoke(BacktestHandler(market_data_client=db), db, today=today)  # type: ignore[arg-type]

    assert set(response) == MCP_KEYS
    assert TICKERS == ["005930", "000660", "035420", "005380", "051910"]
    past = [d for d in db.days if d <= today]
    assert response["start_date"] == past[-20].isoformat()
    assert response["end_date"] == past[-1].isoformat()
    assert {c[0] for c in db.calls} >= set(TICKERS)
    assert response["is_placeholder"] is False


def test_db_args_come_from_kr_quant_db_when_set() -> None:
    from backtest_smoke import db_kwargs

    assert db_kwargs({}) == {}
    dsn = "postgresql://u:p%40ss@10.0.0.9:5433/kr_quant"  # pragma: allowlist secret
    assert db_kwargs({"KR_QUANT_DB": dsn}) == {
        "host": "10.0.0.9",
        "port": 5433,
        "dbname": "kr_quant",
        "user": "u",
        "password": "p@ss",  # pragma: allowlist secret
    }
