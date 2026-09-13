"""Integration test: a real backtest run against real KRX historical data
(Task 13, Phase 5).

Proves `BacktestHandler` actually computes metrics from a genuine
simulation over real `daily_bars_adjusted` rows fetched via Task 6's
`QuantAirflowDBClient`, not hardcoded stub numbers -- and that
`BacktestResult.is_placeholder` (in `interfaces/mcp/tools.py`) is `False`
for a real run.

Same reachability-skip pattern as
`test_quant_airflow_db_client_integration.py`: if the quant-airflow
TimescaleDB isn't reachable, the whole module is skipped rather than
failing noisily.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import psycopg2
import pytest

from daytrade_it.application.handlers.backtest import BacktestCommand, BacktestHandler
from daytrade_it.domain.services.cost_model import KoreanCostModel
from daytrade_it.infrastructure.external.quant_airflow_db_client import QuantAirflowDBClient
from daytrade_it.shared.config.settings import QuantAirflowDBSettings

# Real, well-known large-cap KRX code (Samsung Electronics), same ticker
# Task 6's integration test uses.
SAMSUNG_CODE = "005930"


def _password_from_sibling_quant_airflow_env() -> str:
    """Same local-dev convenience as the Task 6 integration test: fall back
    to the sibling `quant-airflow` checkout's `.env` for a password so this
    runs un-configured on a dev machine that has that sibling checked out.
    """
    from pathlib import Path

    candidate = Path.home() / "Documents" / "git" / "quant-airflow" / ".env"
    if not candidate.is_file():
        return ""
    for line in candidate.read_text(encoding="utf-8").splitlines():
        if line.startswith("TIMESCALE_PASSWORD="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _test_settings() -> QuantAirflowDBSettings:
    import os

    password = os.environ.get(
        "GPTQ_QUANT_AIRFLOW_DB_PASSWORD"
    ) or _password_from_sibling_quant_airflow_env()

    return QuantAirflowDBSettings(
        host=os.environ.get("GPTQ_QUANT_AIRFLOW_DB_HOST", "localhost"),
        port=int(os.environ.get("GPTQ_QUANT_AIRFLOW_DB_PORT", "5432")),
        dbname=os.environ.get("GPTQ_QUANT_AIRFLOW_DB_DBNAME", "kr_quant"),
        user=os.environ.get("GPTQ_QUANT_AIRFLOW_DB_USER", "kr_quant"),
        password=password,
        connect_timeout=3.0,
    )


def _is_reachable(settings: QuantAirflowDBSettings) -> bool:
    try:
        conn = psycopg2.connect(
            host=settings.host,
            port=settings.port,
            dbname=settings.dbname,
            user=settings.user,
            password=settings.password.get_secret_value(),
            connect_timeout=int(settings.connect_timeout),
        )
    except psycopg2.Error:
        return False
    else:
        conn.close()
        return True


_SETTINGS = _test_settings()
_REACHABLE = _is_reachable(_SETTINGS)

pytestmark = [
    pytest.mark.skipif(
        not _REACHABLE,
        reason=(
            "quant-airflow TimescaleDB not reachable "
            f"({_SETTINGS.host}:{_SETTINGS.port}/{_SETTINGS.dbname}) - "
            "set GPTQ_QUANT_AIRFLOW_DB_* env vars or run on a machine with "
            "LAN access to it."
        ),
    ),
    pytest.mark.asyncio,
]


@pytest.fixture
def handler() -> BacktestHandler:
    client = QuantAirflowDBClient(settings=_SETTINGS)
    return BacktestHandler(market_data_client=client, cost_model=KoreanCostModel())


class TestRealBacktestRun:
    async def test_backtest_computes_real_metrics_from_real_krx_data(
        self, handler: BacktestHandler
    ) -> None:
        """A genuine backtest over Samsung Electronics' real daily bars
        must produce metrics that are actually computed from the
        simulation -- not any of the old hardcoded stub values
        (final_capital = initial * 1.15, sharpe=1.2, total_trades=150,
        win_rate=0.6, etc.).
        """
        command = BacktestCommand(
            tickers=[SAMSUNG_CODE],
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2026, 3, 31, tzinfo=UTC),
            initial_capital=Decimal("10000000"),
            strategy="news_momentum",
            params={},
        )

        result = await handler.handle(command)

        # Real data was actually fetched and simulated over.
        assert len(result.equity_curve) > 0
        assert result.metrics.total_trades > 0

        # Not the old hardcoded MCP-tool stub shape (final = initial * 1.15,
        # total_trades = 150) -- proves the result wasn't silently
        # short-circuited to a fixed placeholder. (Individual metric
        # values like win_rate/losing_trades aren't asserted != their old
        # stub numbers here since a real run coincidentally landing on the
        # same small integer/rate is possible and would make this test
        # flaky against real, changing market data.)
        assert result.final_capital != result.initial_capital * Decimal("1.15")
        assert result.metrics.total_trades != 150

        # Sanity bounds on genuinely-computed metrics (same bounds the
        # stale e2e test suite already asserts for a real backtest).
        assert -10 <= float(result.metrics.sharpe_ratio) <= 10
        assert 0 <= float(result.metrics.max_drawdown_pct) <= 1
        assert 0 <= float(result.metrics.win_rate) <= 1
        assert result.metrics.total_trades == (
            result.metrics.winning_trades + result.metrics.losing_trades
        )

        # Trades were actually costed through KoreanCostModel: a round
        # trip's exit price is never exactly the reference close (slippage
        # was applied), and every closed trade's pnl differs from what a
        # cost-free (price - entry) * quantity computation would give,
        # confirming commission/tax were actually deducted.
        for trade in result.trades:
            if trade.exit_price is None:
                continue
            gross_pnl = (trade.exit_price - trade.entry_price) * trade.quantity
            assert trade.pnl != gross_pnl, (
                "trade.pnl should differ from cost-free gross P&L once "
                "commission/tax are deducted"
            )

    async def test_backtest_result_is_not_placeholder_via_mcp_tool(self) -> None:
        """Regression guard: interfaces/mcp/tools.py's backtest() must
        return is_placeholder=False for a real run against real data,
        with metrics relayed verbatim from the handler.
        """
        from daytrade_it.interfaces.mcp.tools import MCPToolContext, backtest

        ctx = MCPToolContext(
            backtest_handler=BacktestHandler(
                market_data_client=QuantAirflowDBClient(settings=_SETTINGS)
            )
        )
        MCPToolContext.set_instance(ctx)
        try:
            result = await backtest(
                strategy="news_momentum",
                tickers=[SAMSUNG_CODE],
                start_date="2025-01-01",
                end_date="2026-03-31",
                initial_capital=10_000_000,
            )
        finally:
            MCPToolContext.set_instance(None)  # type: ignore[arg-type]

        assert result.is_placeholder is False
        assert result.total_trades > 0
