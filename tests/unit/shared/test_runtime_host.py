"""Backtests are refused off simnode, at the MCP tool and the CLI."""

import pytest
from typer.testing import CliRunner

from daytrade_it.interfaces.cli.main import app
from daytrade_it.interfaces.mcp.tools import MCPToolContext, backtest
from daytrade_it.shared.runtime_host import require_backtest_host


def _host(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setattr("daytrade_it.shared.runtime_host.socket.gethostname", lambda: name)


def test_simnode_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _host(monkeypatch, "simnode")
    require_backtest_host()


def test_trader_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _host(monkeypatch, "trader")
    with pytest.raises(RuntimeError, match="simnode"):
        require_backtest_host()


@pytest.mark.asyncio
async def test_mcp_backtest_tool_refuses_before_touching_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _host(monkeypatch, "trader")

    class _Handler:
        calls = 0

        async def handle(self, command):  # noqa: ANN001, ANN201, ARG002
            _Handler.calls += 1

    MCPToolContext.set_instance(MCPToolContext(backtest_handler=_Handler()))
    try:
        with pytest.raises(RuntimeError, match="simnode"):
            await backtest(strategy="composite", tickers=["005930"])
    finally:
        MCPToolContext.set_instance(None)  # type: ignore[arg-type]
    assert _Handler.calls == 0


def test_cli_backtest_exits_nonzero_off_simnode(monkeypatch: pytest.MonkeyPatch) -> None:
    _host(monkeypatch, "trader")
    result = CliRunner().invoke(app, ["backtest", "--start", "2026-01-01", "--end", "2026-02-01"])
    assert result.exit_code == 1
    assert "Running backtest" not in result.output
