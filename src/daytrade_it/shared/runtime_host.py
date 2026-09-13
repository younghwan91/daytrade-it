"""Host placement rules for heavy workloads -- re-exported from krx-quant-core.

Backtests run only on simnode (user directive, 2026-09-13). Guarded at the
entry points (MCP tool, CLI), not inside `BacktestHandler`, so the engine
itself stays testable anywhere. The check is shared with scalp-it in
:mod:`krx_quant_core.runtime.host`; patch
``krx_quant_core.runtime.host.socket.gethostname`` to fake the host in tests.
"""

from __future__ import annotations

from krx_quant_core.runtime.host import BACKTEST_HOST, require_backtest_host

__all__ = ["BACKTEST_HOST", "require_backtest_host"]
