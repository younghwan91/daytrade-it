"""Host placement rules for heavy workloads.

Backtests run only on simnode (user directive, 2026-09-13): the trader host
runs live collection and order daemons, and simnode has the CPU and the
primary DB. Guarded at the entry points (MCP tool, CLI), not inside
`BacktestHandler`, so the engine itself stays testable anywhere.

scalp-it carries its own copy of this check (`cli_closebet.py`); a shared
backtest engine is planned to absorb both.
"""

from __future__ import annotations

import socket

BACKTEST_HOST = "simnode"


def require_backtest_host() -> None:
    host = socket.gethostname()
    if host != BACKTEST_HOST:
        raise RuntimeError(
            f"Backtests run only on {BACKTEST_HOST!r} (current host: {host!r}). "
            "ssh simnode-local and run it there."
        )
