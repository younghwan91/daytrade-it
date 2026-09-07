"""Application Handlers - Command and Query execution"""

import importlib

import structlog

from krx_signal_engine.application.handlers.backtest import (
    BacktestCommand,
    BacktestHandler,
    BacktestMetrics,
    BacktestResult,
    BacktestTrade,
)

# NOTE (Task 13 side-fix, discovered not caused by this task):
# news_analysis.py, signal_generation.py, and trade_execution.py each have
# pre-existing, untested import-time defects independent of anything
# Task 13 touches:
#   - news_analysis.py imports `domain.entities.news.SentimentAnalysis`,
#     which does not exist (news.py only defines `NewsArticle`).
#   - signal_generation.py imports `domain.events.trading_events
#     .SignalGeneratedEvent`, which was apparently renamed to
#     `SignalGenerated` at some point without updating this import.
#   - trade_execution.py (transitively, via this package's import order)
#     hits the same class of stale-name breakage.
# A repo-wide grep found zero tests importing any of
# NewsAnalysisHandler/SignalGenerationHandler/TradeExecutionHandler, so all
# three have been silently broken with no coverage. Fully repairing them is
# out of this task's scope (unrelated handlers, not part of the
# backtest/cost-model wiring Task 13 does) -- but eagerly importing them at
# module level here was blocking `from
# krx_signal_engine.application.handlers.backtest import ...` (and therefore
# Task 13's real backtest engine) from importing at all, since Python must
# fully execute a package's __init__ before any of its submodules.
# Isolating each import so one broken, uncovered handler can't take the
# others down with it is the minimal fix; the underlying defects are left
# for whoever owns those handlers next.
_logger = structlog.get_logger(__name__)


def _optional_import(module_path: str, names: tuple[str, ...]) -> dict[str, object | None]:
    """Best-effort import of `names` from `module_path`.

    Returns a dict of name -> imported object, or name -> None (with a
    logged warning) if the module fails to import. See the module-level
    NOTE above for why this exists.
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover
        _logger.warning(
            f"{module_path} unavailable at import time "
            "(pre-existing, unrelated defect -- see this module's NOTE)",
            error=str(exc),
        )
        return dict.fromkeys(names)
    return {name: getattr(module, name) for name in names}


_news_analysis = _optional_import(
    "krx_signal_engine.application.handlers.news_analysis",
    ("AnalyzeNewsCommand", "NewsAnalysisHandler", "NewsAnalysisResult"),
)
AnalyzeNewsCommand = _news_analysis["AnalyzeNewsCommand"]
NewsAnalysisHandler = _news_analysis["NewsAnalysisHandler"]
NewsAnalysisResult = _news_analysis["NewsAnalysisResult"]

_signal_generation = _optional_import(
    "krx_signal_engine.application.handlers.signal_generation",
    (
        "GenerateSignalCommand",
        "SignalGenerationConfig",
        "SignalGenerationHandler",
        "SignalGenerationResult",
    ),
)
GenerateSignalCommand = _signal_generation["GenerateSignalCommand"]
SignalGenerationConfig = _signal_generation["SignalGenerationConfig"]
SignalGenerationHandler = _signal_generation["SignalGenerationHandler"]
SignalGenerationResult = _signal_generation["SignalGenerationResult"]

_trade_execution = _optional_import(
    "krx_signal_engine.application.handlers.trade_execution",
    (
        "ExecuteTradeCommand",
        "ManualTradeCommand",
        "PositionSizingConfig",
        "TradeExecutionHandler",
        "TradeExecutionResult",
    ),
)
ExecuteTradeCommand = _trade_execution["ExecuteTradeCommand"]
ManualTradeCommand = _trade_execution["ManualTradeCommand"]
PositionSizingConfig = _trade_execution["PositionSizingConfig"]
TradeExecutionHandler = _trade_execution["TradeExecutionHandler"]
TradeExecutionResult = _trade_execution["TradeExecutionResult"]

__all__ = [
    # News Analysis
    "AnalyzeNewsCommand",
    "NewsAnalysisHandler",
    "NewsAnalysisResult",
    # Signal Generation
    "GenerateSignalCommand",
    "SignalGenerationConfig",
    "SignalGenerationHandler",
    "SignalGenerationResult",
    # Trade Execution
    "ExecuteTradeCommand",
    "ManualTradeCommand",
    "PositionSizingConfig",
    "TradeExecutionHandler",
    "TradeExecutionResult",
    # Backtest
    "BacktestCommand",
    "BacktestHandler",
    "BacktestMetrics",
    "BacktestResult",
    "BacktestTrade",
]
