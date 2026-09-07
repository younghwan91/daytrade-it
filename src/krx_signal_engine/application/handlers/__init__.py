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

# Whole-branch review minor finding: names from a broken handler module
# used to be bound at module level as plain `None` (see `_optional_import`
# below), which let `from krx_signal_engine.application.handlers import
# NewsAnalysisHandler` succeed silently -- the caller would only discover
# the problem later, as a confusing `TypeError: 'NoneType' object is not
# callable` (or similar) at the point of use, with no link back to the
# real cause. `_BROKEN_NAMES` + the module `__getattr__` below fix that:
# a broken name is simply never bound, so accessing it falls through to
# `__getattr__`, which raises a clear `ImportError` chained to the
# original cause instead.
_BROKEN_NAMES: dict[str, ImportError] = {}


def _optional_import(module_path: str, names: tuple[str, ...]) -> dict[str, object]:
    """Best-effort import of `names` from `module_path`.

    Returns a dict of name -> imported object for names that imported
    successfully. If the module fails to import, none of `names` are
    included in the returned dict -- instead each is recorded in
    `_BROKEN_NAMES` so the module-level `__getattr__` below can raise a
    clear, chained `ImportError` when/if something actually tries to use
    one of them. See the module-level NOTE above for why this exists.
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover
        _logger.warning(
            f"{module_path} unavailable at import time "
            "(pre-existing, unrelated defect -- see this module's NOTE)",
            error=str(exc),
        )
        for name in names:
            _BROKEN_NAMES[name] = exc
        return {}
    return {name: getattr(module, name) for name in names}


def __getattr__(name: str) -> object:
    """Raise a clear, chained ImportError for a name whose defining
    handler module failed to import, instead of the normal
    AttributeError -- see `_BROKEN_NAMES` above."""
    if name in _BROKEN_NAMES:
        raise ImportError(
            f"krx_signal_engine.application.handlers.{name} is unavailable: "
            "its defining module failed to import at package-init time "
            "(pre-existing, unrelated defect -- see this package's "
            "__init__.py module-level NOTE for details, not fixed here)."
        ) from _BROKEN_NAMES[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_news_analysis = _optional_import(
    "krx_signal_engine.application.handlers.news_analysis",
    ("AnalyzeNewsCommand", "NewsAnalysisHandler", "NewsAnalysisResult"),
)
globals().update(_news_analysis)

_signal_generation = _optional_import(
    "krx_signal_engine.application.handlers.signal_generation",
    (
        "GenerateSignalCommand",
        "SignalGenerationConfig",
        "SignalGenerationHandler",
        "SignalGenerationResult",
    ),
)
globals().update(_signal_generation)

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
globals().update(_trade_execution)

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
