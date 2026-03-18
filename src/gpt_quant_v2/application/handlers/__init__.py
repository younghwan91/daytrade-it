"""Application Handlers - Command and Query execution"""

from gpt_quant_v2.application.handlers.backtest import (
    BacktestCommand,
    BacktestHandler,
    BacktestMetrics,
    BacktestResult,
    BacktestTrade,
)
from gpt_quant_v2.application.handlers.news_analysis import (
    AnalyzeNewsCommand,
    NewsAnalysisHandler,
    NewsAnalysisResult,
)
from gpt_quant_v2.application.handlers.signal_generation import (
    GenerateSignalCommand,
    SignalGenerationConfig,
    SignalGenerationHandler,
    SignalGenerationResult,
)
from gpt_quant_v2.application.handlers.trade_execution import (
    ExecuteTradeCommand,
    ManualTradeCommand,
    PositionSizingConfig,
    TradeExecutionHandler,
    TradeExecutionResult,
)

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
