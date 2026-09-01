# ADR-004: MCP Tool Interface for AI Agents

## Status

Accepted

## Context

A key goal of this refactor is making the system "AI-agent friendly". AI agents (like Claude, GPT) can use tools to interact with external systems. The Model Context Protocol (MCP) is emerging as a standard for defining these tools.

## Decision

We will expose all major system capabilities as MCP-compatible tools:

### Tool Categories

1. **Analysis Tools**
   - `analyze_ticker(ticker)` - Full analysis with news + technicals
   - `get_news_sentiment(ticker, hours)` - Recent news sentiment
   - `get_price_data(ticker, period)` - Historical prices

2. **Trading Tools**
   - `get_signals(filters)` - Current trading signals
   - `create_signal(params)` - Generate manual signal
   - `execute_trade(signal_id)` - Execute a trade
   - `close_position(ticker)` - Close a position

3. **Portfolio Tools**
   - `get_portfolio()` - Current positions
   - `get_trades(filters)` - Trade history
   - `get_performance()` - P&L metrics

4. **Backtesting Tools**
   - `backtest(strategy, params)` - Run backtest
   - `compare_strategies(strategies)` - Compare results

5. **System Tools**
   - `get_system_status()` - Health check
   - `get_metrics()` - Performance metrics
   - `update_config(params)` - Runtime config

### Tool Design Principles

1. **Self-documenting**: Clear descriptions and examples
2. **Type-safe**: Pydantic models for inputs/outputs
3. **Idempotent**: Safe to retry
4. **Composable**: Tools can be chained
5. **Observable**: All actions are logged

## Consequences

### Positive

- AI agents can fully operate the system
- Clear, documented API surface
- Forces good API design
- Enables automation

### Negative

- Additional abstraction layer
- Must maintain MCP compatibility

## Implementation

```python
@mcp_tool(name="analyze_ticker", description="Get comprehensive analysis for a stock ticker")
async def analyze_ticker(
    ticker: Annotated[str, "Stock ticker symbol (e.g., AAPL)"],
    include_news: bool = True,
    include_technicals: bool = True,
) -> TickerAnalysis: ...
```
