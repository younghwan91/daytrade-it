# MCP Integration for GPT Quant V2

This document describes the Model Context Protocol (MCP) integration for GPT Quant V2, enabling AI agents to interact with the trading system.

## Overview

### What is MCP?

The Model Context Protocol (MCP) is a standardized protocol for AI agent interoperability. It enables AI assistants (like Claude, GPT, or LangChain agents) to discover and invoke tools exposed by external systems using JSON-RPC 2.0 over stdio.

### How GPT Quant V2 Uses MCP

GPT Quant V2 implements an MCP server that exposes trading tools to AI agents. This allows AI systems to:

- **Analyze stocks** with real-time prices, news, and sentiment
- **Generate and retrieve trading signals** based on ML models
- **Execute trades** through connected brokers
- **Monitor portfolios** and system health
- **Run backtests** on trading strategies
- **Train and manage ML models**

The MCP server communicates via JSON-RPC 2.0 over stdin/stdout, making it compatible with any MCP-compliant AI agent.

```
┌─────────────────┐     stdin/stdout      ┌─────────────────┐
│    AI Agent     │◄────JSON-RPC 2.0─────►│  MCP Server     │
│  (Claude, GPT)  │                       │  (gpt_quant_v2) │
└─────────────────┘                       └─────────────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │  Trading Tools  │
                                          │  - Broker API   │
                                          │  - Market Data  │
                                          │  - ML Models    │
                                          └─────────────────┘
```

---

## Available Tools

### 1. analyze_ticker

Perform comprehensive analysis of a stock ticker including price data, news sentiment, and trading signals.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `ticker` | string | Yes | - | Stock ticker symbol (e.g., AAPL, MSFT, GOOGL) |
| `include_news` | boolean | No | `true` | Whether to include recent news analysis |
| `include_signals` | boolean | No | `true` | Whether to include active trading signals |
| `news_hours` | integer | No | `24` | Hours of news history to analyze |

**Response:**

```json
{
  "ticker": "AAPL",
  "name": "AAPL",
  "exchange": "NASDAQ",
  "current_price": "178.50",
  "price_change_pct": 1.25,
  "sentiment": {
    "average_score": 0.35,
    "articles_analyzed": 8,
    "bullish_count": 5,
    "bearish_count": 1,
    "neutral_count": 2
  },
  "recent_news": [
    {
      "title": "Apple Reports Record Q4 Earnings",
      "source": "Reuters",
      "published_at": "2024-01-15T14:30:00Z",
      "url": "https://...",
      "sentiment_score": 0.72,
      "sentiment_label": "bullish",
      "confidence": 0.89,
      "key_points": ["Revenue beat expectations", "iPhone sales strong"]
    }
  ],
  "signals": [
    {
      "id": "a1b2c3d4-...",
      "direction": "BUY",
      "confidence": 0.85,
      "source": "news_momentum",
      "predicted_return": 0.035,
      "expires_at": "2024-01-16T00:00:00Z"
    }
  ],
  "analysis_timestamp": "2024-01-15T15:00:00Z"
}
```

---

### 2. get_signals

Retrieve active trading signals with optional filtering.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `min_confidence` | float | No | `0.7` | Minimum confidence threshold (0.0-1.0) |
| `ticker` | string | No | `null` | Filter by specific ticker |
| `direction` | string | No | `null` | Filter by direction (`BUY` or `SELL`) |
| `limit` | integer | No | `20` | Maximum signals to return |

**Response:**

```json
[
  {
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "ticker": "AAPL",
    "direction": "BUY",
    "confidence": 0.85,
    "source": "news_momentum",
    "predicted_return": 0.035,
    "predicted_drawdown": 0.012,
    "risk_reward_ratio": 2.9,
    "expires_at": "2024-01-16T00:00:00Z",
    "created_at": "2024-01-15T10:00:00Z"
  }
]
```

---

### 3. execute_trade

Execute a trade based on a trading signal.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `signal_id` | string | Yes | - | UUID of the trading signal to execute |
| `quantity` | integer | No | `null` | Number of shares (null = auto-size using 2% risk) |
| `order_type` | string | No | `"MARKET"` | Order type (`MARKET` or `LIMIT`) |
| `limit_price` | float | No | `null` | Limit price (required for LIMIT orders) |

**Response:**

```json
{
  "trade_id": "f1e2d3c4-b5a6-7890-1234-567890abcdef",
  "signal_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "ticker": "AAPL",
  "direction": "BUY",
  "quantity": 55,
  "price": "178.50",
  "total_value": "9817.50",
  "status": "filled",
  "executed_at": "2024-01-15T15:30:00Z"
}
```

**Note:** When `quantity` is not specified, the system uses 2% of portfolio value for automatic position sizing.

---

### 4. backtest

Run a backtest on a trading strategy with historical data.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `strategy` | string | No | `"news_momentum"` | Strategy name (`news_momentum`, `technical`, `combined`) |
| `start_date` | string | No | 1 year ago | Start date (YYYY-MM-DD format) |
| `end_date` | string | No | today | End date (YYYY-MM-DD format) |
| `initial_capital` | float | No | `100000` | Starting capital in USD |
| `parameters` | object | No | `null` | Strategy-specific parameters |

**Response:**

```json
{
  "strategy": "news_momentum",
  "start_date": "2023-01-15",
  "end_date": "2024-01-15",
  "initial_capital": "100000.00",
  "final_capital": "115000.00",
  "total_return_pct": 15.0,
  "sharpe_ratio": 1.2,
  "max_drawdown_pct": 8.5,
  "total_trades": 150,
  "winning_trades": 90,
  "losing_trades": 60,
  "win_rate": 0.6
}
```

---

### 5. get_portfolio

Get current portfolio state including positions and P&L.

**Parameters:** None

**Response:**

```json
{
  "cash": "45230.50",
  "total_value": "125450.75",
  "positions": [
    {
      "ticker": "AAPL",
      "quantity": 100,
      "side": "long",
      "average_price": 165.50,
      "current_price": 178.50,
      "market_value": 17850.00,
      "unrealized_pnl": 1300.00,
      "realized_pnl": 0.00
    },
    {
      "ticker": "MSFT",
      "quantity": 50,
      "side": "long",
      "average_price": 380.00,
      "current_price": 395.25,
      "market_value": 19762.50,
      "unrealized_pnl": 762.50,
      "realized_pnl": 250.00
    }
  ],
  "unrealized_pnl": "2062.50",
  "realized_pnl": "250.00",
  "updated_at": "2024-01-15T15:30:00Z"
}
```

---

### 6. get_system_status

Get system health status and component states.

**Parameters:** None

**Response:**

```json
{
  "status": "healthy",
  "components": {
    "tiingo": {
      "status": "connected",
      "type": "market_data"
    },
    "openai": {
      "status": "connected",
      "type": "sentiment_analysis"
    },
    "broker": {
      "status": "connected",
      "type": "trading"
    },
    "signal_repository": {
      "status": "connected",
      "type": "persistence"
    },
    "trade_repository": {
      "status": "connected",
      "type": "persistence"
    }
  },
  "uptime_seconds": 86400,
  "last_signal_at": "2024-01-15T14:00:00Z",
  "last_trade_at": "2024-01-15T13:30:00Z"
}
```

**Status Values:**
- `healthy` - All components connected
- `partial` - Some components connected
- `degraded` - Some components in error state
- `not_configured` - No components configured

---

### 7. get_news

Get recent news articles for a ticker with optional sentiment analysis.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `ticker` | string | Yes | - | Stock ticker symbol |
| `hours` | integer | No | `24` | Hours of news history to fetch |
| `limit` | integer | No | `20` | Maximum number of articles |
| `analyze_sentiment` | boolean | No | `true` | Whether to analyze sentiment |

**Response:**

```json
[
  {
    "id": "news-123456",
    "title": "Apple Unveils New AI Features",
    "source": "TechCrunch",
    "url": "https://...",
    "published_at": "2024-01-15T12:00:00Z",
    "tickers": ["AAPL"],
    "sentiment_score": 0.65,
    "sentiment_label": "bullish",
    "sentiment_confidence": 0.88,
    "key_points": [
      "New AI assistant features announced",
      "Privacy-focused approach emphasized"
    ]
  }
]
```

---

### 8. get_price_history

Get historical OHLCV price data for a ticker.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `ticker` | string | Yes | - | Stock ticker symbol |
| `days` | integer | No | `30` | Number of days of history |
| `interval` | string | No | `"daily"` | Data interval (`daily`, `weekly`, `monthly`) |

**Response:**

```json
[
  {
    "timestamp": "2024-01-15T00:00:00Z",
    "open": "175.50",
    "high": "179.25",
    "low": "174.80",
    "close": "178.50",
    "volume": 52340000
  },
  {
    "timestamp": "2024-01-14T00:00:00Z",
    "open": "173.20",
    "high": "176.40",
    "low": "172.90",
    "close": "175.30",
    "volume": 48210000
  }
]
```

---

### 9. train_model

Trigger ML model training with specified parameters.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `model_name` | string | Yes | - | Name for the trained model |
| `tickers` | array | No | `null` | Tickers to train on (null = all available) |
| `lookback_days` | integer | No | `365` | Days of historical data to use |
| `optimize_hyperparams` | boolean | No | `true` | Whether to run hyperparameter optimization |

**Response:**

```json
{
  "model_name": "news_sentiment_v2",
  "model_version": "1.0.0",
  "metrics": {
    "accuracy": 0.72,
    "f1_weighted": 0.68,
    "precision_weighted": 0.70,
    "recall_weighted": 0.69
  },
  "training_duration_seconds": 145.3,
  "samples_used": 36500,
  "created_at": "2024-01-15T16:00:00Z"
}
```

---

### 10. list_models

List available ML models with their metadata.

**Parameters:** None

**Response:**

```json
[
  {
    "name": "news_sentiment_v1",
    "version": "1.0.0",
    "model_type": "XGBClassifier",
    "metrics": {
      "accuracy": 0.68,
      "f1_weighted": 0.65
    },
    "feature_count": 45,
    "created_at": "2024-01-10T10:00:00Z",
    "is_active": true
  },
  {
    "name": "technical_signals",
    "version": "2.1.0",
    "model_type": "XGBClassifier",
    "metrics": {
      "accuracy": 0.71,
      "f1_weighted": 0.69
    },
    "feature_count": 62,
    "created_at": "2024-01-05T14:30:00Z",
    "is_active": false
  }
]
```

---

## Integration Guide

### Starting the MCP Server

**As a Python Module:**

```bash
python -m gpt_quant_v2.interfaces.mcp.mcp_server
```

**Programmatically:**

```python
import asyncio
from gpt_quant_v2.interfaces.mcp.mcp_server import MCPServer

async def main():
    server = MCPServer()
    await server.run()

asyncio.run(main())
```

**With Custom Context:**

```python
from gpt_quant_v2.interfaces.mcp.mcp_server import MCPServer
from gpt_quant_v2.interfaces.mcp.tools import MCPToolContext

# Create context with dependencies
context = MCPToolContext(
    tiingo_client=tiingo_client,
    openai_client=openai_client,
    broker=broker_instance,
    signal_repository=signal_repo,
    trade_repository=trade_repo,
)

server = MCPServer(context=context)
await server.run()
```

### Connecting an AI Agent

#### Claude Desktop Integration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "gpt-quant-v2": {
      "command": "python",
      "args": ["-m", "gpt_quant_v2.interfaces.mcp.mcp_server"],
      "cwd": "/path/to/gpt_quant_v2"
    }
  }
}
```

#### LangChain Integration

```python
from langchain.tools import StructuredTool
import subprocess
import json

class GPTQuantMCPClient:
    def __init__(self):
        self.process = subprocess.Popen(
            ["python", "-m", "gpt_quant_v2.interfaces.mcp.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        self._initialize()
    
    def _initialize(self):
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "langchain", "version": "1.0.0"}
            }
        }
        self._send(request)
    
    def _send(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())
    
    def call_tool(self, name: str, arguments: dict):
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments}
        }
        return self._send(request)

# Create LangChain tool
client = GPTQuantMCPClient()

analyze_tool = StructuredTool.from_function(
    func=lambda ticker: client.call_tool("analyze_ticker", {"ticker": ticker}),
    name="analyze_ticker",
    description="Analyze a stock ticker with price, news, and signals"
)
```

### Example Tool Invocations

#### Initialize Connection

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {
      "name": "my-ai-agent",
      "version": "1.0.0"
    }
  }
}
```

**Response:**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "serverInfo": {
      "name": "gpt-quant-v2",
      "version": "1.0.0"
    },
    "capabilities": {
      "tools": {}
    }
  }
}
```

#### List Available Tools

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}
```

#### Call a Tool

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "analyze_ticker",
    "arguments": {
      "ticker": "AAPL",
      "include_news": true,
      "include_signals": true,
      "news_hours": 48
    }
  }
}
```

#### Execute a Trade

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/call",
  "params": {
    "name": "execute_trade",
    "arguments": {
      "signal_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "quantity": 100,
      "order_type": "LIMIT",
      "limit_price": 175.00
    }
  }
}
```

---

## Error Handling

### MCP Protocol Error Codes

The MCP server uses standard JSON-RPC 2.0 error codes plus MCP-specific extensions:

| Code | Name | Description |
|------|------|-------------|
| `-32700` | Parse Error | Invalid JSON received |
| `-32600` | Invalid Request | JSON is not a valid request object |
| `-32601` | Method Not Found | Requested method does not exist |
| `-32602` | Invalid Params | Invalid method parameters |
| `-32603` | Internal Error | Internal server error |
| `-32001` | Tool Not Found | Requested tool does not exist |
| `-32002` | Tool Execution Error | Error during tool execution |
| `-32003` | Authentication Error | Authentication required or failed |
| `-32004` | Resource Not Found | Requested resource not found |
| `-32005` | Rate Limit Exceeded | Too many requests |
| `-32006` | Invalid Tool Input | Tool input validation failed |

### Application Error Codes

GPT Quant V2 uses domain-specific error codes for business logic errors:

#### Configuration Errors (GPTQ-1xx)

| Code | Description |
|------|-------------|
| `GPTQ-100` | General configuration error |
| `GPTQ-101` | Required configuration value missing |
| `GPTQ-102` | Invalid configuration value |
| `GPTQ-103` | Configuration file not found |
| `GPTQ-105` | Required environment variable missing |

#### Validation Errors (GPTQ-2xx)

| Code | Description |
|------|-------------|
| `GPTQ-200` | General validation error |
| `GPTQ-201` | Required field missing |
| `GPTQ-202` | Invalid field type |
| `GPTQ-204` | Value out of range |
| `GPTQ-205` | Invalid ticker symbol |
| `GPTQ-206` | Invalid date range |

#### Domain Errors (GPTQ-3xx)

| Code | Description |
|------|-------------|
| `GPTQ-301` | Insufficient funds |
| `GPTQ-302` | Invalid trading signal |
| `GPTQ-303` | Risk limit exceeded |
| `GPTQ-304` | Position not found |
| `GPTQ-305` | Invalid order state |
| `GPTQ-306` | Portfolio constraint violated |
| `GPTQ-307` | Strategy execution error |

#### External API Errors (GPTQ-5xx)

| Code | Description |
|------|-------------|
| `GPTQ-501` | Polygon API error |
| `GPTQ-502` | Polygon rate limit exceeded |
| `GPTQ-510` | OpenAI API error |
| `GPTQ-511` | OpenAI rate limit exceeded |
| `GPTQ-520` | Tiingo API error |
| `GPTQ-521` | Tiingo rate limit exceeded |

#### Broker Errors (GPTQ-6xx)

| Code | Description |
|------|-------------|
| `GPTQ-601` | Order rejected by broker |
| `GPTQ-602` | Insufficient margin |
| `GPTQ-603` | Broker connection error |
| `GPTQ-604` | Broker authentication failed |
| `GPTQ-606` | Market closed |
| `GPTQ-607` | Invalid order quantity |

### Error Response Format

All errors follow this format:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32002,
    "message": "Signal not found: a1b2c3d4-...",
    "data": {
      "error_code": "GPTQ-302",
      "signal_id": "a1b2c3d4-...",
      "traceback": "..."
    }
  }
}
```

### Handling Errors in Client Code

```python
def handle_mcp_response(response: dict):
    if "error" in response:
        error = response["error"]
        code = error["code"]
        message = error["message"]
        
        if code == -32001:  # Tool not found
            raise ToolNotFoundError(message)
        elif code == -32002:  # Tool execution error
            app_code = error.get("data", {}).get("error_code", "")
            if app_code == "GPTQ-301":
                raise InsufficientFundsError(message)
            elif app_code == "GPTQ-303":
                raise RiskLimitError(message)
            else:
                raise ToolExecutionError(message)
        elif code == -32005:  # Rate limit
            raise RateLimitError(message)
        else:
            raise MCPError(code, message)
    
    return response.get("result")
```

---

## Additional Resources

- [MCP Specification](https://modelcontextprotocol.io/)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
- [GPT Quant V2 API Documentation](../api/README.md)
- [Architecture Decision: MCP Integration](../adr/004-mcp-integration.md)
