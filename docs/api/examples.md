# GPT Quant V2 API Examples

This document provides curl examples for common API operations.

## Prerequisites

Set your base URL and authentication token:

```bash
export API_BASE="http://localhost:8000"
export API_TOKEN="your_jwt_token_here"
```

---

## Signal Operations

### List Signals

Get all active trading signals:

```bash
curl -X GET "${API_BASE}/api/v1/signals" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Accept: application/json"
```

#### With Filters

Filter by ticker:

```bash
curl -X GET "${API_BASE}/api/v1/signals?ticker=AAPL" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Accept: application/json"
```

Filter by minimum confidence:

```bash
curl -X GET "${API_BASE}/api/v1/signals?min_confidence=0.75" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Accept: application/json"
```

Filter by direction:

```bash
curl -X GET "${API_BASE}/api/v1/signals?direction=BUY" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Accept: application/json"
```

Combined filters with pagination:

```bash
curl -X GET "${API_BASE}/api/v1/signals?ticker=AAPL&min_confidence=0.7&direction=BUY&page=1&page_size=10" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Accept: application/json"
```

#### Response

```json
{
  "items": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "ticker": "AAPL",
      "direction": "BUY",
      "confidence": 0.85,
      "source": "NEWS_SENTIMENT",
      "predicted_return": "0.025",
      "predicted_drawdown": "-0.01",
      "risk_reward_ratio": "2.5",
      "expires_at": "2024-01-15T16:00:00Z",
      "is_expired": false,
      "metadata": {
        "news_ids": ["article-1", "article-2"],
        "model_version": "1.2.0"
      },
      "news_article_id": "660e8400-e29b-41d4-a716-446655440001",
      "created_at": "2024-01-15T09:30:00Z",
      "updated_at": "2024-01-15T09:30:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20,
  "pages": 1
}
```

---

### Get Signal by ID

Retrieve a specific signal:

```bash
curl -X GET "${API_BASE}/api/v1/signals/550e8400-e29b-41d4-a716-446655440000" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Accept: application/json"
```

#### Response

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "ticker": "AAPL",
  "direction": "BUY",
  "confidence": 0.85,
  "source": "NEWS_SENTIMENT",
  "predicted_return": "0.025",
  "predicted_drawdown": "-0.01",
  "risk_reward_ratio": "2.5",
  "expires_at": "2024-01-15T16:00:00Z",
  "is_expired": false,
  "metadata": {},
  "news_article_id": null,
  "created_at": "2024-01-15T09:30:00Z",
  "updated_at": "2024-01-15T09:30:00Z"
}
```

---

### Create Signal

Create a new manual trading signal:

```bash
curl -X POST "${API_BASE}/api/v1/signals" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "MSFT",
    "direction": "BUY",
    "confidence": 0.75,
    "source": "COMPOSITE",
    "predicted_return": "0.03",
    "predicted_drawdown": "-0.015",
    "risk_reward_ratio": "2.0",
    "expires_at": "2024-01-15T20:00:00Z",
    "metadata": {
      "reason": "Earnings beat expectations",
      "analyst": "manual_entry"
    }
  }'
```

#### Minimal Create Request

Only required fields:

```bash
curl -X POST "${API_BASE}/api/v1/signals" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "GOOGL",
    "direction": "SELL",
    "confidence": 0.65
  }'
```

#### Response (201 Created)

```json
{
  "id": "770e8400-e29b-41d4-a716-446655440002",
  "ticker": "MSFT",
  "direction": "BUY",
  "confidence": 0.75,
  "source": "COMPOSITE",
  "predicted_return": "0.03",
  "predicted_drawdown": "-0.015",
  "risk_reward_ratio": "2.0",
  "expires_at": "2024-01-15T20:00:00Z",
  "is_expired": false,
  "metadata": {
    "reason": "Earnings beat expectations",
    "analyst": "manual_entry"
  },
  "news_article_id": null,
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:00:00Z"
}
```

---

### Delete Signal

Cancel/delete a signal:

```bash
curl -X DELETE "${API_BASE}/api/v1/signals/550e8400-e29b-41d4-a716-446655440000" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Accept: application/json"
```

#### Response

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "deleted": true,
  "message": "Signal successfully deleted"
}
```

---

## Trade Operations

### List Trades

Get all trades:

```bash
curl -X GET "${API_BASE}/api/v1/trades" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Accept: application/json"
```

#### With Filters

Filter by ticker and status:

```bash
curl -X GET "${API_BASE}/api/v1/trades?ticker=AAPL&status=FILLED" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Accept: application/json"
```

Filter by date range:

```bash
curl -X GET "${API_BASE}/api/v1/trades?date_from=2024-01-01T00:00:00Z&date_to=2024-01-15T23:59:59Z" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Accept: application/json"
```

#### Response

```json
{
  "items": [
    {
      "id": "880e8400-e29b-41d4-a716-446655440003",
      "signal_id": "550e8400-e29b-41d4-a716-446655440000",
      "ticker": "AAPL",
      "direction": "BUY",
      "quantity": "100",
      "entry_price": "185.50",
      "exit_price": null,
      "status": "FILLED",
      "pnl": null,
      "fees": "0.50",
      "filled_at": "2024-01-15T09:35:00Z",
      "closed_at": null,
      "notional_value": "18550.00",
      "return_pct": null,
      "created_at": "2024-01-15T09:35:00Z",
      "updated_at": "2024-01-15T09:35:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20,
  "pages": 1
}
```

---

### Execute Trade

Execute a trade based on a signal:

```bash
curl -X POST "${API_BASE}/api/v1/trades" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "signal_id": "550e8400-e29b-41d4-a716-446655440000",
    "quantity": "100"
  }'
```

#### With Limit Price

```bash
curl -X POST "${API_BASE}/api/v1/trades" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "signal_id": "550e8400-e29b-41d4-a716-446655440000",
    "quantity": "50",
    "limit_price": "185.00"
  }'
```

#### Response (201 Created)

```json
{
  "id": "990e8400-e29b-41d4-a716-446655440004",
  "signal_id": "550e8400-e29b-41d4-a716-446655440000",
  "ticker": "AAPL",
  "direction": "BUY",
  "quantity": "100",
  "entry_price": null,
  "exit_price": null,
  "status": "PENDING",
  "pnl": null,
  "fees": "0",
  "filled_at": null,
  "closed_at": null,
  "notional_value": null,
  "return_pct": null,
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:00:00Z"
}
```

---

### Get Trade by ID

```bash
curl -X GET "${API_BASE}/api/v1/trades/880e8400-e29b-41d4-a716-446655440003" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Accept: application/json"
```

---

### Get Trade Summary

Get P&L summary:

```bash
curl -X GET "${API_BASE}/api/v1/trades/summary" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Accept: application/json"
```

#### With Date Range

```bash
curl -X GET "${API_BASE}/api/v1/trades/summary?date_from=2024-01-01T00:00:00Z&date_to=2024-01-31T23:59:59Z" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Accept: application/json"
```

#### Response

```json
{
  "total_trades": 50,
  "open_trades": 5,
  "closed_trades": 45,
  "winning_trades": 28,
  "losing_trades": 17,
  "total_pnl": "12500.00",
  "total_fees": "125.50",
  "net_pnl": "12374.50",
  "win_rate": 0.622,
  "average_win": "625.00",
  "average_loss": "-250.00",
  "profit_factor": "2.50",
  "period_start": "2024-01-01T00:00:00Z",
  "period_end": "2024-01-31T23:59:59Z"
}
```

---

## Portfolio Operations

### Get Portfolio Overview

Get comprehensive portfolio status:

```bash
curl -X GET "${API_BASE}/api/v1/portfolio" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Accept: application/json"
```

#### Response

```json
{
  "total_value": "150000.00",
  "cash_balance": "75000.00",
  "positions_value": "75000.00",
  "total_unrealized_pnl": "2500.00",
  "total_realized_pnl": "12374.50",
  "position_count": 8,
  "long_positions": 6,
  "short_positions": 2,
  "buying_power": "150000.00",
  "margin_used": "0.00",
  "day_pnl": "850.25",
  "day_pnl_pct": 0.57,
  "updated_at": "2024-01-15T15:30:00Z"
}
```

---

### List Positions

Get open positions:

```bash
curl -X GET "${API_BASE}/api/v1/portfolio/positions" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Accept: application/json"
```

#### Filter by Side

```bash
curl -X GET "${API_BASE}/api/v1/portfolio/positions?side=long" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Accept: application/json"
```

#### Response

```json
{
  "items": [
    {
      "id": "aa0e8400-e29b-41d4-a716-446655440005",
      "ticker": "AAPL",
      "quantity": "100",
      "average_price": "185.50",
      "current_price": "188.25",
      "market_value": "18825.00",
      "cost_basis": "18550.00",
      "unrealized_pnl": "275.00",
      "return_pct": "1.48",
      "is_long": true,
      "opened_at": "2024-01-15T09:35:00Z",
      "created_at": "2024-01-15T09:35:00Z",
      "updated_at": "2024-01-15T15:30:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20,
  "pages": 1
}
```

---

### Get Performance Metrics

Get portfolio performance:

```bash
curl -X GET "${API_BASE}/api/v1/portfolio/performance" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Accept: application/json"
```

#### With Benchmark Comparison

```bash
curl -X GET "${API_BASE}/api/v1/portfolio/performance?benchmark=SPY&date_from=2024-01-01T00:00:00Z" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Accept: application/json"
```

#### Response

```json
{
  "total_return": "14874.50",
  "total_return_pct": 9.92,
  "daily_return_pct": 0.57,
  "weekly_return_pct": 2.15,
  "monthly_return_pct": 5.25,
  "ytd_return_pct": 9.92,
  "sharpe_ratio": 1.85,
  "sortino_ratio": 2.45,
  "max_drawdown": -3.25,
  "max_drawdown_duration": 5,
  "volatility": 12.5,
  "beta": 0.92,
  "alpha": 0.035,
  "win_rate": 0.622,
  "profit_factor": "2.50",
  "calmar_ratio": 3.05,
  "period_start": "2024-01-01T00:00:00Z",
  "period_end": "2024-01-15T15:30:00Z"
}
```

---

### Close Position

Close an entire position:

```bash
curl -X POST "${API_BASE}/api/v1/portfolio/positions/AAPL/close" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{}'
```

#### Partial Close with Limit Price

```bash
curl -X POST "${API_BASE}/api/v1/portfolio/positions/AAPL/close" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "quantity": "50",
    "limit_price": "188.00"
  }'
```

#### Response

```json
{
  "ticker": "AAPL",
  "quantity_closed": "50",
  "exit_price": "188.25",
  "realized_pnl": "137.50",
  "remaining_quantity": "50",
  "status": "FILLED",
  "message": "Successfully closed 50 shares of AAPL"
}
```

---

## Error Examples

### Validation Error (400)

```bash
curl -X POST "${API_BASE}/api/v1/signals" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "TOOLONG",
    "direction": "BUY",
    "confidence": 1.5
  }'
```

Response:
```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "Validation failed",
  "details": {
    "ticker": "String should have at most 5 characters",
    "confidence": "Input should be less than or equal to 1"
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "correlation_id": "abc123-def456"
}
```

### Not Found (404)

```bash
curl -X GET "${API_BASE}/api/v1/signals/nonexistent-uuid" \
  -H "Authorization: Bearer ${API_TOKEN}"
```

Response:
```json
{
  "error_code": "SIGNAL_NOT_FOUND",
  "message": "Signal not found",
  "details": null,
  "timestamp": "2024-01-15T10:30:00Z",
  "correlation_id": "abc123-def456"
}
```

---

## Tips

1. **Use jq for formatting**: Pipe responses to `jq` for pretty printing:
   ```bash
   curl -s ... | jq .
   ```

2. **Save responses**: Redirect to files for later analysis:
   ```bash
   curl -s ... > response.json
   ```

3. **Verbose mode**: Use `-v` flag to see headers:
   ```bash
   curl -v ...
   ```

4. **Time requests**: Use `time` to measure API latency:
   ```bash
   time curl -s ...
   ```
