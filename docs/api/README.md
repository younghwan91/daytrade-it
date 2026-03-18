# GPT Quant V2 API Documentation

## Overview

The GPT Quant V2 API provides programmatic access to an AI-driven algorithmic trading system. It enables you to:

- **Generate Trading Signals**: Leverage ML models to analyze news and market data
- **Execute Trades**: Automate trade execution with risk management
- **Manage Portfolio**: Track positions, P&L, and performance metrics
- **Stream Real-time Data**: WebSocket connections for live updates

### Base URL

```
Production: https://api.gptquant.example.com
Development: http://localhost:8000
```

### API Version

Current version: `v1`

All endpoints are prefixed with `/api/v1/`

---

## Authentication

> **Note**: Authentication is currently a placeholder for future JWT implementation.

The API uses JWT (JSON Web Token) authentication. Include your token in the `Authorization` header:

```http
Authorization: Bearer <your_jwt_token>
```

### Obtaining a Token

```bash
# Future implementation
curl -X POST https://api.gptquant.example.com/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

### Token Refresh

Tokens expire after 24 hours. Refresh before expiration:

```bash
# Future implementation
curl -X POST https://api.gptquant.example.com/api/v1/auth/refresh \
  -H "Authorization: Bearer <your_current_token>"
```

---

## Rate Limits

The API enforces the following rate limits per user:

| Endpoint Category | Limit | Window |
|-------------------|-------|--------|
| Standard endpoints | 100 requests | per minute |
| Trade execution | 50 requests | per minute |
| Data-intensive (portfolio, metrics) | 20 requests | per minute |
| WebSocket connections | 5 concurrent | - |

### Rate Limit Headers

Each response includes rate limit information:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1699900000
```

### Handling Rate Limits

When rate limited, the API returns HTTP 429:

```json
{
  "error_code": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded. Retry after 30 seconds.",
  "details": {
    "retry_after": 30
  }
}
```

---

## Error Handling

The API uses consistent error response format across all endpoints.

### Error Response Format

```json
{
  "error_code": "ERROR_CODE",
  "message": "Human-readable error message",
  "details": { "field": "additional context" },
  "timestamp": "2024-01-15T10:30:00Z",
  "correlation_id": "uuid-for-tracking"
}
```

### HTTP Status Codes

| Code | Description | Error Type |
|------|-------------|------------|
| 400 | Bad Request | Validation errors, malformed input |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource doesn't exist |
| 422 | Unprocessable Entity | Business logic/domain errors |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Unexpected server error |
| 502 | Bad Gateway | External API failure |
| 503 | Service Unavailable | Infrastructure issues |

### Error Codes Reference

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Input validation failed |
| `AUTHENTICATION_REQUIRED` | 401 | Missing authentication token |
| `INVALID_TOKEN` | 401 | Token is invalid or expired |
| `PERMISSION_DENIED` | 403 | Action not allowed |
| `RESOURCE_NOT_FOUND` | 404 | Requested resource not found |
| `SIGNAL_NOT_FOUND` | 404 | Trading signal doesn't exist |
| `TRADE_NOT_FOUND` | 404 | Trade doesn't exist |
| `POSITION_NOT_FOUND` | 404 | Position doesn't exist |
| `INSUFFICIENT_FUNDS` | 422 | Not enough buying power |
| `MARKET_CLOSED` | 422 | Market is not open |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `EXTERNAL_API_ERROR` | 502 | External service failure |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

### Correlation ID

Every response includes a `correlation_id` for tracking requests across systems. Include this ID when contacting support.

---

## Quick Start

### 1. Health Check

Verify the API is running:

```bash
curl -X GET http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "database": "connected",
  "redis": "connected"
}
```

### 2. List Trading Signals

Get active trading signals:

```bash
curl -X GET "http://localhost:8000/api/v1/signals" \
  -H "Authorization: Bearer <token>" \
  -H "Accept: application/json"
```

Response:
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
      "expires_at": "2024-01-15T16:00:00Z",
      "is_expired": false,
      "created_at": "2024-01-15T09:30:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20,
  "pages": 1
}
```

### 3. Create a Trading Signal

Create a manual signal:

```bash
curl -X POST "http://localhost:8000/api/v1/signals" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "MSFT",
    "direction": "BUY",
    "confidence": 0.75,
    "source": "COMPOSITE",
    "predicted_return": "0.03"
  }'
```

### 4. Execute a Trade

Execute a trade based on a signal:

```bash
curl -X POST "http://localhost:8000/api/v1/trades" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "signal_id": "550e8400-e29b-41d4-a716-446655440000",
    "quantity": "100"
  }'
```

### 5. Get Portfolio Overview

View your portfolio status:

```bash
curl -X GET "http://localhost:8000/api/v1/portfolio" \
  -H "Authorization: Bearer <token>"
```

Response:
```json
{
  "total_value": "100000.00",
  "cash_balance": "50000.00",
  "positions_value": "50000.00",
  "total_unrealized_pnl": "1250.50",
  "position_count": 5,
  "day_pnl": "350.25",
  "day_pnl_pct": 0.35,
  "updated_at": "2024-01-15T10:30:00Z"
}
```

---

## API Endpoints Summary

### Signals
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/signals` | List active signals |
| GET | `/api/v1/signals/{id}` | Get signal by ID |
| POST | `/api/v1/signals` | Create manual signal |
| DELETE | `/api/v1/signals/{id}` | Delete/cancel signal |

### Trades
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/trades` | List trades |
| GET | `/api/v1/trades/{id}` | Get trade by ID |
| GET | `/api/v1/trades/summary` | Get P&L summary |
| POST | `/api/v1/trades` | Execute trade |

### Portfolio
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/portfolio` | Portfolio overview |
| GET | `/api/v1/portfolio/positions` | List positions |
| GET | `/api/v1/portfolio/performance` | Performance metrics |
| POST | `/api/v1/portfolio/positions/{ticker}/close` | Close position |

### News
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/news` | List news articles |
| GET | `/api/v1/news/{id}` | Get article by ID |

### WebSocket
| Endpoint | Description |
|----------|-------------|
| `/ws/signals` | Real-time signal updates |
| `/ws/trades` | Real-time trade updates |
| `/ws/portfolio` | Real-time portfolio updates |

---

## Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Spec**: `http://localhost:8000/openapi.json`

---

## SDKs and Libraries

Official SDKs are planned for future releases:

- Python SDK (planned)
- TypeScript SDK (planned)

---

## Changelog

### v0.1.0 (Current)
- Initial API release
- Signal generation and management
- Trade execution
- Portfolio management
- News integration
