"""Integration tests for the Portfolio API endpoints.

Tests cover:
- Getting portfolio overview
- Listing positions
- Getting performance metrics
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_portfolio_empty(async_client: AsyncClient) -> None:
    """Test getting portfolio overview returns zero values when empty."""
    response = await async_client.get("/api/v1/portfolio")
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify portfolio structure with zero values
    assert data["total_value"] == "0"
    assert data["cash_balance"] == "0"
    assert data["positions_value"] == "0"
    assert data["total_unrealized_pnl"] == "0"
    assert data["total_realized_pnl"] == "0"
    assert data["position_count"] == 0
    assert data["long_positions"] == 0
    assert data["short_positions"] == 0
    assert data["buying_power"] == "0"
    assert data["margin_used"] == "0"
    assert data["day_pnl"] == "0"
    assert data["day_pnl_pct"] == 0.0
    assert "updated_at" in data


@pytest.mark.asyncio
async def test_get_positions_empty(async_client: AsyncClient) -> None:
    """Test listing positions returns empty paginated response when no positions exist."""
    response = await async_client.get("/api/v1/portfolio/positions")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["items"] == []
    assert data["total"] == 0
    assert data["page"] == 1
    assert data["page_size"] == 20
    assert data["total_pages"] == 0
    assert data["has_next"] is False
    assert data["has_previous"] is False


@pytest.mark.asyncio
async def test_get_positions_with_pagination(async_client: AsyncClient) -> None:
    """Test listing positions with custom pagination parameters."""
    response = await async_client.get(
        "/api/v1/portfolio/positions",
        params={"page": 1, "page_size": 10},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["page"] == 1
    assert data["page_size"] == 10


@pytest.mark.asyncio
async def test_get_positions_with_ticker_filter(async_client: AsyncClient) -> None:
    """Test listing positions filtered by ticker."""
    response = await async_client.get(
        "/api/v1/portfolio/positions",
        params={"ticker": "AAPL"},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "items" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_get_positions_with_side_filter_long(async_client: AsyncClient) -> None:
    """Test listing positions filtered by long side."""
    response = await async_client.get(
        "/api/v1/portfolio/positions",
        params={"side": "long"},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "items" in data


@pytest.mark.asyncio
async def test_get_positions_with_side_filter_short(async_client: AsyncClient) -> None:
    """Test listing positions filtered by short side."""
    response = await async_client.get(
        "/api/v1/portfolio/positions",
        params={"side": "short"},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "items" in data


@pytest.mark.asyncio
async def test_get_positions_invalid_side(async_client: AsyncClient) -> None:
    """Test listing positions with invalid side filter returns 422."""
    response = await async_client.get(
        "/api/v1/portfolio/positions",
        params={"side": "invalid"},
    )
    
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_positions_invalid_page(async_client: AsyncClient) -> None:
    """Test listing positions with invalid page returns 422."""
    response = await async_client.get(
        "/api/v1/portfolio/positions",
        params={"page": 0},
    )
    
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_performance_metrics_empty(async_client: AsyncClient) -> None:
    """Test getting performance metrics returns zero/null values when empty."""
    response = await async_client.get("/api/v1/portfolio/performance")
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify performance structure
    assert data["total_return"] == "0"
    assert data["total_return_pct"] == 0.0
    assert data["daily_return_pct"] is None
    assert data["weekly_return_pct"] is None
    assert data["monthly_return_pct"] is None
    assert data["ytd_return_pct"] is None
    assert data["sharpe_ratio"] is None
    assert data["sortino_ratio"] is None
    assert data["max_drawdown"] is None
    assert data["volatility"] is None
    assert data["beta"] is None
    assert data["alpha"] is None
    assert data["win_rate"] is None
    assert data["profit_factor"] is None


@pytest.mark.asyncio
async def test_get_performance_metrics_with_date_range(async_client: AsyncClient) -> None:
    """Test getting performance metrics with date range filters."""
    response = await async_client.get(
        "/api/v1/portfolio/performance",
        params={
            "date_from": "2024-01-01T00:00:00Z",
            "date_to": "2024-12-31T23:59:59Z",
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["period_start"] == "2024-01-01T00:00:00Z"
    assert data["period_end"] == "2024-12-31T23:59:59Z"


@pytest.mark.asyncio
async def test_get_performance_metrics_with_benchmark(async_client: AsyncClient) -> None:
    """Test getting performance metrics with benchmark comparison."""
    response = await async_client.get(
        "/api/v1/portfolio/performance",
        params={"benchmark": "SPY"},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "beta" in data
    assert "alpha" in data


@pytest.mark.asyncio
async def test_close_position_not_found(async_client: AsyncClient) -> None:
    """Test closing a non-existent position returns 404."""
    response = await async_client.post(
        "/api/v1/portfolio/positions/AAPL/close",
        json={},
    )
    
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower() or "no open position" in data["detail"].lower()


@pytest.mark.asyncio
async def test_close_position_with_quantity(async_client: AsyncClient) -> None:
    """Test closing a position with specific quantity."""
    response = await async_client.post(
        "/api/v1/portfolio/positions/TSLA/close",
        json={"quantity": "50"},
    )
    
    # Should return 404 since no position exists
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_close_position_with_limit_price(async_client: AsyncClient) -> None:
    """Test closing a position with limit price."""
    response = await async_client.post(
        "/api/v1/portfolio/positions/GOOG/close",
        json={"limit_price": "150.00"},
    )
    
    # Should return 404 since no position exists
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_close_position_invalid_ticker(async_client: AsyncClient) -> None:
    """Test closing a position with invalid ticker format returns 400."""
    response = await async_client.post(
        "/api/v1/portfolio/positions/TOOLONG123/close",
        json={},
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "invalid ticker" in data["detail"].lower()


@pytest.mark.asyncio
async def test_close_position_invalid_quantity(async_client: AsyncClient) -> None:
    """Test closing a position with invalid quantity returns 422."""
    response = await async_client.post(
        "/api/v1/portfolio/positions/AAPL/close",
        json={"quantity": "-50"},  # Negative not allowed
    )
    
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_close_position_invalid_limit_price(async_client: AsyncClient) -> None:
    """Test closing a position with invalid limit price returns 422."""
    response = await async_client.post(
        "/api/v1/portfolio/positions/AAPL/close",
        json={"limit_price": "0"},  # Must be > 0
    )
    
    assert response.status_code == 422
