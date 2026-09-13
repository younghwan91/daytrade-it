"""Integration tests for the Trades API endpoints.

Tests cover:
- Listing trades (empty and populated)
- Getting trade summary
- Getting individual trades
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_trades_empty(async_client: AsyncClient) -> None:
    """Test listing trades returns empty paginated response when no trades exist."""
    response = await async_client.get("/api/v1/trades")

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
async def test_get_trade_summary_empty(async_client: AsyncClient) -> None:
    """Test getting trade summary returns zero values when no trades exist."""
    response = await async_client.get("/api/v1/trades/summary")

    assert response.status_code == 200
    data = response.json()

    # Verify summary structure with zero values
    assert data["total_trades"] == 0
    assert data["open_trades"] == 0
    assert data["closed_trades"] == 0
    assert data["winning_trades"] == 0
    assert data["losing_trades"] == 0
    assert data["total_pnl"] == "0"
    assert data["total_fees"] == "0"
    assert data["net_pnl"] == "0"
    assert data["win_rate"] == 0.0
    assert data["average_win"] is None
    assert data["average_loss"] is None
    assert data["profit_factor"] is None


@pytest.mark.asyncio
async def test_get_trade_summary_with_date_filters(async_client: AsyncClient) -> None:
    """Test getting trade summary with date range filters."""
    response = await async_client.get(
        "/api/v1/trades/summary",
        params={
            "date_from": "2024-01-01T00:00:00Z",
            "date_to": "2024-12-31T23:59:59Z",
            "ticker": "005930",
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify period dates are reflected
    assert data["period_start"] == "2024-01-01T00:00:00Z"
    assert data["period_end"] == "2024-12-31T23:59:59Z"


@pytest.mark.asyncio
async def test_get_trade_not_found(async_client: AsyncClient) -> None:
    """Test getting a non-existent trade returns 404."""
    non_existent_id = uuid4()

    response = await async_client.get(f"/api/v1/trades/{non_existent_id}")

    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


@pytest.mark.asyncio
async def test_get_trade_invalid_uuid(async_client: AsyncClient) -> None:
    """Test getting a trade with invalid UUID format returns 422."""
    response = await async_client.get("/api/v1/trades/not-a-valid-uuid")

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_trades_with_pagination(async_client: AsyncClient) -> None:
    """Test listing trades with custom pagination parameters."""
    response = await async_client.get(
        "/api/v1/trades",
        params={"page": 1, "page_size": 50},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["page"] == 1
    assert data["page_size"] == 50


@pytest.mark.asyncio
async def test_list_trades_with_ticker_filter(async_client: AsyncClient) -> None:
    """Test listing trades filtered by ticker."""
    response = await async_client.get(
        "/api/v1/trades",
        params={"ticker": "000660"},
    )

    assert response.status_code == 200
    data = response.json()

    assert "items" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_list_trades_with_status_filter(async_client: AsyncClient) -> None:
    """Test listing trades filtered by status."""
    response = await async_client.get(
        "/api/v1/trades",
        params={"status": "FILLED"},
    )

    assert response.status_code == 200
    data = response.json()

    assert "items" in data


@pytest.mark.asyncio
async def test_list_trades_with_date_range(async_client: AsyncClient) -> None:
    """Test listing trades filtered by date range."""
    response = await async_client.get(
        "/api/v1/trades",
        params={
            "date_from": "2024-01-01T00:00:00Z",
            "date_to": "2024-06-30T23:59:59Z",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "items" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_list_trades_invalid_page(async_client: AsyncClient) -> None:
    """Test listing trades with invalid page number returns 422."""
    response = await async_client.get(
        "/api/v1/trades",
        params={"page": -1},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_trades_invalid_page_size(async_client: AsyncClient) -> None:
    """Test listing trades with page_size > 100 returns 422."""
    response = await async_client.get(
        "/api/v1/trades",
        params={"page_size": 200},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_trade_signal_not_found(async_client: AsyncClient) -> None:
    """Test creating a trade with non-existent signal returns 404."""
    payload = {
        "signal_id": str(uuid4()),
        "quantity": "100",
    }

    response = await async_client.post("/api/v1/trades", json=payload)

    assert response.status_code == 404
    data = response.json()
    assert "signal" in data["detail"].lower()


@pytest.mark.asyncio
async def test_create_trade_invalid_quantity(async_client: AsyncClient) -> None:
    """Test creating a trade with invalid quantity returns 422."""
    payload = {
        "signal_id": str(uuid4()),
        "quantity": "-100",  # Negative quantity not allowed
    }

    response = await async_client.post("/api/v1/trades", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_trade_invalid_limit_price(async_client: AsyncClient) -> None:
    """Test creating a trade with invalid limit price returns 422."""
    payload = {
        "signal_id": str(uuid4()),
        "quantity": "100",
        "limit_price": "0",  # Must be > 0
    }

    response = await async_client.post("/api/v1/trades", json=payload)

    assert response.status_code == 422
