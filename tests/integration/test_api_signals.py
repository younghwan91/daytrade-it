"""Integration tests for the Signals API endpoints.

Tests cover:
- Listing signals (empty and populated)
- Creating signals with valid and invalid data
- Getting individual signals
- Deleting signals
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_signals_empty(async_client: AsyncClient) -> None:
    """Test listing signals returns empty paginated response when no signals exist."""
    response = await async_client.get("/api/v1/signals")

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
async def test_create_signal_success(
    async_client: AsyncClient,
    sample_signal_payload: dict[str, Any],
) -> None:
    """Test creating a signal with valid data returns 201 and signal details."""
    response = await async_client.post(
        "/api/v1/signals",
        json=sample_signal_payload,
    )

    assert response.status_code == 201
    data = response.json()

    # Verify response structure
    assert "id" in data
    assert data["ticker"] == sample_signal_payload["ticker"].upper()
    assert data["direction"] == sample_signal_payload["direction"]
    assert data["confidence"] == sample_signal_payload["confidence"]
    assert data["source"] == sample_signal_payload["source"]
    assert "created_at" in data
    assert "updated_at" in data
    assert "is_expired" in data


@pytest.mark.asyncio
async def test_create_signal_minimal_payload(async_client: AsyncClient) -> None:
    """Test creating a signal with minimal required fields."""
    payload = {
        "ticker": "000660",
        "direction": "SELL",
        "confidence": 0.70,
    }

    response = await async_client.post("/api/v1/signals", json=payload)

    assert response.status_code == 201
    data = response.json()

    assert data["ticker"] == "000660"
    assert data["direction"] == "SELL"
    assert data["confidence"] == 0.70


@pytest.mark.asyncio
async def test_create_signal_invalid_ticker_too_long(async_client: AsyncClient) -> None:
    """Test creating a signal with ticker longer than 5 chars returns 422."""
    payload = {
        "ticker": "TOOLONG",
        "direction": "BUY",
        "confidence": 0.50,
    }

    response = await async_client.post("/api/v1/signals", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_signal_invalid_ticker_empty(async_client: AsyncClient) -> None:
    """Test creating a signal with empty ticker returns 422."""
    payload = {
        "ticker": "",
        "direction": "BUY",
        "confidence": 0.50,
    }

    response = await async_client.post("/api/v1/signals", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_signal_invalid_confidence_too_high(async_client: AsyncClient) -> None:
    """Test creating a signal with confidence > 1.0 returns 422."""
    payload = {
        "ticker": "005930",
        "direction": "BUY",
        "confidence": 1.5,
    }

    response = await async_client.post("/api/v1/signals", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_signal_invalid_confidence_negative(async_client: AsyncClient) -> None:
    """Test creating a signal with negative confidence returns 422."""
    payload = {
        "ticker": "005930",
        "direction": "BUY",
        "confidence": -0.5,
    }

    response = await async_client.post("/api/v1/signals", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_signal_invalid_direction(async_client: AsyncClient) -> None:
    """Test creating a signal with invalid direction returns 422."""
    payload = {
        "ticker": "005930",
        "direction": "INVALID",
        "confidence": 0.50,
    }

    response = await async_client.post("/api/v1/signals", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_signal_missing_required_field(async_client: AsyncClient) -> None:
    """Test creating a signal without required fields returns 422."""
    payload = {
        "ticker": "005930",
        # missing direction and confidence
    }

    response = await async_client.post("/api/v1/signals", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_signal_not_found(async_client: AsyncClient) -> None:
    """Test getting a non-existent signal returns 404."""
    non_existent_id = uuid4()

    response = await async_client.get(f"/api/v1/signals/{non_existent_id}")

    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


@pytest.mark.asyncio
async def test_get_signal_invalid_uuid(async_client: AsyncClient) -> None:
    """Test getting a signal with invalid UUID format returns 422."""
    response = await async_client.get("/api/v1/signals/not-a-valid-uuid")

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_delete_signal_not_found(async_client: AsyncClient) -> None:
    """Test deleting a non-existent signal returns 404."""
    non_existent_id = uuid4()

    response = await async_client.delete(f"/api/v1/signals/{non_existent_id}")

    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


@pytest.mark.asyncio
async def test_delete_signal_invalid_uuid(async_client: AsyncClient) -> None:
    """Test deleting a signal with invalid UUID format returns 422."""
    response = await async_client.delete("/api/v1/signals/invalid-uuid")

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_signals_with_pagination(async_client: AsyncClient) -> None:
    """Test listing signals with custom pagination parameters."""
    response = await async_client.get(
        "/api/v1/signals",
        params={"page": 2, "page_size": 10},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["page"] == 2
    assert data["page_size"] == 10


@pytest.mark.asyncio
async def test_list_signals_with_filters(async_client: AsyncClient) -> None:
    """Test listing signals with filter parameters."""
    response = await async_client.get(
        "/api/v1/signals",
        params={
            "ticker": "005930",
            "min_confidence": 0.8,
            "direction": "BUY",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "items" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_list_signals_invalid_page(async_client: AsyncClient) -> None:
    """Test listing signals with invalid page number returns 422."""
    response = await async_client.get(
        "/api/v1/signals",
        params={"page": 0},  # Must be >= 1
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_signals_invalid_page_size(async_client: AsyncClient) -> None:
    """Test listing signals with page_size > 100 returns 422."""
    response = await async_client.get(
        "/api/v1/signals",
        params={"page_size": 500},  # Max is 100
    )

    assert response.status_code == 422
