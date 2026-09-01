"""Integration tests for the Health check endpoint.

Tests cover:
- Basic health check returns 200
- Health response includes all required fields
- Health response structure validation
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint_returns_200(async_client: AsyncClient) -> None:
    """Test that health endpoint returns 200 OK status."""
    response = await async_client.get("/health")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_includes_status(async_client: AsyncClient) -> None:
    """Test that health response includes status field with 'healthy' value."""
    response = await async_client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_includes_version(async_client: AsyncClient) -> None:
    """Test that health response includes version field."""
    response = await async_client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert "version" in data
    assert isinstance(data["version"], str)
    assert len(data["version"]) > 0


@pytest.mark.asyncio
async def test_health_includes_timestamp(async_client: AsyncClient) -> None:
    """Test that health response includes timestamp field."""
    response = await async_client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert "timestamp" in data
    # Timestamp should be ISO format string
    assert isinstance(data["timestamp"], str)


@pytest.mark.asyncio
async def test_health_includes_database_status(async_client: AsyncClient) -> None:
    """Test that health response includes database connection status."""
    response = await async_client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert "database" in data
    # In test environment, database might be disconnected or error
    assert data["database"] in ["connected", "disconnected", "error", "unknown"]


@pytest.mark.asyncio
async def test_health_includes_redis_status(async_client: AsyncClient) -> None:
    """Test that health response includes Redis connection status."""
    response = await async_client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert "redis" in data
    # With mocked redis, should show as connected
    assert data["redis"] in ["connected", "disconnected", "error", "unknown"]


@pytest.mark.asyncio
async def test_health_response_structure(async_client: AsyncClient) -> None:
    """Test that health response has complete expected structure."""
    response = await async_client.get("/health")

    assert response.status_code == 200
    data = response.json()

    # Verify all expected fields are present
    expected_fields = ["status", "version", "timestamp", "database", "redis"]
    for field in expected_fields:
        assert field in data, f"Missing field: {field}"


@pytest.mark.asyncio
async def test_health_response_content_type(async_client: AsyncClient) -> None:
    """Test that health endpoint returns JSON content type."""
    response = await async_client.get("/health")

    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_health_endpoint_is_fast(async_client: AsyncClient) -> None:
    """Test that health endpoint responds quickly (under 1 second)."""
    import time

    start = time.monotonic()
    response = await async_client.get("/health")
    elapsed = time.monotonic() - start

    assert response.status_code == 200
    assert elapsed < 1.0, f"Health check took too long: {elapsed:.2f}s"
