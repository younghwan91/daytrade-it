"""Integration tests for the News API router's ticker validation.

Regression coverage for Task 5's review finding (fixed in Task 14): the
``analyze_news`` endpoint used to gate on ``ticker.isalpha()`` + a 1-5 char
length, rejecting every real KRX 6-digit ticker. These tests prove a real
KRX code (e.g. Samsung Electronics' "005930") is accepted and that
non-KRX-shaped input is rejected with a clear KRX-flavored error message.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_analyze_news_accepts_krx_ticker(async_client: AsyncClient) -> None:
    """A real 6-digit KRX ticker must be accepted (not rejected as invalid)."""
    response = await async_client.post(
        "/api/v1/news/analyze",
        json={"ticker": "005930", "hours": 24},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["ticker"] == "005930"


@pytest.mark.asyncio
async def test_analyze_news_rejects_us_style_ticker(async_client: AsyncClient) -> None:
    """A US-style letter ticker no longer matches the (now numeric) KRX format.

    Pydantic's `min_length=6, max_length=6` field constraint on the request
    body rejects a 4-letter ticker like "AAPL" before the handler even runs,
    so this surfaces as a 422 (schema validation), not the handler's 400.
    """
    response = await async_client.post(
        "/api/v1/news/analyze",
        json={"ticker": "AAPL", "hours": 24},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_analyze_news_rejects_malformed_6_char_ticker(async_client: AsyncClient) -> None:
    """A 6-character but non-numeric value must be rejected by the handler's
    KRX-format check (400), not silently accepted."""
    response = await async_client.post(
        "/api/v1/news/analyze",
        json={"ticker": "ABCDEF", "hours": 24},
    )

    assert response.status_code == 400
    data = response.json()
    assert "invalid ticker" in data["detail"].lower()
