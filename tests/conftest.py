"""
Pytest Configuration and Fixtures
"""

import pytest


@pytest.fixture
def sample_ticker() -> str:
    """Sample ticker symbol for testing."""
    return "AAPL"


@pytest.fixture
def sample_news_content() -> dict:
    """Sample news article content."""
    return {
        "title": "Apple Reports Record Quarterly Revenue",
        "content": "Apple Inc. announced record quarterly revenue today...",
        "source": "Reuters",
        "published_at": "2024-01-15T10:30:00Z",
    }


@pytest.fixture
def sample_price_data() -> dict:
    """Sample price data."""
    return {
        "open": 185.50,
        "high": 188.25,
        "low": 184.75,
        "close": 187.50,
        "volume": 45000000,
    }
