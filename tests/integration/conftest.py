"""Integration test fixtures for GPT Quant API testing.

Provides fixtures for:
- Async HTTP client with TestClient
- In-memory SQLite database for isolation
- Redis mock for caching tests
- Settings overrides for test environment
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gpt_quant_v2.interfaces.api.main import create_app
from gpt_quant_v2.shared.config.settings import Settings


@pytest.fixture
def settings_override() -> Settings:
    """Override settings for test environment.

    Returns:
        Settings configured for test isolation with in-memory SQLite
        and minimal external dependencies.
    """
    return Settings(
        _env_file=None,  # Don't load .env files in tests
    )


@pytest.fixture
async def test_db() -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    """Create an in-memory SQLite database for testing.

    Provides an isolated database session factory that creates
    fresh tables for each test. SQLite is used for speed and simplicity.

    Yields:
        Async session factory bound to the test database.
    """
    # Use SQLite for in-memory testing (aiosqlite driver)
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    # Create session factory
    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create tables if needed (import models to register with metadata)
    # For now, we don't have ORM models so we skip table creation
    async with engine.begin():
        pass  # Tables would be created here if we had ORM models

    yield session_factory

    # Cleanup
    await engine.dispose()


@pytest.fixture
def redis_mock() -> MagicMock:
    """Create a mock Redis client for testing.

    Returns:
        MagicMock configured to behave like an async Redis client
        with common operations returning sensible defaults.
    """
    mock = MagicMock()

    # Configure async methods
    mock.ping = AsyncMock(return_value=True)
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.close = AsyncMock(return_value=None)
    mock.exists = AsyncMock(return_value=0)
    mock.expire = AsyncMock(return_value=True)
    mock.incr = AsyncMock(return_value=1)
    mock.decr = AsyncMock(return_value=0)
    mock.lpush = AsyncMock(return_value=1)
    mock.rpush = AsyncMock(return_value=1)
    mock.lpop = AsyncMock(return_value=None)
    mock.rpop = AsyncMock(return_value=None)
    mock.lrange = AsyncMock(return_value=[])
    mock.hset = AsyncMock(return_value=1)
    mock.hget = AsyncMock(return_value=None)
    mock.hgetall = AsyncMock(return_value={})
    mock.hdel = AsyncMock(return_value=1)
    mock.publish = AsyncMock(return_value=1)

    return mock


@pytest.fixture
async def async_client(
    settings_override: Settings,
    test_db: async_sessionmaker[AsyncSession],
    redis_mock: MagicMock,
) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client for API testing.

    This fixture creates a FastAPI test application with:
    - Overridden settings for test environment
    - In-memory SQLite database
    - Mocked Redis connection

    Args:
        settings_override: Test settings fixture
        test_db: In-memory database session factory
        redis_mock: Mocked Redis client

    Yields:
        AsyncClient configured to send requests to the test app.
    """
    # Create app with test settings
    app = create_app(settings=settings_override)

    # Override app state with test dependencies
    app.state.db_session_factory = test_db
    app.state.db_engine = None  # Not needed for tests
    app.state.redis = redis_mock

    # Create async client with ASGI transport
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        yield client


@pytest.fixture
def mock_db_session() -> AsyncMock:
    """Create a mock database session for unit-style tests.

    Returns:
        AsyncMock configured as a SQLAlchemy async session.
    """
    session = AsyncMock(spec=AsyncSession)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.delete = MagicMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def sample_signal_payload() -> dict[str, Any]:
    """Sample valid signal creation payload."""
    return {
        "ticker": "AAPL",
        "direction": "BUY",
        "confidence": 0.85,
        "source": "COMPOSITE",
        "predicted_return": "0.05",
        "metadata": {"strategy": "momentum"},
    }


@pytest.fixture
def sample_trade_payload() -> dict[str, Any]:
    """Sample valid trade creation payload."""
    return {
        "signal_id": "00000000-0000-0000-0000-000000000001",
        "quantity": "100",
        "limit_price": "150.00",
    }
