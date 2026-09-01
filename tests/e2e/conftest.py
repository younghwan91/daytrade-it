"""
E2E Test Configuration and Fixtures

Provides fixtures for:
- Docker compose management (TimescaleDB + Redis)
- Database seeding with test data
- API client with authentication
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio

# Default test configuration
E2E_BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost:8000")
E2E_DOCKER_COMPOSE_PATH = os.environ.get(
    "E2E_DOCKER_COMPOSE_PATH",
    os.path.join(os.path.dirname(__file__), "../../docker/docker-compose.yml"),
)
E2E_SKIP_DOCKER = os.environ.get("E2E_SKIP_DOCKER", "false").lower() == "true"
E2E_TIMEOUT = int(os.environ.get("E2E_TIMEOUT", "60"))


# ============================================================================
# Docker Compose Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def docker_compose_file() -> str:
    """Path to docker-compose.yml for E2E tests."""
    return E2E_DOCKER_COMPOSE_PATH


@pytest.fixture(scope="session")
def docker_compose_project_name() -> str:
    """Docker compose project name for test isolation."""
    return "gpt_quant_e2e_test"


@pytest.fixture(scope="session")
def docker_services(
    docker_compose_file: str,
    docker_compose_project_name: str,
) -> Generator[dict[str, Any], None, None]:
    """Start and manage Docker services for E2E tests.

    Starts TimescaleDB and Redis containers before tests,
    and tears them down after all tests complete.

    Set E2E_SKIP_DOCKER=true to skip Docker management
    (useful when services are already running).
    """
    if E2E_SKIP_DOCKER:
        yield {"skipped": True}
        return

    compose_cmd = [
        "docker",
        "compose",
        "-f",
        docker_compose_file,
        "-p",
        docker_compose_project_name,
    ]

    # Check if docker compose is available
    try:
        subprocess.run(
            ["docker", "compose", "version"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("Docker Compose not available")

    # Start services
    try:
        # Set required env vars for docker-compose
        env = os.environ.copy()
        env.setdefault("POSTGRES_PASSWORD", "test_password")
        env.setdefault("POSTGRES_USER", "gpt_quant")
        env.setdefault("POSTGRES_DB", "gpt_quant_test")
        env.setdefault("POSTGRES_PORT", "5433")  # Use different port to avoid conflicts
        env.setdefault("REDIS_PORT", "6380")  # Use different port to avoid conflicts

        subprocess.run(
            [*compose_cmd, "up", "-d", "--wait"],
            check=True,
            capture_output=True,
            env=env,
        )

        # Wait for services to be healthy
        _wait_for_services(compose_cmd, timeout=E2E_TIMEOUT, env=env)

        yield {
            "postgres_host": "localhost",
            "postgres_port": int(env["POSTGRES_PORT"]),
            "postgres_user": env["POSTGRES_USER"],
            "postgres_password": env["POSTGRES_PASSWORD"],
            "postgres_db": env["POSTGRES_DB"],
            "redis_host": "localhost",
            "redis_port": int(env["REDIS_PORT"]),
        }

    finally:
        # Cleanup - stop and remove containers
        subprocess.run(
            [*compose_cmd, "down", "-v", "--remove-orphans"],
            capture_output=True,
            env=env,
        )


def _wait_for_services(
    compose_cmd: list[str],
    timeout: int = 60,
    env: dict[str, str] | None = None,
) -> None:
    """Wait for all Docker services to be healthy."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        result = subprocess.run(
            [*compose_cmd, "ps", "--format", "json"],
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode == 0:
            # Check if any service is still starting
            output = result.stdout.strip()
            if output and "unhealthy" not in output.lower() and "starting" not in output.lower():
                return

        time.sleep(2)

    raise TimeoutError(f"Docker services did not become healthy within {timeout}s")


# ============================================================================
# Database Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def database_url(docker_services: dict[str, Any]) -> str:
    """Get database URL for E2E tests."""
    if docker_services.get("skipped"):
        # Use default localhost settings when Docker is skipped
        return os.environ.get(
            "E2E_DATABASE_URL",
            "postgresql+asyncpg://gpt_quant:test_password@localhost:5432/gpt_quant",
        )

    return (
        f"postgresql+asyncpg://"
        f"{docker_services['postgres_user']}:{docker_services['postgres_password']}"
        f"@{docker_services['postgres_host']}:{docker_services['postgres_port']}"
        f"/{docker_services['postgres_db']}"
    )


@pytest.fixture(scope="session")
def redis_url(docker_services: dict[str, Any]) -> str:
    """Get Redis URL for E2E tests."""
    if docker_services.get("skipped"):
        return os.environ.get("E2E_REDIS_URL", "redis://localhost:6379")

    return f"redis://{docker_services['redis_host']}:{docker_services['redis_port']}"


@pytest_asyncio.fixture(scope="function")
async def seed_database() -> AsyncGenerator[dict[str, Any], None]:
    """Seed database with test data for E2E tests.

    Creates sample news articles, signals, trades, and positions
    for testing the full trading workflow.

    Returns a dict with IDs of created entities for use in tests.
    """
    # Test data seed - this would normally insert into the database
    # For now, we provide the expected test data structure

    test_data = {
        "news_articles": [
            {
                "id": uuid4(),
                "title": "Apple Reports Record Quarterly Revenue",
                "content": (
                    "Apple Inc. (AAPL) announced record quarterly revenue of $124 billion, "
                    "exceeding analyst expectations by 5%. The company's services segment "
                    "showed particularly strong growth at 17% year-over-year."
                ),
                "source": "Reuters",
                "url": "https://example.com/news/aapl-earnings",
                "published_at": datetime.now(UTC) - timedelta(hours=2),
                "tickers": ["AAPL"],
            },
            {
                "id": uuid4(),
                "title": "Tech Sector Shows Strong Momentum",
                "content": (
                    "The technology sector continues to show strong momentum with "
                    "major companies reporting better-than-expected earnings. "
                    "Microsoft and Google parent Alphabet both beat estimates."
                ),
                "source": "Bloomberg",
                "url": "https://example.com/news/tech-earnings",
                "published_at": datetime.now(UTC) - timedelta(hours=1),
                "tickers": ["MSFT", "GOOGL"],
            },
        ],
        "signals": [
            {
                "id": uuid4(),
                "ticker": "AAPL",
                "direction": "BUY",
                "confidence": 0.85,
                "source": "NEWS",
                "predicted_return": Decimal("2.5"),
                "predicted_drawdown": Decimal("-1.0"),
                "risk_reward_ratio": Decimal("2.5"),
                "expires_at": datetime.now(UTC) + timedelta(hours=24),
            },
            {
                "id": uuid4(),
                "ticker": "MSFT",
                "direction": "BUY",
                "confidence": 0.78,
                "source": "NEWS",
                "predicted_return": Decimal("1.8"),
                "predicted_drawdown": Decimal("-0.8"),
                "risk_reward_ratio": Decimal("2.25"),
                "expires_at": datetime.now(UTC) + timedelta(hours=24),
            },
        ],
        "trades": [
            {
                "id": uuid4(),
                "ticker": "AAPL",
                "direction": "BUY",
                "quantity": Decimal("100"),
                "entry_price": Decimal("185.50"),
                "status": "FILLED",
                "fees": Decimal("1.00"),
            },
        ],
        "positions": [
            {
                "id": uuid4(),
                "ticker": "AAPL",
                "quantity": Decimal("100"),
                "average_price": Decimal("185.50"),
                "current_price": Decimal("187.25"),
            },
        ],
    }

    yield test_data

    # Cleanup would happen here in a real implementation
    # await cleanup_test_data(test_data)


# ============================================================================
# API Client Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def api_base_url() -> str:
    """Base URL for API requests."""
    return E2E_BASE_URL


@pytest.fixture(scope="function")
def api_headers() -> dict[str, str]:
    """Default headers for API requests including auth token."""
    # In a real implementation, this would generate a valid JWT token
    # For E2E tests, you might use a test-specific auth mechanism
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer test_e2e_token",
        "X-Correlation-ID": str(uuid4()),
    }


@pytest_asyncio.fixture(scope="function")
async def api_client(
    api_base_url: str,
    api_headers: dict[str, str],
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async HTTP client for API requests.

    Configured with:
    - Base URL from environment or default
    - Default headers including auth
    - Reasonable timeouts for E2E tests
    """
    async with httpx.AsyncClient(
        base_url=api_base_url,
        headers=api_headers,
        timeout=httpx.Timeout(30.0, connect=10.0),
        follow_redirects=True,
    ) as client:
        yield client


@pytest.fixture(scope="function")
def sync_api_client(
    api_base_url: str,
    api_headers: dict[str, str],
) -> Generator[httpx.Client, None, None]:
    """Synchronous HTTP client for API requests.

    Useful for tests that don't require async operations.
    """
    with httpx.Client(
        base_url=api_base_url,
        headers=api_headers,
        timeout=httpx.Timeout(30.0, connect=10.0),
        follow_redirects=True,
    ) as client:
        yield client


# ============================================================================
# Test Data Factories
# ============================================================================


@pytest.fixture
def news_article_factory() -> callable:
    """Factory for creating test news article data."""

    def _create(
        ticker: str = "AAPL",
        sentiment: str = "BULLISH",
        hours_ago: int = 1,
        **overrides: Any,
    ) -> dict[str, Any]:
        data = {
            "title": f"Breaking: {ticker} Shows Strong Performance",
            "content": (
                f"The stock {ticker} has shown remarkable performance in recent trading. "
                "Analysts are bullish on the company's prospects for the coming quarter."
            ),
            "summary": f"Positive outlook for {ticker}",
            "source": "Financial Times",
            "url": f"https://example.com/news/{ticker.lower()}-{uuid4().hex[:8]}",
            "published_at": (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat(),
            "tickers": [ticker],
            "sentiment": {
                "score": 0.75
                if sentiment == "BULLISH"
                else -0.5
                if sentiment == "BEARISH"
                else 0.0,
                "label": sentiment,
                "confidence": 0.85,
            },
        }
        data.update(overrides)
        return data

    return _create


@pytest.fixture
def signal_create_request_factory() -> callable:
    """Factory for creating signal creation request data."""

    def _create(
        ticker: str = "AAPL",
        direction: str = "BUY",
        confidence: float = 0.8,
        **overrides: Any,
    ) -> dict[str, Any]:
        data = {
            "ticker": ticker,
            "direction": direction,
            "confidence": confidence,
            "source": "NEWS",
            "predicted_return": "2.5",
            "predicted_drawdown": "-1.0",
            "risk_reward_ratio": "2.5",
            "expires_at": (datetime.now(UTC) + timedelta(hours=24)).isoformat(),
            "metadata": {"test": True, "e2e": True},
        }
        data.update(overrides)
        return data

    return _create


@pytest.fixture
def trade_create_request_factory() -> callable:
    """Factory for creating trade creation request data."""

    def _create(
        signal_id: str | None = None,
        quantity: str = "100",
        limit_price: str | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        data = {
            "signal_id": signal_id or str(uuid4()),
            "quantity": quantity,
        }
        if limit_price:
            data["limit_price"] = limit_price
        data.update(overrides)
        return data

    return _create


# ============================================================================
# Backtest Fixtures
# ============================================================================


@pytest.fixture
def backtest_config() -> dict[str, Any]:
    """Default backtest configuration for E2E tests."""
    return {
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "initial_capital": "100000.00",
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "strategy": {
            "type": "news_sentiment",
            "parameters": {
                "min_confidence": 0.7,
                "position_size_pct": 0.1,
                "max_positions": 5,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.10,
            },
        },
        "benchmark": "SPY",
    }


@pytest.fixture
def expected_backtest_metrics() -> dict[str, Any]:
    """Expected structure of backtest results metrics."""
    return {
        "total_return": float,
        "total_return_pct": float,
        "sharpe_ratio": (float, type(None)),
        "sortino_ratio": (float, type(None)),
        "max_drawdown": (float, type(None)),
        "max_drawdown_pct": (float, type(None)),
        "win_rate": (float, type(None)),
        "profit_factor": (float, type(None)),
        "total_trades": int,
        "winning_trades": int,
        "losing_trades": int,
        "avg_trade_duration": (float, type(None)),
        "benchmark_return": (float, type(None)),
        "alpha": (float, type(None)),
        "beta": (float, type(None)),
    }


# ============================================================================
# Skip Markers
# ============================================================================


requires_docker = pytest.mark.skipif(
    E2E_SKIP_DOCKER,
    reason="Docker services not available (E2E_SKIP_DOCKER=true)",
)

requires_api = pytest.mark.skipif(
    os.environ.get("E2E_SKIP_API", "false").lower() == "true",
    reason="API not available (E2E_SKIP_API=true)",
)

requires_external_services = pytest.mark.skipif(
    os.environ.get("E2E_SKIP_EXTERNAL", "true").lower() == "true",
    reason="External services not available (E2E_SKIP_EXTERNAL=true)",
)

slow_test = pytest.mark.skipif(
    os.environ.get("E2E_SKIP_SLOW", "false").lower() == "true",
    reason="Slow tests skipped (E2E_SKIP_SLOW=true)",
)
