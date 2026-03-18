"""FastAPI dependencies for dependency injection."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated, Any

from fastapi import Depends, Request
from pydantic import BaseModel

from gpt_quant_v2.shared.config.settings import Settings, get_settings


class MockUser(BaseModel):
    """Mock user model for auth placeholder."""

    id: str = "mock-user-001"
    username: str = "mock_user"
    email: str = "mock@example.com"
    is_active: bool = True
    roles: list[str] = ["user"]


def get_settings_dependency() -> Settings:
    """Get application settings.

    Returns:
        Settings: Application configuration settings.
    """
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_settings_dependency)]


async def get_db_session(request: Request) -> AsyncGenerator[Any, None]:
    """Get async database session from app state.

    This dependency provides a database session from the application's
    connection pool. Sessions are automatically managed by the app lifecycle.

    Args:
        request: FastAPI request object.

    Yields:
        Database session that is automatically closed after the request.
    """
    session_factory = getattr(request.app.state, "db_session_factory", None)
    if session_factory is None:
        yield None
        return

    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


DbSessionDep = Annotated[Any, Depends(get_db_session)]


async def get_redis(request: Request) -> Any:
    """Get Redis connection from app state.

    This dependency provides a Redis connection from the application's
    connection pool.

    Args:
        request: FastAPI request object.

    Returns:
        Redis connection or None if not configured.
    """
    return getattr(request.app.state, "redis", None)


RedisDep = Annotated[Any, Depends(get_redis)]


async def get_current_user() -> MockUser:
    """Get current authenticated user (placeholder).

    This is a placeholder dependency that returns a mock user.
    Replace with actual authentication logic (JWT, OAuth, etc.).

    Returns:
        MockUser: Mock user for development/testing.

    TODO:
        - Implement JWT token validation
        - Implement OAuth2 flow
        - Add proper user lookup from database
    """
    return MockUser()


CurrentUserDep = Annotated[MockUser, Depends(get_current_user)]


def get_correlation_id(request: Request) -> str | None:
    """Get correlation ID from request state.

    Args:
        request: FastAPI request object.

    Returns:
        Correlation ID if set by middleware, None otherwise.
    """
    return getattr(request.state, "correlation_id", None)


CorrelationIdDep = Annotated[str | None, Depends(get_correlation_id)]
