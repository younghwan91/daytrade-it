"""FastAPI REST API.

This module provides the FastAPI application and related components.

Usage:
    # Run with uvicorn
    uvicorn daytrade_it.interfaces.api.main:app --reload

    # Or create app programmatically
    from daytrade_it.interfaces.api import create_app
    app = create_app()
"""

from daytrade_it.interfaces.api.dependencies import (
    CorrelationIdDep,
    CurrentUserDep,
    DbSessionDep,
    MockUser,
    RedisDep,
    SettingsDep,
    get_current_user,
    get_db_session,
    get_redis,
    get_settings_dependency,
)
from daytrade_it.interfaces.api.main import app, create_app
from daytrade_it.interfaces.api.middleware import (
    RequestLoggingMiddleware,
    TimingMiddleware,
)
from daytrade_it.interfaces.api.schemas import (
    ErrorResponse,
    HealthResponse,
    PaginatedResponse,
    PaginationParams,
)

__all__ = [
    # App
    "app",
    "create_app",
    # Dependencies
    "CorrelationIdDep",
    "CurrentUserDep",
    "DbSessionDep",
    "MockUser",
    "RedisDep",
    "SettingsDep",
    "get_current_user",
    "get_db_session",
    "get_redis",
    "get_settings_dependency",
    # Middleware
    "RequestLoggingMiddleware",
    "TimingMiddleware",
    # Schemas
    "ErrorResponse",
    "HealthResponse",
    "PaginatedResponse",
    "PaginationParams",
]
