"""FastAPI REST API.

This module provides the FastAPI application and related components.

Usage:
    # Run with uvicorn
    uvicorn krx_signal_engine.interfaces.api.main:app --reload

    # Or create app programmatically
    from krx_signal_engine.interfaces.api import create_app
    app = create_app()
"""

from krx_signal_engine.interfaces.api.dependencies import (
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
from krx_signal_engine.interfaces.api.main import app, create_app
from krx_signal_engine.interfaces.api.middleware import (
    RequestLoggingMiddleware,
    TimingMiddleware,
)
from krx_signal_engine.interfaces.api.schemas import (
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
