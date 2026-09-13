"""FastAPI application factory and configuration."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from daytrade_it.interfaces.api.middleware import (
    RequestLoggingMiddleware,
    TimingMiddleware,
)
from daytrade_it.interfaces.api.schemas import ErrorResponse, HealthResponse
from daytrade_it.shared.config.settings import Settings, get_settings
from daytrade_it.shared.errors.exceptions import (
    DomainError,
    ExternalAPIError,
    GptQuantError,
    InfrastructureError,
    ValidationError,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown events.

    Initializes database and Redis connections on startup,
    and cleans them up on shutdown.
    """
    settings = get_settings()
    await logger.ainfo("Starting GPT Quant API", environment=settings.env.value)

    # Initialize database connection
    app.state.db_session_factory = None
    app.state.db_engine = None
    try:
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        engine = create_async_engine(
            str(settings.database.url),
            echo=settings.database.echo,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.pool_overflow,
            pool_timeout=settings.database.pool_timeout,
        )
        app.state.db_engine = engine
        app.state.db_session_factory = async_sessionmaker(
            engine,
            expire_on_commit=False,
        )
        await logger.ainfo("Database connection pool initialized")
    except Exception as e:
        await logger.awarning("Database connection failed", error=str(e))

    # Initialize Redis connection
    app.state.redis = None
    try:
        import redis.asyncio as redis

        app.state.redis = redis.from_url(
            str(settings.redis.url),
            max_connections=settings.redis.max_connections,
            socket_timeout=settings.redis.socket_timeout,
            socket_connect_timeout=settings.redis.socket_connect_timeout,
        )
        # Test connection
        await app.state.redis.ping()
        await logger.ainfo("Redis connection initialized")
    except Exception as e:
        await logger.awarning("Redis connection failed", error=str(e))

    yield

    # Cleanup on shutdown
    await logger.ainfo("Shutting down GPT Quant API")

    if app.state.redis:
        await app.state.redis.close()
        await logger.ainfo("Redis connection closed")

    if app.state.db_engine:
        await app.state.db_engine.dispose()
        await logger.ainfo("Database connection pool closed")


# OpenAPI Tags Metadata
TAGS_METADATA = [
    {
        "name": "signals",
        "description": "Trading signal operations. Generate, retrieve, and manage AI-driven trading signals based on news and market analysis.",
    },
    {
        "name": "trades",
        "description": "Trade execution and management. Execute trades based on signals and track trading history.",
    },
    {
        "name": "portfolio",
        "description": "Portfolio management operations. View positions, performance metrics, and portfolio analytics.",
    },
    {
        "name": "news",
        "description": "News data operations. Access market news and sentiment analysis for trading decisions.",
    },
    {
        "name": "websocket",
        "description": "WebSocket endpoints for real-time data streaming including signals, trades, and market updates.",
    },
    {
        "name": "health",
        "description": "Health check and system status endpoints.",
    },
]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        settings: Optional settings override. Uses get_settings() if not provided.

    Returns:
        Configured FastAPI application instance.
    """
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="GPT Quant V2 API",
        description="""
# GPT Quant V2 API

AI-Agent-Friendly News-Driven Algorithmic Trading System.

## Overview

GPT Quant V2 is a sophisticated algorithmic trading platform that leverages AI and machine learning
to generate trading signals from news and market data. This API provides programmatic access to
all trading functionality.

## Key Features

- **AI-Driven Signals**: Generate trading signals using ML models trained on news sentiment and market data
- **Automated Trade Execution**: Execute trades based on signals with risk management controls
- **Portfolio Management**: Track positions, P&L, and portfolio performance metrics
- **Real-time Updates**: WebSocket support for streaming signals, trades, and market data
- **News Analysis**: Access processed news articles with sentiment scores

## Authentication

All endpoints require authentication via JWT tokens. Include the token in the `Authorization` header:

```
Authorization: Bearer <your_jwt_token>
```

## Rate Limits

- Standard endpoints: 100 requests/minute
- WebSocket connections: 5 concurrent connections per user
- Data-intensive endpoints: 20 requests/minute

## Support

For API support, please open an issue on our GitHub repository.
""",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
    )

    # Store settings in app state
    app.state.settings = settings

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware (order matters - first added = outermost)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # Register exception handlers
    _register_exception_handlers(app)

    # Register routers
    _register_routers(app)

    # Health check endpoint
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["health"],
        summary="Health check",
        description="Check the health status of the API and its dependencies.",
    )
    async def health_check(request: Request) -> HealthResponse:
        """Check API health and dependency status."""
        db_status = "disconnected"
        redis_status = "disconnected"

        # Check database
        if request.app.state.db_session_factory:
            try:
                from sqlalchemy import text

                async with request.app.state.db_session_factory() as session:
                    await session.execute(text("SELECT 1"))
                db_status = "connected"
            except Exception:
                db_status = "error"

        # Check Redis
        if request.app.state.redis:
            try:
                await request.app.state.redis.ping()
                redis_status = "connected"
            except Exception:
                redis_status = "error"

        return HealthResponse(
            status="healthy",
            version="0.1.0",
            timestamp=datetime.now(UTC),
            database=db_status,
            redis=redis_status,
        )

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers for GptQuantError hierarchy."""

    @app.exception_handler(ValidationError)
    async def validation_error_handler(
        request: Request,
        exc: ValidationError,
    ) -> JSONResponse:
        """Handle validation errors (400)."""
        correlation_id = getattr(request.state, "correlation_id", None)
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error_code=exc.error_code,
                message=exc.message,
                details=exc.details,
                timestamp=exc.timestamp,
                correlation_id=correlation_id or exc.correlation_id,
            ).model_dump(mode="json"),
        )

    @app.exception_handler(DomainError)
    async def domain_error_handler(
        request: Request,
        exc: DomainError,
    ) -> JSONResponse:
        """Handle domain/business logic errors (422)."""
        correlation_id = getattr(request.state, "correlation_id", None)
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error_code=exc.error_code,
                message=exc.message,
                details=exc.details,
                timestamp=exc.timestamp,
                correlation_id=correlation_id or exc.correlation_id,
            ).model_dump(mode="json"),
        )

    @app.exception_handler(InfrastructureError)
    async def infrastructure_error_handler(
        request: Request,
        exc: InfrastructureError,
    ) -> JSONResponse:
        """Handle infrastructure errors (503)."""
        correlation_id = getattr(request.state, "correlation_id", None)
        await logger.aerror(
            "Infrastructure error",
            error_code=exc.error_code,
            message=exc.message,
            correlation_id=correlation_id,
        )
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error_code=exc.error_code,
                message="Service temporarily unavailable",
                details=None,  # Hide internal details
                timestamp=exc.timestamp,
                correlation_id=correlation_id or exc.correlation_id,
            ).model_dump(mode="json"),
        )

    @app.exception_handler(ExternalAPIError)
    async def external_api_error_handler(
        request: Request,
        exc: ExternalAPIError,
    ) -> JSONResponse:
        """Handle external API errors (502)."""
        correlation_id = getattr(request.state, "correlation_id", None)
        await logger.aerror(
            "External API error",
            error_code=exc.error_code,
            message=exc.message,
            correlation_id=correlation_id,
        )
        return JSONResponse(
            status_code=502,
            content=ErrorResponse(
                error_code=exc.error_code,
                message="External service error",
                details=None,  # Hide external details
                timestamp=exc.timestamp,
                correlation_id=correlation_id or exc.correlation_id,
            ).model_dump(mode="json"),
        )

    @app.exception_handler(GptQuantError)
    async def gpt_quant_error_handler(
        request: Request,
        exc: GptQuantError,
    ) -> JSONResponse:
        """Handle base GptQuantError and unhandled subclasses (500)."""
        correlation_id = getattr(request.state, "correlation_id", None)
        await logger.aerror(
            "Unhandled application error",
            error_code=exc.error_code,
            message=exc.message,
            correlation_id=correlation_id,
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_code=exc.error_code,
                message="Internal server error",
                details=None,
                timestamp=exc.timestamp,
                correlation_id=correlation_id or exc.correlation_id,
            ).model_dump(mode="json"),
        )


def _register_routers(app: FastAPI) -> None:
    """Register API routers.

    Add router registrations here as they are created.
    """
    from daytrade_it.interfaces.api.routers import news, portfolio, signals, trades

    app.include_router(signals.router, prefix="/api/v1/signals", tags=["signals"])
    app.include_router(trades.router, prefix="/api/v1/trades", tags=["trades"])
    app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["portfolio"])
    app.include_router(news.router, prefix="/api/v1/news", tags=["news"])


# Create default app instance for uvicorn
app = create_app()
