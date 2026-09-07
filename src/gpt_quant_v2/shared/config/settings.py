"""
Configuration Management for GPT Quant v2.

This module provides a comprehensive, type-safe configuration system using
Pydantic v2 BaseSettings. Configuration is loaded from environment variables
with the GPTQ_ prefix, with support for .env files.

Features:
    - Nested configuration models for logical grouping
    - SecretStr for sensitive values (API keys, passwords)
    - Validation for URLs, time formats, and business rules
    - Environment-based overrides (dev, test, prod)
    - Cached singleton access via get_settings()

Usage:
    >>> from gpt_quant_v2.shared.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.database.url)
    >>> print(settings.polygon.api_key.get_secret_value())

Environment Variables:
    All settings can be overridden via environment variables with GPTQ_ prefix.
    See .env.example for a complete list of available settings.
"""

from __future__ import annotations

import re
from datetime import time
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    Field,
    PostgresDsn,
    RedisDsn,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Application environment types."""

    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class LogFormat(StrEnum):
    """Supported logging formats."""

    JSON = "json"
    TEXT = "text"
    CONSOLE = "console"


class BrokerType(StrEnum):
    """Supported broker types."""

    PAPER = "paper"
    ALPACA = "alpaca"
    IBKR = "ibkr"


# =============================================================================
# Nested Settings Models
# =============================================================================


class DatabaseSettings(BaseSettings):
    """
    Database configuration settings.

    Attributes:
        url: PostgreSQL connection URL with asyncpg driver.
        echo: Enable SQLAlchemy query logging.
        pool_size: Connection pool size.
        pool_overflow: Maximum overflow connections beyond pool_size.
        pool_timeout: Seconds to wait for a connection from the pool.
    """

    model_config = SettingsConfigDict(
        env_prefix="GPTQ_DATABASE_",
        extra="ignore",
    )

    url: PostgresDsn = Field(
        default=PostgresDsn("postgresql+asyncpg://gptquant:gptquant@localhost:5432/gptquant"),
        description="PostgreSQL connection URL with asyncpg driver",
    )
    echo: bool = Field(
        default=False,
        description="Enable SQLAlchemy query logging",
    )
    pool_size: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Connection pool size",
    )
    pool_overflow: int = Field(
        default=10,
        ge=0,
        le=50,
        description="Maximum overflow connections",
    )
    pool_timeout: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Pool connection timeout in seconds",
    )

    @field_validator("url", mode="before")
    @classmethod
    def validate_postgres_url(cls, v: Any) -> Any:
        """Ensure the URL uses asyncpg driver."""
        if isinstance(v, str) and v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v


class RedisSettings(BaseSettings):
    """
    Redis configuration settings.

    Attributes:
        url: Redis connection URL.
        max_connections: Maximum connections in the pool.
        socket_timeout: Socket timeout in seconds.
        socket_connect_timeout: Connection timeout in seconds.
    """

    model_config = SettingsConfigDict(
        env_prefix="GPTQ_REDIS_",
        extra="ignore",
    )

    url: RedisDsn = Field(
        default=RedisDsn("redis://localhost:6379/0"),
        description="Redis connection URL",
    )
    max_connections: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum pool connections",
    )
    socket_timeout: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="Socket timeout in seconds",
    )
    socket_connect_timeout: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="Connection timeout in seconds",
    )


class PolygonSettings(BaseSettings):
    """
    Polygon.io API configuration settings.

    Attributes:
        api_key: Polygon.io API key (required for market data).
        rate_limit: Maximum requests per minute.
        base_url: API base URL.
        timeout: Request timeout in seconds.
    """

    model_config = SettingsConfigDict(
        env_prefix="GPTQ_POLYGON_",
        extra="ignore",
    )

    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Polygon.io API key",
    )
    rate_limit: int = Field(
        default=5,
        ge=1,
        le=1000,
        description="Maximum requests per minute",
    )
    base_url: str = Field(
        default="https://api.polygon.io",
        description="Polygon API base URL",
    )
    timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Request timeout in seconds",
    )

    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key.get_secret_value())


class OpenAISettings(BaseSettings):
    """
    OpenAI API configuration settings.

    Attributes:
        api_key: OpenAI API key (required for sentiment analysis).
        model: Model identifier to use.
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens in response.
        timeout: Request timeout in seconds.
    """

    model_config = SettingsConfigDict(
        env_prefix="GPTQ_OPENAI_",
        extra="ignore",
    )

    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API key",
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="Model identifier",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=128000,
        description="Maximum tokens in response",
    )
    timeout: float = Field(
        default=60.0,
        ge=1.0,
        le=600.0,
        description="Request timeout in seconds",
    )

    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key.get_secret_value())


class TiingoSettings(BaseSettings):
    """
    Tiingo API configuration settings.

    Attributes:
        api_key: Tiingo API key (required for news data).
        base_url: API base URL.
        timeout: Request timeout in seconds.
        rate_limit: Maximum requests per minute.
    """

    model_config = SettingsConfigDict(
        env_prefix="GPTQ_TIINGO_",
        extra="ignore",
    )

    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Tiingo API key",
    )
    base_url: str = Field(
        default="https://api.tiingo.com",
        description="Tiingo API base URL",
    )
    timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Request timeout in seconds",
    )
    rate_limit: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum requests per minute",
    )

    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key.get_secret_value())


class QuantAirflowDBSettings(BaseSettings):
    """
    Connection settings for the ``quant-airflow`` TimescaleDB (read-only).

    This points at the same Postgres/TimescaleDB instance the user's
    ``quant-airflow`` pipeline writes to. gpt-quant-v2 only ever reads from
    it (batch/EOD Korean market data ingested via kiwoom-client/DART/etc.) —
    it must never write into quant-airflow's schema. See
    ``quant-airflow/README.md`` ("데이터 읽는 법") for the reference
    connection example this mirrors.

    Attributes:
        host: TimescaleDB host (the "spare PC" exposing it on the LAN).
        port: TimescaleDB port.
        dbname: Database name.
        user: Read/query user.
        password: Password for `user`.
        connect_timeout: Seconds to wait for the TCP connection before
            giving up (kept short — this is a LAN host, not a WAN API).
    """

    model_config = SettingsConfigDict(
        env_prefix="GPTQ_QUANT_AIRFLOW_DB_",
        extra="ignore",
    )

    host: str = Field(
        default="localhost",
        description="TimescaleDB host exposing quant-airflow's data on the LAN",
    )
    port: int = Field(
        default=5432,
        ge=1,
        le=65535,
        description="TimescaleDB port",
    )
    dbname: str = Field(
        default="kr_quant",
        description="TimescaleDB database name",
    )
    user: str = Field(
        default="kr_quant",
        description="Read-only query user",
    )
    password: SecretStr = Field(
        default=SecretStr(""),
        description="Password for `user`",
    )
    connect_timeout: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="Connection timeout in seconds",
    )

    @property
    def is_configured(self) -> bool:
        """Check if a password is configured."""
        return bool(self.password.get_secret_value())


class TradingSettings(BaseSettings):
    """
    Trading window and risk management configuration.

    Attributes:
        window_start: Trading window start time (Eastern Time).
        window_end: Trading window end time (Eastern Time).
        max_position_size: Maximum position size in dollars.
        min_risk_reward: Minimum risk/reward ratio for trades.
        max_daily_loss: Maximum daily loss limit in dollars.
        position_risk_percent: Percentage of portfolio risk per position.
        min_signal_confidence: Minimum confidence for trade signals.
        min_sentiment_score: Minimum sentiment score for trades.
    """

    model_config = SettingsConfigDict(
        env_prefix="GPTQ_TRADING_",
        extra="ignore",
    )

    window_start: time = Field(
        default=time(9, 30),
        description="Trading window start (Eastern Time)",
    )
    window_end: time = Field(
        default=time(16, 0),
        description="Trading window end (Eastern Time)",
    )
    max_position_size: float = Field(
        default=10000.0,
        ge=0.0,
        description="Maximum position size in dollars",
    )
    min_risk_reward: float = Field(
        default=1.5,
        ge=0.0,
        description="Minimum risk/reward ratio",
    )
    max_daily_loss: float = Field(
        default=1000.0,
        ge=0.0,
        description="Maximum daily loss in dollars",
    )
    position_risk_percent: float = Field(
        default=1.0,
        ge=0.0,
        le=100.0,
        description="Position risk as percentage of portfolio",
    )
    min_signal_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum signal confidence threshold",
    )
    min_sentiment_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum sentiment score threshold",
    )

    @field_validator("window_start", "window_end", mode="before")
    @classmethod
    def parse_time(cls, v: Any) -> time:
        """Parse time from string format HH:MM."""
        if isinstance(v, time):
            return v
        if isinstance(v, str):
            match = re.match(r"^(\d{1,2}):(\d{2})$", v.strip())
            if match:
                hour, minute = int(match.group(1)), int(match.group(2))
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    return time(hour, minute)
            raise ValueError(f"Invalid time format: {v}. Expected HH:MM")
        raise ValueError(f"Cannot parse time from {type(v)}")

    @model_validator(mode="after")
    def validate_window(self) -> TradingSettings:
        """Ensure trading window end is after start."""
        if self.window_end <= self.window_start:
            raise ValueError("Trading window end must be after window start")
        return self


class BrokerSettings(BaseSettings):
    """
    Broker configuration settings.

    Attributes:
        type: Broker type (paper, alpaca, ibkr).
        alpaca_api_key: Alpaca API key (required for alpaca broker).
        alpaca_secret_key: Alpaca secret key (required for alpaca broker).
        alpaca_base_url: Alpaca API base URL.
    """

    model_config = SettingsConfigDict(
        env_prefix="GPTQ_",
        extra="ignore",
    )

    type: BrokerType = Field(
        default=BrokerType.PAPER,
        alias="broker_type",
        description="Broker type",
    )
    alpaca_api_key: SecretStr | None = Field(
        default=None,
        description="Alpaca API key",
    )
    alpaca_secret_key: SecretStr | None = Field(
        default=None,
        description="Alpaca secret key",
    )
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca API base URL",
    )

    @field_validator("alpaca_base_url")
    @classmethod
    def validate_alpaca_url(cls, v: str) -> str:
        """Validate Alpaca base URL format."""
        if not v.startswith(("https://", "http://")):
            raise ValueError("Alpaca base URL must start with http:// or https://")
        return v.rstrip("/")

    @model_validator(mode="after")
    def validate_alpaca_credentials(self) -> BrokerSettings:
        """Ensure Alpaca credentials are provided when using Alpaca broker."""
        if self.type == BrokerType.ALPACA:
            if not self.alpaca_api_key or not self.alpaca_api_key.get_secret_value():
                raise ValueError("Alpaca API key is required when using Alpaca broker")
            if not self.alpaca_secret_key or not self.alpaca_secret_key.get_secret_value():
                raise ValueError("Alpaca secret key is required when using Alpaca broker")
        return self

    @property
    def is_paper_trading(self) -> bool:
        """Check if paper trading is enabled."""
        return self.type == BrokerType.PAPER or "paper" in self.alpaca_base_url.lower()


class ServerSettings(BaseSettings):
    """
    Server configuration settings.

    Attributes:
        host: Server bind host.
        port: Server bind port.
        workers: Number of worker processes.
        debug: Enable debug mode.
        reload: Enable auto-reload (development only).
    """

    model_config = SettingsConfigDict(
        env_prefix="GPTQ_",
        extra="ignore",
    )

    host: str = Field(
        default="0.0.0.0",
        description="Server bind host",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server bind port",
    )
    workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of worker processes",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )
    reload: bool = Field(
        default=False,
        description="Enable auto-reload (development)",
    )


class LoggingSettings(BaseSettings):
    """
    Logging configuration settings.

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format: Log format (json, text, console).
        file: Log file path (optional).
        rotation: Log rotation size (e.g., "10 MB").
        retention: Log retention period (e.g., "7 days").
    """

    model_config = SettingsConfigDict(
        env_prefix="GPTQ_LOG_",
        extra="ignore",
    )

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Log level",
    )
    format: LogFormat = Field(
        default=LogFormat.JSON,
        description="Log format",
    )
    file: Path | None = Field(
        default=Path("logs/gptquant.log"),
        description="Log file path",
    )
    rotation: str = Field(
        default="10 MB",
        description="Log rotation size",
    )
    retention: str = Field(
        default="7 days",
        description="Log retention period",
    )

    @field_validator("file", mode="before")
    @classmethod
    def parse_path(cls, v: Any) -> Path | None:
        """Parse path from string."""
        if v is None or v == "":
            return None
        if isinstance(v, Path):
            return v
        return Path(v)


class MLSettings(BaseSettings):
    """
    Machine Learning configuration settings.

    Attributes:
        mlflow_tracking_uri: MLflow tracking server URI.
        model_registry_path: Path to model registry.
        experiment_name: Default MLflow experiment name.
    """

    model_config = SettingsConfigDict(
        env_prefix="GPTQ_",
        extra="ignore",
    )

    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlruns.db",
        description="MLflow tracking URI",
    )
    model_registry_path: Path = Field(
        default=Path("./models"),
        description="Model registry path",
    )
    experiment_name: str = Field(
        default="gptquant",
        description="MLflow experiment name",
    )


class SecuritySettings(BaseSettings):
    """
    Security configuration settings.

    Attributes:
        secret_key: Application secret key for signing.
        cors_origins: Allowed CORS origins.
        allowed_hosts: Allowed host headers.
    """

    model_config = SettingsConfigDict(
        env_prefix="GPTQ_",
        extra="ignore",
    )

    secret_key: SecretStr = Field(
        default=SecretStr("change-me-in-production"),
        description="Application secret key",
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins",
    )
    allowed_hosts: list[str] = Field(
        default=["*"],
        description="Allowed host headers",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            # Handle JSON array string
            if v.startswith("["):
                import json

                parsed: list[str] = json.loads(v)
                return parsed
            # Handle comma-separated string
            return [origin.strip() for origin in v.split(",")]
        return list(v)


# =============================================================================
# Main Settings Class
# =============================================================================


class Settings(BaseSettings):
    """
    Main application settings.

    This class aggregates all nested configuration models and provides
    a single entry point for accessing all application settings.

    Attributes:
        env: Application environment (dev, test, prod).
        app_name: Application name.
        database: Database configuration.
        redis: Redis configuration.
        polygon: Polygon.io API configuration.
        openai: OpenAI API configuration.
        tiingo: Tiingo API configuration.
        trading: Trading configuration.
        broker: Broker configuration.
        server: Server configuration.
        logging: Logging configuration.
        ml: Machine learning configuration.
        security: Security configuration.

    Example:
        >>> settings = Settings()
        >>> settings.database.url
        PostgresDsn('postgresql+asyncpg://...')
        >>> settings.is_production
        False
    """

    model_config = SettingsConfigDict(
        env_prefix="GPTQ_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # Environment
    env: Environment = Field(
        default=Environment.DEV,
        description="Application environment",
    )
    app_name: str = Field(
        default="GPT Quant v2",
        description="Application name",
    )

    # Nested settings - initialized with defaults
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    polygon: PolygonSettings = Field(default_factory=PolygonSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    tiingo: TiingoSettings = Field(default_factory=TiingoSettings)
    quant_airflow_db: QuantAirflowDBSettings = Field(default_factory=QuantAirflowDBSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    broker: BrokerSettings = Field(default_factory=BrokerSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.env == Environment.PROD

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.env == Environment.DEV

    @property
    def is_testing(self) -> bool:
        """Check if running in test environment."""
        return self.env == Environment.TEST

    def validate_api_keys(self) -> dict[str, bool]:
        """
        Validate that all required API keys are configured.

        Returns:
            Dictionary mapping service name to configuration status.
        """
        return {
            "polygon": self.polygon.is_configured,
            "openai": self.openai.is_configured,
            "tiingo": self.tiingo.is_configured,
        }

    def get_missing_api_keys(self) -> list[str]:
        """
        Get list of unconfigured API keys.

        Returns:
            List of service names with missing API keys.
        """
        return [
            service for service, configured in self.validate_api_keys().items() if not configured
        ]


# =============================================================================
# Settings Factory
# =============================================================================


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached application settings instance.

    This function returns a singleton Settings instance, loading configuration
    from environment variables and .env file. The result is cached for
    subsequent calls.

    Returns:
        Settings: Application settings instance.

    Example:
        >>> settings = get_settings()
        >>> print(settings.database.url)
        >>> print(settings.server.port)

    Note:
        To reload settings (e.g., after environment changes), call:
        >>> get_settings.cache_clear()
        >>> settings = get_settings()
    """
    return Settings()


def get_settings_for_env(env: Environment | str) -> Settings:
    """
    Get settings for a specific environment.

    This factory function creates a new Settings instance with the specified
    environment, useful for testing or environment-specific configuration.

    Args:
        env: Target environment (dev, test, prod) or Environment enum.

    Returns:
        Settings: Settings instance configured for the specified environment.

    Example:
        >>> prod_settings = get_settings_for_env(Environment.PROD)
        >>> test_settings = get_settings_for_env("test")
    """
    if isinstance(env, str):
        env = Environment(env)
    return Settings(env=env)


def clear_settings_cache() -> None:
    """
    Clear the cached settings instance.

    Call this function to force settings to be reloaded on the next
    call to get_settings(). Useful after environment variable changes.

    Example:
        >>> clear_settings_cache()
        >>> settings = get_settings()  # Reloads from environment
    """
    get_settings.cache_clear()
