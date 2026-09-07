"""
Configuration Management for GPT Quant v2.

This module provides centralized configuration management using Pydantic v2
BaseSettings with environment variable support and .env file loading.

Example:
    >>> from gpt_quant_v2.shared.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.database.url)
    >>> print(settings.server.port)
"""

from gpt_quant_v2.shared.config.settings import (
    BrokerSettings,
    BrokerType,
    DatabaseSettings,
    Environment,
    KrxNewsSettings,
    LogFormat,
    LoggingSettings,
    MLSettings,
    OpenAISettings,
    QuantAirflowDBSettings,
    RedisSettings,
    SecuritySettings,
    ServerSettings,
    Settings,
    TradingSettings,
    clear_settings_cache,
    get_settings,
    get_settings_for_env,
)

__all__ = [
    # Main settings
    "Settings",
    "get_settings",
    "get_settings_for_env",
    "clear_settings_cache",
    # Nested settings models
    "DatabaseSettings",
    "RedisSettings",
    "OpenAISettings",
    "KrxNewsSettings",
    "QuantAirflowDBSettings",
    "TradingSettings",
    "BrokerSettings",
    "ServerSettings",
    "LoggingSettings",
    "MLSettings",
    "SecuritySettings",
    # Enums
    "Environment",
    "LogFormat",
    "BrokerType",
]
