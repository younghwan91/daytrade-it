"""Error Handling"""

from gpt_quant_v2.shared.errors import error_codes
from gpt_quant_v2.shared.errors.exceptions import (
    BrokerError,
    CacheError,
    ConfigurationError,
    DatabaseError,
    DomainError,
    ExternalAPIError,
    GptQuantError,
    InfrastructureError,
    InsufficientFundsError,
    InsufficientMarginError,
    InvalidSignalError,
    OpenAIAPIError,
    OrderRejectedError,
    PolygonAPIError,
    RiskLimitExceededError,
    TiingoAPIError,
    ValidationError,
)

__all__ = [
    # Error codes module
    "error_codes",
    # Base exception
    "GptQuantError",
    # Configuration (GPTQ-1xx)
    "ConfigurationError",
    # Validation (GPTQ-2xx)
    "ValidationError",
    # Domain (GPTQ-3xx)
    "DomainError",
    "InsufficientFundsError",
    "InvalidSignalError",
    "RiskLimitExceededError",
    # Infrastructure (GPTQ-4xx)
    "InfrastructureError",
    "DatabaseError",
    "CacheError",
    # External API (GPTQ-5xx)
    "ExternalAPIError",
    "PolygonAPIError",
    "OpenAIAPIError",
    "TiingoAPIError",
    # Broker (GPTQ-6xx)
    "BrokerError",
    "OrderRejectedError",
    "InsufficientMarginError",
]
