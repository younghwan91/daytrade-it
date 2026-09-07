"""Exception hierarchy for GPT Quant V2."""

from datetime import UTC, datetime
from typing import Any

from gpt_quant_v2.shared.errors import error_codes


class GptQuantError(Exception):
    """Base exception class for all GPT Quant V2 errors.

    Attributes:
        error_code: Unique error identifier (e.g., "GPTQ-001")
        message: Human-readable error message
        details: Optional dictionary with additional context
        timestamp: When the error occurred (UTC)
        correlation_id: Optional ID for tracing across services
    """

    error_code: str = "GPTQ-000"

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if error_code is not None:
            self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now(UTC)
        self.correlation_id = correlation_id

    def to_dict(self) -> dict[str, Any]:
        """Serialize exception to dictionary for logging/API responses."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "type": self.__class__.__name__,
        }

    def __str__(self) -> str:
        """Format error for display."""
        parts = [f"[{self.error_code}] {self.message}"]
        if self.correlation_id:
            parts.append(f"(correlation_id: {self.correlation_id})")
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"| Details: {details_str}")
        return " ".join(parts)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"error_code={self.error_code!r}, "
            f"message={self.message!r}, "
            f"details={self.details!r}, "
            f"correlation_id={self.correlation_id!r})"
        )


# =============================================================================
# Configuration Errors (GPTQ-1xx)
# =============================================================================


class ConfigurationError(GptQuantError):
    """Raised when there is a configuration problem."""

    error_code = error_codes.CONFIG_ERROR

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code or self.error_code,
            details=details,
            correlation_id=correlation_id,
        )


# =============================================================================
# Validation Errors (GPTQ-2xx)
# =============================================================================


class ValidationError(GptQuantError):
    """Raised when input validation fails."""

    error_code = error_codes.VALIDATION_ERROR

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        field: str | None = None,
        value: Any = None,
    ) -> None:
        if field:
            details = details or {}
            details["field"] = field
        if value is not None:
            details = details or {}
            details["value"] = value
        super().__init__(
            message=message,
            error_code=error_code or self.error_code,
            details=details,
            correlation_id=correlation_id,
        )


# =============================================================================
# Domain Errors (GPTQ-3xx) - Business Logic Errors
# =============================================================================


class DomainError(GptQuantError):
    """Base class for business logic errors."""

    error_code = error_codes.DOMAIN_ERROR

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code or self.error_code,
            details=details,
            correlation_id=correlation_id,
        )


class InsufficientFundsError(DomainError):
    """Raised when there are insufficient funds for an operation."""

    error_code = error_codes.INSUFFICIENT_FUNDS

    def __init__(
        self,
        message: str = "Insufficient funds for this operation",
        required_amount: float | None = None,
        available_amount: float | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        details = details or {}
        if required_amount is not None:
            details["required_amount"] = required_amount
        if available_amount is not None:
            details["available_amount"] = available_amount
        super().__init__(
            message=message,
            error_code=self.error_code,
            details=details,
            correlation_id=correlation_id,
        )


class InvalidSignalError(DomainError):
    """Raised when a trading signal is invalid or malformed."""

    error_code = error_codes.INVALID_SIGNAL

    def __init__(
        self,
        message: str = "Invalid trading signal",
        signal_type: str | None = None,
        symbol: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        details = details or {}
        if signal_type:
            details["signal_type"] = signal_type
        if symbol:
            details["symbol"] = symbol
        super().__init__(
            message=message,
            error_code=self.error_code,
            details=details,
            correlation_id=correlation_id,
        )


class RiskLimitExceededError(DomainError):
    """Raised when a risk limit has been exceeded."""

    error_code = error_codes.RISK_LIMIT_EXCEEDED

    def __init__(
        self,
        message: str = "Risk limit exceeded",
        limit_type: str | None = None,
        current_value: float | None = None,
        limit_value: float | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        details = details or {}
        if limit_type:
            details["limit_type"] = limit_type
        if current_value is not None:
            details["current_value"] = current_value
        if limit_value is not None:
            details["limit_value"] = limit_value
        super().__init__(
            message=message,
            error_code=self.error_code,
            details=details,
            correlation_id=correlation_id,
        )


# =============================================================================
# Infrastructure Errors (GPTQ-4xx)
# =============================================================================


class InfrastructureError(GptQuantError):
    """Base class for infrastructure-related errors."""

    error_code = error_codes.INFRASTRUCTURE_ERROR

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        if cause:
            details = details or {}
            details["cause"] = str(cause)
            details["cause_type"] = type(cause).__name__
        super().__init__(
            message=message,
            error_code=error_code or self.error_code,
            details=details,
            correlation_id=correlation_id,
        )
        self.__cause__ = cause


class DatabaseError(InfrastructureError):
    """Raised when a database operation fails."""

    error_code = error_codes.DATABASE_ERROR

    def __init__(
        self,
        message: str = "Database operation failed",
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        cause: Exception | None = None,
        query: str | None = None,
    ) -> None:
        details = details or {}
        if query:
            details["query"] = query[:200]  # Truncate for safety
        super().__init__(
            message=message,
            error_code=error_code or self.error_code,
            details=details,
            correlation_id=correlation_id,
            cause=cause,
        )


class CacheError(InfrastructureError):
    """Raised when a cache operation fails."""

    error_code = error_codes.CACHE_ERROR

    def __init__(
        self,
        message: str = "Cache operation failed",
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        cause: Exception | None = None,
        key: str | None = None,
    ) -> None:
        details = details or {}
        if key:
            details["key"] = key
        super().__init__(
            message=message,
            error_code=error_code or self.error_code,
            details=details,
            correlation_id=correlation_id,
            cause=cause,
        )


# =============================================================================
# External API Errors (GPTQ-5xx)
# =============================================================================


class ExternalAPIError(GptQuantError):
    """Base class for external API errors."""

    error_code = error_codes.EXTERNAL_API_ERROR

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        details = details or {}
        if status_code is not None:
            details["status_code"] = status_code
        if response_body:
            details["response_body"] = response_body[:500]  # Truncate
        super().__init__(
            message=message,
            error_code=error_code or self.error_code,
            details=details,
            correlation_id=correlation_id,
        )


class PolygonAPIError(ExternalAPIError):
    """Raised when Polygon API request fails."""

    error_code = error_codes.POLYGON_API_ERROR

    def __init__(
        self,
        message: str = "Polygon API request failed",
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        status_code: int | None = None,
        response_body: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        details = details or {}
        if endpoint:
            details["endpoint"] = endpoint
        super().__init__(
            message=message,
            error_code=error_code or self.error_code,
            details=details,
            correlation_id=correlation_id,
            status_code=status_code,
            response_body=response_body,
        )


class OpenAIAPIError(ExternalAPIError):
    """Raised when OpenAI API request fails."""

    error_code = error_codes.OPENAI_API_ERROR

    def __init__(
        self,
        message: str = "OpenAI API request failed",
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        status_code: int | None = None,
        response_body: str | None = None,
        model: str | None = None,
    ) -> None:
        details = details or {}
        if model:
            details["model"] = model
        super().__init__(
            message=message,
            error_code=error_code or self.error_code,
            details=details,
            correlation_id=correlation_id,
            status_code=status_code,
            response_body=response_body,
        )


class TiingoAPIError(ExternalAPIError):
    """Raised when Tiingo API request fails."""

    error_code = error_codes.TIINGO_API_ERROR

    def __init__(
        self,
        message: str = "Tiingo API request failed",
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        status_code: int | None = None,
        response_body: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        details = details or {}
        if endpoint:
            details["endpoint"] = endpoint
        super().__init__(
            message=message,
            error_code=error_code or self.error_code,
            details=details,
            correlation_id=correlation_id,
            status_code=status_code,
            response_body=response_body,
        )


class TossNewsAPIError(ExternalAPIError):
    """Raised when the Toss dashboard news API request fails."""

    error_code = error_codes.TOSS_NEWS_API_ERROR

    def __init__(
        self,
        message: str = "Toss news dashboard request failed",
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code or self.error_code,
            details=details,
            correlation_id=correlation_id,
            status_code=status_code,
            response_body=response_body,
        )


class DartAPIError(ExternalAPIError):
    """Raised when a DART OpenAPI request fails (other than quota exhaustion)."""

    error_code = error_codes.DART_API_ERROR

    def __init__(
        self,
        message: str = "DART OpenAPI request failed",
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code or self.error_code,
            details=details,
            correlation_id=correlation_id,
            status_code=status_code,
            response_body=response_body,
        )


# =============================================================================
# Broker Errors (GPTQ-6xx)
# =============================================================================


class BrokerError(GptQuantError):
    """Base class for broker-related errors."""

    error_code = error_codes.BROKER_ERROR

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        broker_name: str | None = None,
    ) -> None:
        details = details or {}
        if broker_name:
            details["broker_name"] = broker_name
        super().__init__(
            message=message,
            error_code=error_code or self.error_code,
            details=details,
            correlation_id=correlation_id,
        )


class OrderRejectedError(BrokerError):
    """Raised when an order is rejected by the broker."""

    error_code = error_codes.ORDER_REJECTED

    def __init__(
        self,
        message: str = "Order rejected by broker",
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        broker_name: str | None = None,
        order_id: str | None = None,
        rejection_reason: str | None = None,
        symbol: str | None = None,
    ) -> None:
        details = details or {}
        if order_id:
            details["order_id"] = order_id
        if rejection_reason:
            details["rejection_reason"] = rejection_reason
        if symbol:
            details["symbol"] = symbol
        super().__init__(
            message=message,
            error_code=self.error_code,
            details=details,
            correlation_id=correlation_id,
            broker_name=broker_name,
        )


class InsufficientMarginError(BrokerError):
    """Raised when there is insufficient margin for an order."""

    error_code = error_codes.INSUFFICIENT_MARGIN

    def __init__(
        self,
        message: str = "Insufficient margin for order",
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        broker_name: str | None = None,
        required_margin: float | None = None,
        available_margin: float | None = None,
    ) -> None:
        details = details or {}
        if required_margin is not None:
            details["required_margin"] = required_margin
        if available_margin is not None:
            details["available_margin"] = available_margin
        super().__init__(
            message=message,
            error_code=self.error_code,
            details=details,
            correlation_id=correlation_id,
            broker_name=broker_name,
        )
