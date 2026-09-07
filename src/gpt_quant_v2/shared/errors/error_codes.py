"""Error code constants for GPT Quant V2."""

# Configuration Errors (GPTQ-1xx)
CONFIG_ERROR = "GPTQ-100"
CONFIG_MISSING_REQUIRED = "GPTQ-101"
CONFIG_INVALID_VALUE = "GPTQ-102"
CONFIG_FILE_NOT_FOUND = "GPTQ-103"
CONFIG_PARSE_ERROR = "GPTQ-104"
CONFIG_ENV_VAR_MISSING = "GPTQ-105"

# Validation Errors (GPTQ-2xx)
VALIDATION_ERROR = "GPTQ-200"
VALIDATION_REQUIRED_FIELD = "GPTQ-201"
VALIDATION_INVALID_TYPE = "GPTQ-202"
VALIDATION_INVALID_FORMAT = "GPTQ-203"
VALIDATION_OUT_OF_RANGE = "GPTQ-204"
VALIDATION_INVALID_SYMBOL = "GPTQ-205"
VALIDATION_INVALID_DATE_RANGE = "GPTQ-206"

# Domain Errors (GPTQ-3xx)
DOMAIN_ERROR = "GPTQ-300"
INSUFFICIENT_FUNDS = "GPTQ-301"
INVALID_SIGNAL = "GPTQ-302"
RISK_LIMIT_EXCEEDED = "GPTQ-303"
POSITION_NOT_FOUND = "GPTQ-304"
INVALID_ORDER_STATE = "GPTQ-305"
PORTFOLIO_CONSTRAINT_VIOLATED = "GPTQ-306"
STRATEGY_EXECUTION_ERROR = "GPTQ-307"

# Infrastructure Errors (GPTQ-4xx)
INFRASTRUCTURE_ERROR = "GPTQ-400"
DATABASE_ERROR = "GPTQ-401"
DATABASE_CONNECTION_ERROR = "GPTQ-402"
DATABASE_QUERY_ERROR = "GPTQ-403"
CACHE_ERROR = "GPTQ-410"
CACHE_CONNECTION_ERROR = "GPTQ-411"
CACHE_KEY_NOT_FOUND = "GPTQ-412"
MESSAGE_QUEUE_ERROR = "GPTQ-420"

# External API Errors (GPTQ-5xx)
EXTERNAL_API_ERROR = "GPTQ-500"
POLYGON_API_ERROR = "GPTQ-501"
POLYGON_RATE_LIMIT = "GPTQ-502"
POLYGON_INVALID_RESPONSE = "GPTQ-503"
OPENAI_API_ERROR = "GPTQ-510"
OPENAI_RATE_LIMIT = "GPTQ-511"
OPENAI_INVALID_RESPONSE = "GPTQ-512"
OPENAI_CONTEXT_LENGTH_EXCEEDED = "GPTQ-513"
TIINGO_API_ERROR = "GPTQ-520"
TIINGO_RATE_LIMIT = "GPTQ-521"
TIINGO_INVALID_RESPONSE = "GPTQ-522"
TOSS_NEWS_API_ERROR = "GPTQ-530"
TOSS_NEWS_CIRCUIT_OPEN = "GPTQ-531"
DART_API_ERROR = "GPTQ-540"
DART_QUOTA_EXCEEDED = "GPTQ-541"

# Broker Errors (GPTQ-6xx)
BROKER_ERROR = "GPTQ-600"
ORDER_REJECTED = "GPTQ-601"
INSUFFICIENT_MARGIN = "GPTQ-602"
BROKER_CONNECTION_ERROR = "GPTQ-603"
BROKER_AUTHENTICATION_ERROR = "GPTQ-604"
ORDER_NOT_FOUND = "GPTQ-605"
MARKET_CLOSED = "GPTQ-606"
INVALID_ORDER_QUANTITY = "GPTQ-607"


# Error code descriptions for logging and debugging
ERROR_DESCRIPTIONS: dict[str, str] = {
    # Configuration
    CONFIG_ERROR: "General configuration error",
    CONFIG_MISSING_REQUIRED: "Required configuration value is missing",
    CONFIG_INVALID_VALUE: "Configuration value is invalid",
    CONFIG_FILE_NOT_FOUND: "Configuration file not found",
    CONFIG_PARSE_ERROR: "Failed to parse configuration file",
    CONFIG_ENV_VAR_MISSING: "Required environment variable is missing",
    # Validation
    VALIDATION_ERROR: "General validation error",
    VALIDATION_REQUIRED_FIELD: "Required field is missing",
    VALIDATION_INVALID_TYPE: "Field has invalid type",
    VALIDATION_INVALID_FORMAT: "Field has invalid format",
    VALIDATION_OUT_OF_RANGE: "Value is out of acceptable range",
    VALIDATION_INVALID_SYMBOL: "Invalid ticker symbol",
    VALIDATION_INVALID_DATE_RANGE: "Invalid date range specified",
    # Domain
    DOMAIN_ERROR: "General domain/business logic error",
    INSUFFICIENT_FUNDS: "Insufficient funds for operation",
    INVALID_SIGNAL: "Trading signal is invalid or malformed",
    RISK_LIMIT_EXCEEDED: "Risk limit has been exceeded",
    POSITION_NOT_FOUND: "Position not found in portfolio",
    INVALID_ORDER_STATE: "Order is in an invalid state for this operation",
    PORTFOLIO_CONSTRAINT_VIOLATED: "Portfolio constraint has been violated",
    STRATEGY_EXECUTION_ERROR: "Error during strategy execution",
    # Infrastructure
    INFRASTRUCTURE_ERROR: "General infrastructure error",
    DATABASE_ERROR: "General database error",
    DATABASE_CONNECTION_ERROR: "Failed to connect to database",
    DATABASE_QUERY_ERROR: "Database query failed",
    CACHE_ERROR: "General cache error",
    CACHE_CONNECTION_ERROR: "Failed to connect to cache",
    CACHE_KEY_NOT_FOUND: "Cache key not found",
    MESSAGE_QUEUE_ERROR: "Message queue error",
    # External API
    EXTERNAL_API_ERROR: "General external API error",
    POLYGON_API_ERROR: "Polygon API error",
    POLYGON_RATE_LIMIT: "Polygon API rate limit exceeded",
    POLYGON_INVALID_RESPONSE: "Invalid response from Polygon API",
    OPENAI_API_ERROR: "OpenAI API error",
    OPENAI_RATE_LIMIT: "OpenAI API rate limit exceeded",
    OPENAI_INVALID_RESPONSE: "Invalid response from OpenAI API",
    OPENAI_CONTEXT_LENGTH_EXCEEDED: "OpenAI context length exceeded",
    TIINGO_API_ERROR: "Tiingo API error",
    TIINGO_RATE_LIMIT: "Tiingo API rate limit exceeded",
    TIINGO_INVALID_RESPONSE: "Invalid response from Tiingo API",
    TOSS_NEWS_API_ERROR: "Toss news dashboard API error",
    TOSS_NEWS_CIRCUIT_OPEN: "Toss news poller circuit breaker is open (degraded to DART-only)",
    DART_API_ERROR: "DART OpenAPI error",
    DART_QUOTA_EXCEEDED: "DART OpenAPI daily quota exhausted",
    # Broker
    BROKER_ERROR: "General broker error",
    ORDER_REJECTED: "Order was rejected by broker",
    INSUFFICIENT_MARGIN: "Insufficient margin for order",
    BROKER_CONNECTION_ERROR: "Failed to connect to broker",
    BROKER_AUTHENTICATION_ERROR: "Broker authentication failed",
    ORDER_NOT_FOUND: "Order not found",
    MARKET_CLOSED: "Market is closed",
    INVALID_ORDER_QUANTITY: "Invalid order quantity",
}


def get_error_description(error_code: str) -> str:
    """Get human-readable description for an error code."""
    return ERROR_DESCRIPTIONS.get(error_code, f"Unknown error code: {error_code}")
