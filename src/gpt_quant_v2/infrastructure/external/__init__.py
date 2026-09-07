"""External API Clients."""

from gpt_quant_v2.infrastructure.external.openai_client import (
    OpenAIClient,
    OpenAIClientConfig,
    RateLimitState,
    SentimentAnalysis,
    UsageStats,
)
from gpt_quant_v2.infrastructure.external.polygon_client import (
    Bar,
    CacheBackend,
    DayData,
    InMemoryCache,
    MarketType,
    MinuteData,
    PolygonClient,
    PrevDayData,
    Quote,
    RedisCache,
    Snapshot,
    TickerDetails,
    TickerType,
    Trade,
)
from gpt_quant_v2.infrastructure.external.polygon_client import (
    RateLimiter as PolygonRateLimiter,
)
from gpt_quant_v2.infrastructure.external.prompts import (
    SENTIMENT_ANALYSIS_SCHEMA,
    SENTIMENT_SYSTEM_PROMPT,
    SENTIMENT_USER_PROMPT_TEMPLATE,
    format_sentiment_user_prompt,
)
from gpt_quant_v2.infrastructure.external.quant_airflow_db_client import (
    AdjustedDailyBar,
    ConsensusRow,
    EarningsRow,
    NewsJudgmentRow,
    QuantAirflowDBClient,
    SectorIndexRow,
    SupplyDemandRow,
)
from gpt_quant_v2.infrastructure.external.tiingo_client import (
    NewsItem,
    TiingoClient,
)
from gpt_quant_v2.infrastructure.external.tiingo_client import (
    RateLimiter as TiingoRateLimiter,
)

__all__ = [
    # Polygon Client
    "PolygonClient",
    # Polygon Models
    "TickerDetails",
    "Bar",
    "Quote",
    "Trade",
    "Snapshot",
    "DayData",
    "MinuteData",
    "PrevDayData",
    # Polygon Enums
    "MarketType",
    "TickerType",
    # Polygon Cache
    "CacheBackend",
    "InMemoryCache",
    "RedisCache",
    # Polygon Rate Limiter
    "PolygonRateLimiter",
    # OpenAI Client
    "OpenAIClient",
    "OpenAIClientConfig",
    "SentimentAnalysis",
    "UsageStats",
    "RateLimitState",
    # Prompts
    "SENTIMENT_SYSTEM_PROMPT",
    "SENTIMENT_USER_PROMPT_TEMPLATE",
    "SENTIMENT_ANALYSIS_SCHEMA",
    "format_sentiment_user_prompt",
    # Tiingo Client
    "TiingoClient",
    "NewsItem",
    "TiingoRateLimiter",
    # quant-airflow TimescaleDB Client (read-only)
    "QuantAirflowDBClient",
    "AdjustedDailyBar",
    "SupplyDemandRow",
    "SectorIndexRow",
    "EarningsRow",
    "ConsensusRow",
    "NewsJudgmentRow",
]
