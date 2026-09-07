"""External API Clients."""

from krx_signal_engine.infrastructure.external.dart_client import (
    DartClient,
    map_dart_disclosure_to_domain,
)
from krx_signal_engine.infrastructure.external.news_poller import (
    CircuitState,
    NewsPoller,
)
from krx_signal_engine.infrastructure.external.openai_client import (
    OpenAIClient,
    OpenAIClientConfig,
    RateLimitState,
    SentimentAnalysis,
    UsageStats,
)
from krx_signal_engine.infrastructure.external.polygon_client import (
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
from krx_signal_engine.infrastructure.external.polygon_client import (
    RateLimiter as PolygonRateLimiter,
)
from krx_signal_engine.infrastructure.external.prompts import (
    SENTIMENT_ANALYSIS_SCHEMA,
    SENTIMENT_SYSTEM_PROMPT,
    SENTIMENT_USER_PROMPT_TEMPLATE,
    format_sentiment_user_prompt,
)
from krx_signal_engine.infrastructure.external.quant_airflow_db_client import (
    AdjustedDailyBar,
    ConsensusRow,
    EarningsRow,
    NewsJudgmentRow,
    QuantAirflowDBClient,
    SectorIndexRow,
    SupplyDemandRow,
)
from krx_signal_engine.infrastructure.external.tiingo_client import (
    NewsItem,
    TiingoClient,
)
from krx_signal_engine.infrastructure.external.tiingo_client import (
    RateLimiter as TiingoRateLimiter,
)
from krx_signal_engine.infrastructure.external.toss_news_client import (
    TossNewsClient,
    map_toss_article_to_domain,
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
    # Toss/DART lightweight news poller (Task 7)
    "TossNewsClient",
    "map_toss_article_to_domain",
    "DartClient",
    "map_dart_disclosure_to_domain",
    "NewsPoller",
    "CircuitState",
]
