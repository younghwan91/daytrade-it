"""
Polygon.io Market Data Client.

Provides async access to Polygon.io REST API with:
- Rate limiting (configurable, defaults to 5 req/min for free tier)
- Caching with TTL (Redis or in-memory fallback)
- Retry logic with exponential backoff
- Structured logging with correlation ID support

Usage:
    >>> from gpt_quant_v2.infrastructure.external import PolygonClient
    >>> from gpt_quant_v2.shared.config import get_settings
    >>>
    >>> async with PolygonClient(get_settings().polygon) as client:
    ...     details = await client.get_ticker_details("AAPL")
    ...     bars = await client.get_aggregates("AAPL", 1, "day", "2024-01-01", "2024-01-31")
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, Field

from gpt_quant_v2.shared.errors import error_codes
from gpt_quant_v2.shared.errors.exceptions import PolygonAPIError
from gpt_quant_v2.shared.logging import get_logger

if TYPE_CHECKING:
    from gpt_quant_v2.shared.config.settings import PolygonSettings

logger = get_logger(__name__)


# =============================================================================
# Response Models
# =============================================================================


class MarketType(StrEnum):
    """Market type classification."""

    STOCKS = "stocks"
    CRYPTO = "crypto"
    FX = "fx"
    OTC = "otc"
    INDICES = "indices"


class TickerType(StrEnum):
    """Ticker type classification."""

    CS = "CS"  # Common Stock
    PFD = "PFD"  # Preferred Stock
    WARRANT = "WARRANT"
    RIGHT = "RIGHT"
    BOND = "BOND"
    ETF = "ETF"
    ETN = "ETN"
    FUND = "FUND"
    SP = "SP"  # Structured Product
    ADRC = "ADRC"  # ADR Common
    ADRP = "ADRP"  # ADR Preferred
    ADRW = "ADRW"  # ADR Warrant
    ADRR = "ADRR"  # ADR Right
    UNIT = "UNIT"
    OS = "OS"  # Ordinary Shares
    GDR = "GDR"  # Global Depositary Receipt
    OTHER = "OTHER"


class TickerDetails(BaseModel):
    """Ticker/security details from Polygon."""

    ticker: str = Field(..., description="The ticker symbol")
    name: str = Field(..., description="Company/security name")
    market: MarketType | None = Field(None, description="Market type")
    locale: str | None = Field(None, description="Locale (e.g., 'us')")
    primary_exchange: str | None = Field(None, description="Primary exchange code")
    type: TickerType | str | None = Field(None, description="Security type")
    active: bool = Field(True, description="Whether the ticker is active")
    currency_name: str | None = Field(None, description="Currency name")
    cik: str | None = Field(None, description="CIK number for SEC filings")
    composite_figi: str | None = Field(None, description="Composite FIGI")
    share_class_figi: str | None = Field(None, description="Share class FIGI")
    market_cap: float | None = Field(None, description="Market capitalization")
    phone_number: str | None = Field(None, description="Company phone number")
    address: dict[str, Any] | None = Field(None, description="Company address")
    description: str | None = Field(None, description="Company description")
    sic_code: str | None = Field(None, description="SIC code")
    sic_description: str | None = Field(None, description="SIC description")
    ticker_root: str | None = Field(None, description="Ticker root")
    homepage_url: str | None = Field(None, description="Company homepage URL")
    total_employees: int | None = Field(None, description="Total employees")
    list_date: str | None = Field(None, description="IPO/listing date")
    branding: dict[str, Any] | None = Field(None, description="Branding info")
    share_class_shares_outstanding: int | None = Field(
        None, description="Shares outstanding"
    )
    weighted_shares_outstanding: int | None = Field(
        None, description="Weighted shares outstanding"
    )
    round_lot: int | None = Field(None, description="Round lot size")


class Bar(BaseModel):
    """OHLCV bar/candle data."""

    open: float = Field(..., alias="o", description="Open price")
    high: float = Field(..., alias="h", description="High price")
    low: float = Field(..., alias="l", description="Low price")
    close: float = Field(..., alias="c", description="Close price")
    volume: float = Field(..., alias="v", description="Volume")
    vwap: float | None = Field(None, alias="vw", description="Volume weighted avg price")
    timestamp: int = Field(..., alias="t", description="Unix timestamp in ms")
    transactions: int | None = Field(None, alias="n", description="Number of transactions")

    model_config = {"populate_by_name": True}

    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp / 1000)


class Quote(BaseModel):
    """Last quote (NBBO) data."""

    ask_price: float = Field(..., alias="P", description="Ask price")
    ask_size: int = Field(..., alias="S", description="Ask size")
    ask_exchange: int | None = Field(None, alias="x", description="Ask exchange ID")
    bid_price: float = Field(..., alias="p", description="Bid price")
    bid_size: int = Field(..., alias="s", description="Bid size")
    bid_exchange: int | None = Field(None, alias="X", description="Bid exchange ID")
    timestamp: int = Field(..., alias="t", description="Unix timestamp in ns")
    conditions: list[int] | None = Field(None, alias="c", description="Condition flags")
    indicators: list[int] | None = Field(None, alias="i", description="Indicators")
    sequence_number: int | None = Field(
        None, alias="q", description="Sequence number"
    )
    tape: int | None = Field(None, alias="z", description="Tape")

    model_config = {"populate_by_name": True}

    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp / 1_000_000_000)

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask_price - self.bid_price

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.ask_price + self.bid_price) / 2


class Trade(BaseModel):
    """Last trade data."""

    price: float = Field(..., alias="p", description="Trade price")
    size: int = Field(..., alias="s", description="Trade size")
    exchange: int = Field(..., alias="x", description="Exchange ID")
    timestamp: int = Field(..., alias="t", description="Unix timestamp in ns")
    conditions: list[int] | None = Field(None, alias="c", description="Condition flags")
    trade_id: str | None = Field(None, alias="i", description="Trade ID")
    tape: int | None = Field(None, alias="z", description="Tape")
    sequence_number: int | None = Field(
        None, alias="q", description="Sequence number"
    )

    model_config = {"populate_by_name": True}

    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp / 1_000_000_000)


class DayData(BaseModel):
    """Daily aggregate data for snapshot."""

    open: float = Field(..., alias="o", description="Open price")
    high: float = Field(..., alias="h", description="High price")
    low: float = Field(..., alias="l", description="Low price")
    close: float = Field(..., alias="c", description="Close price")
    volume: float = Field(..., alias="v", description="Volume")
    vwap: float | None = Field(None, alias="vw", description="VWAP")

    model_config = {"populate_by_name": True}


class MinuteData(BaseModel):
    """Most recent minute aggregate data."""

    open: float = Field(..., alias="o", description="Open price")
    high: float = Field(..., alias="h", description="High price")
    low: float = Field(..., alias="l", description="Low price")
    close: float = Field(..., alias="c", description="Close price")
    volume: float = Field(..., alias="v", description="Volume")
    vwap: float | None = Field(None, alias="vw", description="VWAP")
    accumulated_volume: float | None = Field(None, alias="av", description="Accumulated volume")

    model_config = {"populate_by_name": True}


class PrevDayData(BaseModel):
    """Previous day's aggregate data."""

    open: float = Field(..., alias="o", description="Open price")
    high: float = Field(..., alias="h", description="High price")
    low: float = Field(..., alias="l", description="Low price")
    close: float = Field(..., alias="c", description="Close price")
    volume: float = Field(..., alias="v", description="Volume")
    vwap: float | None = Field(None, alias="vw", description="VWAP")

    model_config = {"populate_by_name": True}


class Snapshot(BaseModel):
    """Full market snapshot for a ticker."""

    ticker: str = Field(..., description="Ticker symbol")
    todays_change: float | None = Field(
        None, alias="todaysChange", description="Today's price change"
    )
    todays_change_perc: float | None = Field(
        None, alias="todaysChangePerc", description="Today's percent change"
    )
    updated: int | None = Field(None, description="Last update timestamp in ns")
    day: DayData | None = Field(None, description="Current day's aggregate")
    min: MinuteData | None = Field(None, description="Most recent minute aggregate")
    prev_day: PrevDayData | None = Field(
        None, alias="prevDay", description="Previous day's aggregate"
    )
    last_trade: Trade | None = Field(
        None, alias="lastTrade", description="Most recent trade"
    )
    last_quote: Quote | None = Field(
        None, alias="lastQuote", description="Most recent quote"
    )

    model_config = {"populate_by_name": True}

    @property
    def datetime(self) -> datetime | None:
        """Convert updated timestamp to datetime."""
        if self.updated:
            return datetime.fromtimestamp(self.updated / 1_000_000_000)
        return None


# =============================================================================
# Cache Interface & Implementations
# =============================================================================


class CacheBackend[T](ABC):
    """Abstract cache backend interface."""

    @abstractmethod
    async def get(self, key: str) -> T | None:
        """Get value from cache."""
        ...

    @abstractmethod
    async def set(self, key: str, value: T, ttl: int | None = None) -> None:
        """Set value in cache with optional TTL in seconds."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached values."""
        ...


@dataclass
class CacheEntry[T]:
    """Cache entry with expiration tracking."""

    value: T
    expires_at: float | None = None

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.monotonic() > self.expires_at


class InMemoryCache(CacheBackend[Any]):
    """Thread-safe in-memory cache with TTL support."""

    def __init__(self) -> None:
        self._cache: dict[str, CacheEntry[Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """Get value from cache if not expired."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._cache[key]
                return None
            return entry.value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value with optional TTL in seconds."""
        expires_at = time.monotonic() + ttl if ttl else None
        async with self._lock:
            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)

    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        async with self._lock:
            self._cache.pop(key, None)

    async def clear(self) -> None:
        """Clear all cached values."""
        async with self._lock:
            self._cache.clear()


class RedisCache(CacheBackend[Any]):
    """Redis cache backend."""

    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "polygon") -> None:
        self._redis_url = redis_url
        self._prefix = prefix
        self._redis: Any = None

    async def _get_redis(self) -> Any:
        """Lazy initialize Redis connection."""
        if self._redis is None:
            import redis.asyncio as redis

            self._redis = redis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self._prefix}:{key}"

    async def get(self, key: str) -> Any | None:
        """Get value from Redis."""
        import json

        redis_client = await self._get_redis()
        value = await redis_client.get(self._make_key(key))
        if value is not None:
            return json.loads(value)
        return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in Redis with optional TTL."""
        import json

        redis_client = await self._get_redis()
        serialized = json.dumps(value)
        if ttl:
            await redis_client.setex(self._make_key(key), ttl, serialized)
        else:
            await redis_client.set(self._make_key(key), serialized)

    async def delete(self, key: str) -> None:
        """Delete value from Redis."""
        redis_client = await self._get_redis()
        await redis_client.delete(self._make_key(key))

    async def clear(self) -> None:
        """Clear all prefixed keys."""
        redis_client = await self._get_redis()
        cursor = 0
        pattern = f"{self._prefix}:*"
        while True:
            cursor, keys = await redis_client.scan(cursor, match=pattern, count=100)
            if keys:
                await redis_client.delete(*keys)
            if cursor == 0:
                break

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


# =============================================================================
# Rate Limiter
# =============================================================================


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter.

    Allows bursting up to max_requests, then throttles to stay within rate limit.
    """

    max_requests: int = 5  # Requests per period
    period_seconds: float = 60.0  # Period duration
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)
    _lock: asyncio.Lock = field(init=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.max_requests)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Acquire a rate limit token, waiting if necessary.

        Raises:
            asyncio.CancelledError: If waiting is cancelled
        """
        async with self._lock:
            await self._refill()

            if self._tokens >= 1:
                self._tokens -= 1
                return

            # Calculate wait time for next token
            tokens_needed = 1 - self._tokens
            wait_time = (tokens_needed / self.max_requests) * self.period_seconds
            logger.debug(
                "Rate limit reached, waiting",
                wait_seconds=round(wait_time, 2),
                tokens_available=round(self._tokens, 2),
            )

        # Wait outside lock
        await asyncio.sleep(wait_time)

        async with self._lock:
            await self._refill()
            self._tokens = max(0, self._tokens - 1)

    async def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        tokens_to_add = (elapsed / self.period_seconds) * self.max_requests
        self._tokens = min(self.max_requests, self._tokens + tokens_to_add)
        self._last_refill = now


# =============================================================================
# Polygon Client
# =============================================================================


class PolygonClient:
    """
    Async Polygon.io REST API client.

    Features:
        - Rate limiting (configurable, defaults to 5 req/min)
        - Caching with TTL (Redis or in-memory)
        - Retry with exponential backoff
        - Structured logging with correlation ID

    Example:
        >>> async with PolygonClient(settings.polygon) as client:
        ...     details = await client.get_ticker_details("AAPL")
        ...     print(f"Market cap: ${details.market_cap:,.0f}")
    """

    DEFAULT_TTL = 300  # 5 minutes default cache TTL
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY = 1.0

    def __init__(
        self,
        settings: PolygonSettings,
        *,
        cache: CacheBackend[Any] | None = None,
        cache_ttl: int = DEFAULT_TTL,
        use_redis: bool = False,
        redis_url: str = "redis://localhost:6379",
    ) -> None:
        """
        Initialize the Polygon client.

        Args:
            settings: Polygon configuration settings
            cache: Optional custom cache backend
            cache_ttl: Default cache TTL in seconds
            use_redis: Whether to use Redis for caching (falls back to in-memory)
            redis_url: Redis connection URL if use_redis is True
        """
        self._settings = settings
        self._cache_ttl = cache_ttl
        self._client: httpx.AsyncClient | None = None
        self._rate_limiter = RateLimiter(
            max_requests=settings.rate_limit,
            period_seconds=60.0,
        )

        # Initialize cache
        self._cache: CacheBackend[Any]
        if cache is not None:
            self._cache = cache
        elif use_redis:
            self._cache = RedisCache(redis_url=redis_url)
        else:
            self._cache = InMemoryCache()

        self._owns_cache = cache is None  # Track if we created the cache

    async def __aenter__(self) -> PolygonClient:
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._settings.base_url,
                timeout=httpx.Timeout(self._settings.timeout),
                headers={
                    "Authorization": f"Bearer {self._settings.api_key.get_secret_value()}",
                    "Accept": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

        if self._owns_cache and isinstance(self._cache, RedisCache):
            await self._cache.close()

    def _cache_key(self, endpoint: str, **params: Any) -> str:
        """Generate cache key from endpoint and params."""
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if v is not None)
        key_data = f"{endpoint}?{param_str}" if param_str else endpoint
        return hashlib.md5(key_data.encode()).hexdigest()

    async def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        use_cache: bool = True,
        cache_ttl: int | None = None,
    ) -> dict[str, Any]:
        """
        Make an API request with rate limiting, caching, and retry logic.

        Args:
            method: HTTP method
            endpoint: API endpoint path
            params: Query parameters
            use_cache: Whether to use caching for this request
            cache_ttl: Cache TTL override for this request

        Returns:
            Parsed JSON response

        Raises:
            PolygonAPIError: On API errors or after max retries
        """
        params = params or {}
        ttl = cache_ttl if cache_ttl is not None else self._cache_ttl

        # Check cache for GET requests
        cache_key = self._cache_key(endpoint, **params)
        if method == "GET" and use_cache:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit", endpoint=endpoint, cache_key=cache_key[:8])
                if isinstance(cached, dict):
                    return cached
                return dict(cached)

        # Rate limiting
        await self._rate_limiter.acquire()

        # Make request with retry logic
        client = await self._ensure_client()
        last_error: Exception | None = None

        for attempt in range(self.MAX_RETRIES):
            try:
                response = await client.request(method, endpoint, params=params)

                # Handle rate limiting response
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(
                        "Rate limited by Polygon API",
                        endpoint=endpoint,
                        retry_after=retry_after,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(retry_after)
                    continue

                # Raise for other HTTP errors
                if response.status_code >= 400:
                    error_body = response.text
                    raise PolygonAPIError(
                        message=f"Polygon API error: {response.status_code}",
                        error_code=error_codes.POLYGON_API_ERROR,
                        status_code=response.status_code,
                        response_body=error_body,
                        endpoint=endpoint,
                    )

                data: dict[str, Any] = response.json()

                # Check for API-level errors
                if data.get("status") == "ERROR":
                    raise PolygonAPIError(
                        message=str(data.get("error", "Unknown Polygon API error")),
                        error_code=error_codes.POLYGON_INVALID_RESPONSE,
                        endpoint=endpoint,
                        details={"response": data},
                    )

                # Cache successful response
                if method == "GET" and use_cache:
                    await self._cache.set(cache_key, data, ttl)
                    logger.debug("Cache set", endpoint=endpoint, cache_key=cache_key[:8], ttl=ttl)

                logger.debug(
                    "API request successful",
                    endpoint=endpoint,
                    status_code=response.status_code,
                )
                return data

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    "Request timeout",
                    endpoint=endpoint,
                    attempt=attempt + 1,
                    max_retries=self.MAX_RETRIES,
                )
            except httpx.HTTPStatusError as e:
                last_error = e
                # Don't retry client errors (4xx except 429)
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise PolygonAPIError(
                        message=f"Client error: {e.response.status_code}",
                        error_code=error_codes.POLYGON_API_ERROR,
                        status_code=e.response.status_code,
                        response_body=e.response.text,
                        endpoint=endpoint,
                    ) from e
                logger.warning(
                    "Request failed",
                    endpoint=endpoint,
                    error=str(e),
                    attempt=attempt + 1,
                    max_retries=self.MAX_RETRIES,
                )
            except httpx.RequestError as e:
                last_error = e
                logger.warning(
                    "Request error",
                    endpoint=endpoint,
                    error=str(e),
                    attempt=attempt + 1,
                    max_retries=self.MAX_RETRIES,
                )
            except PolygonAPIError:
                raise

            # Exponential backoff
            if attempt < self.MAX_RETRIES - 1:
                delay = self.INITIAL_RETRY_DELAY * (2**attempt)
                logger.debug("Retrying after delay", delay_seconds=delay)
                await asyncio.sleep(delay)

        raise PolygonAPIError(
            message=f"Max retries exceeded for {endpoint}",
            error_code=error_codes.POLYGON_API_ERROR,
            endpoint=endpoint,
            details={"last_error": str(last_error)},
        )

    # =========================================================================
    # Public API Methods
    # =========================================================================

    async def get_ticker_details(
        self,
        ticker: str,
        *,
        date: str | None = None,
        use_cache: bool = True,
    ) -> TickerDetails:
        """
        Get detailed information about a ticker.

        Args:
            ticker: Stock ticker symbol
            date: Optional date for historical data (YYYY-MM-DD)
            use_cache: Whether to use cached response

        Returns:
            TickerDetails with company/security information

        Raises:
            PolygonAPIError: On API errors
        """
        endpoint = f"/v3/reference/tickers/{ticker.upper()}"
        params = {"date": date} if date else {}

        data = await self._request("GET", endpoint, params=params, use_cache=use_cache)

        results = data.get("results", {})
        return TickerDetails.model_validate(results)

    async def get_aggregates(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        from_date: str,
        to_date: str,
        *,
        adjusted: bool = True,
        sort: str = "asc",
        limit: int = 50000,
        use_cache: bool = True,
    ) -> list[Bar]:
        """
        Get aggregate bars (OHLCV) for a ticker.

        Args:
            ticker: Stock ticker symbol
            multiplier: Size of the timespan multiplier (e.g., 1 for 1 day)
            timespan: Size of the time window (minute, hour, day, week, month, quarter, year)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            adjusted: Whether to adjust for splits
            sort: Sort direction (asc or desc)
            limit: Maximum number of results
            use_cache: Whether to use cached response

        Returns:
            List of Bar objects with OHLCV data

        Raises:
            PolygonAPIError: On API errors
        """
        endpoint = f"/v2/aggs/ticker/{ticker.upper()}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            "adjusted": str(adjusted).lower(),
            "sort": sort,
            "limit": limit,
        }

        data = await self._request("GET", endpoint, params=params, use_cache=use_cache)

        results = data.get("results", [])
        return [Bar.model_validate(bar) for bar in results]

    async def get_last_quote(
        self,
        ticker: str,
        *,
        use_cache: bool = False,  # Real-time data, don't cache by default
    ) -> Quote:
        """
        Get the most recent NBBO (quote) for a ticker.

        Args:
            ticker: Stock ticker symbol
            use_cache: Whether to use cached response (default: False for real-time)

        Returns:
            Quote with bid/ask data

        Raises:
            PolygonAPIError: On API errors
        """
        endpoint = f"/v2/last/nbbo/{ticker.upper()}"

        data = await self._request(
            "GET", endpoint, use_cache=use_cache, cache_ttl=5  # Short TTL if cached
        )

        results = data.get("results", {})
        return Quote.model_validate(results)

    async def get_last_trade(
        self,
        ticker: str,
        *,
        use_cache: bool = False,  # Real-time data
    ) -> Trade:
        """
        Get the most recent trade for a ticker.

        Args:
            ticker: Stock ticker symbol
            use_cache: Whether to use cached response (default: False for real-time)

        Returns:
            Trade with price/size/exchange data

        Raises:
            PolygonAPIError: On API errors
        """
        endpoint = f"/v2/last/trade/{ticker.upper()}"

        data = await self._request(
            "GET", endpoint, use_cache=use_cache, cache_ttl=5
        )

        results = data.get("results", {})
        return Trade.model_validate(results)

    async def get_previous_close(
        self,
        ticker: str,
        *,
        adjusted: bool = True,
        use_cache: bool = True,
    ) -> Bar:
        """
        Get the previous day's OHLCV bar for a ticker.

        Args:
            ticker: Stock ticker symbol
            adjusted: Whether to adjust for splits
            use_cache: Whether to use cached response

        Returns:
            Bar with previous close data

        Raises:
            PolygonAPIError: On API errors
        """
        endpoint = f"/v2/aggs/ticker/{ticker.upper()}/prev"
        params = {"adjusted": str(adjusted).lower()}

        data = await self._request("GET", endpoint, params=params, use_cache=use_cache)

        results = data.get("results", [])
        if not results:
            raise PolygonAPIError(
                message=f"No previous close data found for {ticker}",
                error_code=error_codes.POLYGON_INVALID_RESPONSE,
                endpoint=endpoint,
            )

        return Bar.model_validate(results[0])

    async def get_snapshot(
        self,
        ticker: str,
        *,
        use_cache: bool = False,  # Real-time data
    ) -> Snapshot:
        """
        Get a full market snapshot for a ticker.

        Includes current day's data, previous day's data, last trade, and last quote.

        Args:
            ticker: Stock ticker symbol
            use_cache: Whether to use cached response (default: False for real-time)

        Returns:
            Snapshot with comprehensive market data

        Raises:
            PolygonAPIError: On API errors
        """
        endpoint = f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker.upper()}"

        data = await self._request(
            "GET", endpoint, use_cache=use_cache, cache_ttl=10
        )

        ticker_data = data.get("ticker", {})
        return Snapshot.model_validate(ticker_data)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def clear_cache(self) -> None:
        """Clear all cached data."""
        await self._cache.clear()
        logger.info("Polygon client cache cleared")

    async def health_check(self) -> bool:
        """
        Check if the Polygon API is accessible.

        Returns:
            True if API is healthy, False otherwise
        """
        try:
            # Use a simple endpoint that doesn't consume much rate limit
            await self._request(
                "GET",
                "/v3/reference/tickers",
                params={"limit": 1},
                use_cache=False,
            )
            return True
        except PolygonAPIError:
            return False


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Client
    "PolygonClient",
    # Models
    "TickerDetails",
    "Bar",
    "Quote",
    "Trade",
    "Snapshot",
    "DayData",
    "MinuteData",
    "PrevDayData",
    # Enums
    "MarketType",
    "TickerType",
    # Cache
    "CacheBackend",
    "InMemoryCache",
    "RedisCache",
    # Rate Limiter
    "RateLimiter",
]
