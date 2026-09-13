"""Tiingo News API client for fetching financial news data.

This module provides an async client for interacting with the Tiingo News API,
including support for pagination, rate limiting, retries, and deduplication.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid5

import httpx
from pydantic import BaseModel, Field, HttpUrl, field_validator

from daytrade_it.domain.entities.news import NewsArticle
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.shared.errors import error_codes
from daytrade_it.shared.errors.exceptions import TiingoAPIError
from daytrade_it.shared.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

# Retired defaults: this client is superseded for Korean-equity news by
# TossNewsClient/DartClient (Task 7). `TiingoSettings` was removed from
# shared.config.settings (Task 8) since Tiingo is a dead US-vendor
# dependency -- these module-level defaults keep this file constructible
# (e.g. by legacy interfaces/mcp callers) without depending on a settings
# class that no longer exists.
_DEFAULT_BASE_URL = "https://api.tiingo.com"
_DEFAULT_TIMEOUT = 30.0
_DEFAULT_RATE_LIMIT = 50

# Namespace UUID for generating deterministic article IDs from Tiingo IDs
TIINGO_NAMESPACE = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class NewsItem(BaseModel):
    """Pydantic model representing a Tiingo news article response.

    Maps directly to the Tiingo API news response format.
    """

    id: str = Field(description="Unique identifier from Tiingo")
    title: str = Field(description="Article headline")
    description: str = Field(default="", description="Article description/summary")
    url: HttpUrl = Field(description="Link to the original article")
    source: str = Field(description="News source/publisher")
    published_date: datetime = Field(alias="publishedDate", description="Publication timestamp")
    tickers: list[str] = Field(default_factory=list, description="Associated stock tickers")
    tags: list[str] = Field(default_factory=list, description="Content tags")

    model_config = {"populate_by_name": True}

    @field_validator("description", mode="before")
    @classmethod
    def ensure_description(cls, v: Any) -> str:
        """Ensure description is never None."""
        return v or ""

    @field_validator("tickers", mode="before")
    @classmethod
    def ensure_tickers_list(cls, v: Any) -> list[str]:
        """Ensure tickers is always a list."""
        if v is None:
            return []
        return list(v) if isinstance(v, (list, tuple)) else [v]

    @field_validator("tags", mode="before")
    @classmethod
    def ensure_tags_list(cls, v: Any) -> list[str]:
        """Ensure tags is always a list."""
        if v is None:
            return []
        return list(v) if isinstance(v, (list, tuple)) else [v]

    def to_domain_entity(self) -> NewsArticle:
        """Convert this NewsItem to a domain NewsArticle entity.

        Generates a deterministic UUID from the Tiingo article ID for consistency.
        """
        # Create deterministic UUID from Tiingo ID
        article_id = uuid5(TIINGO_NAMESPACE, self.id)

        # Convert ticker strings to Ticker value objects (filter invalid ones)
        valid_tickers: list[Ticker] = []
        for ticker_str in self.tickers:
            try:
                valid_tickers.append(Ticker(value=ticker_str))
            except ValueError:
                logger.debug(
                    "Skipping invalid ticker",
                    ticker=ticker_str,
                    article_id=self.id,
                )

        return NewsArticle(
            id=article_id,
            title=self.title,
            content=self.description,  # Use description as content
            summary=self.description[:500] if self.description else None,
            source=self.source,
            url=self.url,
            published_at=self.published_date,
            tickers=valid_tickers,
        )


class RateLimiter:
    """Token bucket rate limiter for API requests."""

    def __init__(self, requests_per_minute: int) -> None:
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum number of requests allowed per minute.
        """
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self._last_request_time: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire permission to make a request, waiting if necessary."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            time_since_last = now - self._last_request_time
            if time_since_last < self.interval:
                wait_time = self.interval - time_since_last
                logger.debug("Rate limiting", wait_seconds=wait_time)
                await asyncio.sleep(wait_time)
            self._last_request_time = asyncio.get_event_loop().time()


class TiingoClient:
    """Async client for the Tiingo News API.

    Provides methods for fetching news articles with support for:
    - Multiple tickers
    - Date filtering
    - Pagination
    - Rate limiting
    - Retry with exponential backoff
    - Deduplication

    Example:
        >>> async with TiingoClient() as client:
        ...     news = await client.get_news(["AAPL", "MSFT"], limit=50)
        ...     for article in news:
        ...         print(article.title)
    """

    NEWS_ENDPOINT = "/tiingo/news"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        rate_limit: int | None = None,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
    ) -> None:
        """Initialize the Tiingo client.

        Args:
            api_key: Tiingo API key. If not provided, defaults to empty
                (this client is retired for Korean-equity use -- see the
                module-level note above).
            base_url: API base URL. Defaults to the Tiingo API.
            timeout: Request timeout in seconds. Defaults to 30s.
            rate_limit: Maximum requests per minute. Defaults to 50.
            max_retries: Maximum number of retry attempts for failed requests.
            initial_retry_delay: Initial delay in seconds for exponential backoff.
        """
        self.api_key = api_key or ""
        self.base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout or _DEFAULT_TIMEOUT
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay

        rate_limit_value = rate_limit or _DEFAULT_RATE_LIMIT
        self.rate_limiter = RateLimiter(rate_limit_value)

        self._client: httpx.AsyncClient | None = None
        self._seen_ids: set[str] = set()

        logger.info(
            "TiingoClient initialized",
            base_url=self.base_url,
            timeout=self.timeout,
            rate_limit=rate_limit_value,
            max_retries=self.max_retries,
        )

    async def __aenter__(self) -> TiingoClient:
        """Enter async context manager."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.debug("TiingoClient closed")

    def clear_seen_ids(self) -> None:
        """Clear the set of seen article IDs for deduplication."""
        self._seen_ids.clear()
        logger.debug("Cleared seen article IDs")

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Make an HTTP request with rate limiting and exponential backoff retry.

        Args:
            method: HTTP method (GET, POST, etc.).
            endpoint: API endpoint path.
            params: Query parameters.

        Returns:
            Parsed JSON response as a list of dicts.

        Raises:
            TiingoAPIError: If the request fails after all retries.
        """
        client = await self._ensure_client()
        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                await self.rate_limiter.acquire()

                logger.debug(
                    "Making Tiingo API request",
                    method=method,
                    endpoint=endpoint,
                    params=params,
                    attempt=attempt + 1,
                )

                response = await client.request(method, endpoint, params=params)

                if response.status_code == 429:
                    # Rate limited - extract retry-after if available
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(
                        "Rate limited by Tiingo API",
                        retry_after=retry_after,
                        attempt=attempt + 1,
                    )
                    raise TiingoAPIError(
                        message="Rate limit exceeded",
                        error_code=error_codes.TIINGO_RATE_LIMIT,
                        status_code=429,
                        endpoint=endpoint,
                    )

                if response.status_code >= 500:
                    # Server error - retry
                    logger.warning(
                        "Tiingo API server error",
                        status_code=response.status_code,
                        attempt=attempt + 1,
                    )
                    raise TiingoAPIError(
                        message=f"Server error: {response.status_code}",
                        status_code=response.status_code,
                        endpoint=endpoint,
                    )

                if response.status_code >= 400:
                    # Client error - don't retry
                    response_text = response.text
                    logger.error(
                        "Tiingo API client error",
                        status_code=response.status_code,
                        response_body=response_text[:500],
                        endpoint=endpoint,
                    )
                    raise TiingoAPIError(
                        message=f"Client error: {response.status_code}",
                        status_code=response.status_code,
                        response_body=response_text,
                        endpoint=endpoint,
                    )

                data = response.json()

                # Tiingo returns a list for news endpoints
                if isinstance(data, list):
                    return data
                # Handle unexpected response format
                logger.warning(
                    "Unexpected Tiingo response format",
                    response_type=type(data).__name__,
                )
                return []

            except TiingoAPIError as e:
                last_exception = e
                if (
                    e.details.get("status_code", 0) < 500
                    and e.error_code != error_codes.TIINGO_RATE_LIMIT
                ):
                    # Don't retry client errors (except rate limits)
                    raise

            except httpx.TimeoutException as e:
                logger.warning(
                    "Tiingo API request timeout",
                    endpoint=endpoint,
                    attempt=attempt + 1,
                )
                last_exception = TiingoAPIError(
                    message="Request timeout",
                    endpoint=endpoint,
                    details={"cause": str(e)},
                )

            except httpx.RequestError as e:
                logger.warning(
                    "Tiingo API request error",
                    endpoint=endpoint,
                    error=str(e),
                    attempt=attempt + 1,
                )
                last_exception = TiingoAPIError(
                    message=f"Request error: {e}",
                    endpoint=endpoint,
                    details={"cause": str(e)},
                )

            # Exponential backoff
            if attempt < self.max_retries - 1:
                delay = self.initial_retry_delay * (2**attempt)
                logger.info(
                    "Retrying Tiingo API request",
                    delay=delay,
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        raise TiingoAPIError(
            message=f"Request failed after {self.max_retries} retries",
            endpoint=endpoint,
            details={
                "last_error": str(last_exception) if last_exception else "Unknown error",
            },
        )

    def _deduplicate_items(self, items: list[NewsItem]) -> list[NewsItem]:
        """Remove duplicate articles based on ID.

        Args:
            items: List of NewsItem objects.

        Returns:
            List of unique NewsItem objects not seen before.
        """
        unique_items: list[NewsItem] = []
        for item in items:
            if item.id not in self._seen_ids:
                self._seen_ids.add(item.id)
                unique_items.append(item)
        return unique_items

    async def get_news(
        self,
        tickers: Sequence[str],
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 100,
        offset: int = 0,
        deduplicate: bool = True,
    ) -> list[NewsItem]:
        """Fetch news articles for the specified tickers.

        Args:
            tickers: List of stock ticker symbols to get news for.
            start_date: Start date in YYYY-MM-DD format (inclusive).
            end_date: End date in YYYY-MM-DD format (inclusive).
            limit: Maximum number of articles to return (default: 100).
            offset: Number of articles to skip for pagination (default: 0).
            deduplicate: Whether to filter out previously seen articles (default: True).

        Returns:
            List of NewsItem objects.

        Example:
            >>> news = await client.get_news(["AAPL", "MSFT"], limit=50)
        """
        params: dict[str, Any] = {
            "tickers": ",".join(tickers),
            "limit": limit,
            "offset": offset,
        }

        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date

        logger.info(
            "Fetching news for tickers",
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )

        raw_items = await self._request_with_retry("GET", self.NEWS_ENDPOINT, params=params)

        items = [NewsItem.model_validate(item) for item in raw_items]

        if deduplicate:
            items = self._deduplicate_items(items)

        logger.info(
            "Fetched news articles",
            ticker_count=len(tickers),
            article_count=len(items),
        )

        return items

    async def get_news_for_ticker(
        self,
        ticker: str,
        limit: int = 50,
        start_date: str | None = None,
        end_date: str | None = None,
        deduplicate: bool = True,
    ) -> list[NewsItem]:
        """Fetch news articles for a single ticker.

        Convenience method that wraps get_news for a single ticker.

        Args:
            ticker: Stock ticker symbol.
            limit: Maximum number of articles to return (default: 50).
            start_date: Start date in YYYY-MM-DD format (inclusive).
            end_date: End date in YYYY-MM-DD format (inclusive).
            deduplicate: Whether to filter out previously seen articles (default: True).

        Returns:
            List of NewsItem objects.

        Example:
            >>> news = await client.get_news_for_ticker("AAPL", limit=20)
        """
        return await self.get_news(
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            deduplicate=deduplicate,
        )

    async def search_news(
        self,
        query: str,
        limit: int = 50,
        start_date: str | None = None,
        end_date: str | None = None,
        tickers: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        deduplicate: bool = True,
    ) -> list[NewsItem]:
        """Search for news articles matching a query.

        The Tiingo News API doesn't support full-text search directly,
        so this method filters results by tags and applies basic filtering.

        Args:
            query: Search query (used for tag matching).
            limit: Maximum number of articles to return (default: 50).
            start_date: Start date in YYYY-MM-DD format (inclusive).
            end_date: End date in YYYY-MM-DD format (inclusive).
            tickers: Optional list of tickers to filter by.
            tags: Optional list of tags to filter by.
            deduplicate: Whether to filter out previously seen articles (default: True).

        Returns:
            List of NewsItem objects matching the search criteria.

        Example:
            >>> news = await client.search_news("earnings", tickers=["AAPL"])
        """
        params: dict[str, Any] = {
            "limit": limit,
        }

        if tickers:
            params["tickers"] = ",".join(tickers)
        if tags:
            params["tags"] = ",".join(tags)
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date

        logger.info(
            "Searching news",
            query=query,
            tickers=tickers,
            tags=tags,
            limit=limit,
        )

        raw_items = await self._request_with_retry("GET", self.NEWS_ENDPOINT, params=params)

        items = [NewsItem.model_validate(item) for item in raw_items]

        # Client-side filtering by query in title/description
        query_lower = query.lower()
        filtered_items = [
            item
            for item in items
            if query_lower in item.title.lower() or query_lower in item.description.lower()
        ]

        if deduplicate:
            filtered_items = self._deduplicate_items(filtered_items)

        logger.info(
            "Search results",
            query=query,
            total_fetched=len(items),
            after_filter=len(filtered_items),
        )

        return filtered_items

    async def get_news_paginated(
        self,
        tickers: Sequence[str],
        start_date: str | None = None,
        end_date: str | None = None,
        total_limit: int = 500,
        page_size: int = 100,
        deduplicate: bool = True,
    ) -> list[NewsItem]:
        """Fetch news articles with automatic pagination.

        Fetches multiple pages of results until the total_limit is reached
        or no more results are available.

        Args:
            tickers: List of stock ticker symbols.
            start_date: Start date in YYYY-MM-DD format (inclusive).
            end_date: End date in YYYY-MM-DD format (inclusive).
            total_limit: Maximum total articles to fetch (default: 500).
            page_size: Number of articles per page (default: 100).
            deduplicate: Whether to filter out duplicate articles (default: True).

        Returns:
            List of NewsItem objects.

        Example:
            >>> news = await client.get_news_paginated(
            ...     ["AAPL", "MSFT"],
            ...     start_date="2024-01-01",
            ...     total_limit=200
            ... )
        """
        all_items: list[NewsItem] = []
        offset = 0

        logger.info(
            "Starting paginated news fetch",
            tickers=tickers,
            total_limit=total_limit,
            page_size=page_size,
        )

        while len(all_items) < total_limit:
            remaining = total_limit - len(all_items)
            current_limit = min(page_size, remaining)

            items = await self.get_news(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                limit=current_limit,
                offset=offset,
                deduplicate=deduplicate,
            )

            if not items:
                # No more results
                break

            all_items.extend(items)
            offset += len(items)

            logger.debug(
                "Fetched page",
                offset=offset,
                page_items=len(items),
                total_items=len(all_items),
            )

            # If we got fewer items than requested, we've hit the end
            if len(items) < current_limit:
                break

        logger.info(
            "Completed paginated fetch",
            total_items=len(all_items),
            pages_fetched=(offset // page_size) + 1,
        )

        return all_items

    async def get_news_as_articles(
        self,
        tickers: Sequence[str],
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 100,
    ) -> list[NewsArticle]:
        """Fetch news and convert to domain NewsArticle entities.

        Convenience method that fetches news and converts to domain entities.

        Args:
            tickers: List of stock ticker symbols.
            start_date: Start date in YYYY-MM-DD format (inclusive).
            end_date: End date in YYYY-MM-DD format (inclusive).
            limit: Maximum number of articles to return (default: 100).

        Returns:
            List of NewsArticle domain entities.

        Example:
            >>> articles = await client.get_news_as_articles(["AAPL"], limit=10)
            >>> for article in articles:
            ...     print(f"{article.title} - {article.source}")
        """
        items = await self.get_news(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

        return [item.to_domain_entity() for item in items]
