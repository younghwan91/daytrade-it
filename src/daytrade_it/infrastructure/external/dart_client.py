"""DART disclosure poller, wrapping ``krx_news_client.DartScraper``.

This is the DART half of Task 7's lightweight news poller, and also the
fallback path the poller's circuit breaker (``news_poller.NewsPoller``)
degrades to when Toss polling fails repeatedly -- DART's OpenAPI is an
official, keyed API (unlike Toss's unofficial dashboard scrape), so it is
the more reliable of the two, at the cost of only covering disclosures, not
general market news.

Per the task brief, this module only *surfaces* ``disclosure_type`` on the
mapped domain entity for Phase 4's later risk-gate work to consume -- it
does not implement any severity/risk classification itself.

``DartScraper.scrape_news()`` (the library's generic "news" method) always
returns an empty list for DART -- the library only implements disclosures
for this source (see ``krx_news_client/scrapers/dart.py``). This client
exposes disclosures.
"""

from __future__ import annotations

from uuid import UUID, uuid5

from krx_news_client import DartQuotaExceededError, DartScraper
from krx_news_client import Disclosure as KrxDisclosure

from daytrade_it.domain.entities.news import NewsArticle
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.shared.config import get_settings
from daytrade_it.shared.errors import error_codes
from daytrade_it.shared.errors.exceptions import DartAPIError
from daytrade_it.shared.logging import get_logger

logger = get_logger(__name__)

# Namespace UUID for generating deterministic article IDs from krx-news-client
# disclosure IDs (mirrors TOSS_NEWS_NAMESPACE / TiingoClient's approach).
DART_DISCLOSURE_NAMESPACE = UUID("2f1e9c2a-2b3a-4c5d-8e6f-1a2b3c4d5e6f")

_EMPTY_CONTENT_FALLBACK = "(no content provided by DART)"

# Re-exported so callers of this module don't need to import from
# krx_news_client directly to catch quota exhaustion.
__all__ = [
    "DART_DISCLOSURE_NAMESPACE",
    "DartClient",
    "DartQuotaExceededError",
    "map_dart_disclosure_to_domain",
]


def map_dart_disclosure_to_domain(disclosure: KrxDisclosure) -> NewsArticle:
    """Convert a ``krx_news_client`` DART ``Disclosure`` to the domain entity.

    ``Disclosure`` has no free-text content field (unlike Toss's
    ``NewsArticle``) -- DART's list API only gives structured metadata, the
    full filing lives behind ``disclosure.url``. Content is synthesized from
    company + disclosure type so the domain entity's ``content``
    (min_length=1) is never violated.
    """
    disclosure_id = uuid5(DART_DISCLOSURE_NAMESPACE, disclosure.id)

    content_parts = [
        p for p in (disclosure.company, disclosure.disclosure_type or disclosure.title) if p
    ]
    content = ": ".join(content_parts) if content_parts else _EMPTY_CONTENT_FALLBACK

    tickers: list[Ticker] = []
    if disclosure.ticker:
        try:
            tickers.append(Ticker(value=disclosure.ticker))
        except ValueError:
            logger.debug(
                "Skipping invalid ticker from DART disclosure",
                ticker=disclosure.ticker,
                disclosure_id=disclosure.id,
            )

    return NewsArticle(
        id=disclosure_id,
        title=disclosure.title or disclosure.disclosure_type or _EMPTY_CONTENT_FALLBACK,
        content=content,
        summary=None,
        source=disclosure.source.value,
        url=disclosure.url,
        published_at=disclosure.published_at,
        tickers=tickers,
        disclosure_type=disclosure.disclosure_type or None,
    )


class DartClient:
    """Poll loop + dedup cache over ``krx_news_client.DartScraper`` disclosures.

    Example:
        >>> async with DartClient() as client:
        ...     new_disclosures = await client.poll_once()
        ...     for article in new_disclosures:
        ...         print(article.title, article.disclosure_type)
    """

    def __init__(
        self,
        api_key: str | None = None,
        scraper: DartScraper | None = None,
    ) -> None:
        """Initialize the DART client.

        Args:
            api_key: DART OpenAPI key. Falls back to
                ``settings.krx_news.dart_api_key``.
            scraper: Injectable ``DartScraper`` instance, primarily for tests.
        """
        settings = get_settings().krx_news
        self.api_key = api_key or settings.dart_api_key.get_secret_value()
        self._scraper = scraper or DartScraper(api_key=self.api_key)
        self._seen_ids: set[str] = set()

        logger.info(
            "DartClient initialized",
            api_key_configured=bool(self.api_key),
        )

    async def __aenter__(self) -> DartClient:
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying scraper's HTTP client."""
        await self._scraper.close()

    def clear_seen_ids(self) -> None:
        """Clear the dedup cache. Useful for tests or a deliberate resync."""
        self._seen_ids.clear()

    @property
    def seen_count(self) -> int:
        """Number of distinct disclosure ids seen so far this session."""
        return len(self._seen_ids)

    async def poll_once(self) -> list[NewsArticle]:
        """Fetch today's/yesterday's DART disclosures and return only new ones.

        Raises:
            DartQuotaExceededError: Propagated as-is from the library when
                the configured API key has exhausted its daily quota
                (status=020) -- a distinct condition from "no disclosures in
                range" (status=013, silently returns []), which callers
                such as the circuit breaker may want to handle differently
                from a generic failure.
            DartAPIError: If the underlying scraper call raises any other
                unexpected exception.
        """
        try:
            raw_disclosures = await self._scraper.scrape_disclosures()
        except DartQuotaExceededError:
            logger.warning("DART daily quota exhausted")
            raise
        except Exception as exc:  # noqa: BLE001 -- wrap any other failure shape uniformly
            logger.warning("DART disclosure poll failed", error=str(exc))
            raise DartAPIError(
                message=f"DART disclosure request failed: {exc}",
                error_code=error_codes.DART_API_ERROR,
            ) from exc

        new_disclosures = [d for d in raw_disclosures if d.id not in self._seen_ids]
        for d in new_disclosures:
            self._seen_ids.add(d.id)

        logger.info(
            "DART disclosure poll complete",
            fetched=len(raw_disclosures),
            new=len(new_disclosures),
            seen_total=len(self._seen_ids),
        )

        return [map_dart_disclosure_to_domain(d) for d in new_disclosures]
