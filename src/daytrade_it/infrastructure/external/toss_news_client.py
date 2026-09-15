"""Toss dashboard news poller for latency-sensitive news-to-signal.

Wraps ``krx_news_client.TossScraper`` (an unofficial Toss Invest dashboard
API client) with a poll loop and dedup-by-id cache. This is the *other* data
path from Task 6's ``QuantAirflowDBClient``: spike 004 concluded that
``quant-airflow`` already handles slow/batch Korean news ingestion into its
own TimescaleDB, so this client is deliberately kept separate and lighter --
a fast-cadence poller for latency-sensitive signals, not a historical store.

``TossScraper.scrape_news()`` returns only a current dashboard snapshot with
no pagination or date filter (confirmed in Task 2's spike,
``scripts/spikes/toss_reliability_spike.py`` / ``docs/spikes/002-toss-reliability.md``) --
repeated calls return the same ~76 articles until the dashboard's own content
rotates. This client polls that snapshot on an interval and only emits
articles it has not already returned, using the library's stable
``NewsArticle.id`` (``f"{source}:{md5(url)[:12]}"``, per
``krx_news_client.scrapers.base.make_article_id``).

Follows the same client-class conventions as ``tiingo_client.TiingoClient``:
a settings-backed constructor, a ``close()`` for the underlying HTTP client,
and a mapping function to the domain ``NewsArticle`` entity (mirroring
``TiingoClient.NewsItem.to_domain_entity()``). The mapping here is a free
function rather than a method on a Pydantic response model, because the
"response model" is ``krx_news_client``'s own ``NewsArticle`` schema, which
this repo does not own and should not subclass/monkeypatch.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import UUID, uuid5

import httpx
from krx_news_client import Deduplicator, TossScraper
from krx_news_client import NewsArticle as KrxNewsArticle
from krx_news_client.scrapers.toss import NEWS_DETAIL_URL, extract_body_text, news_id_from_url

from daytrade_it.domain.entities.news import NewsArticle
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.shared.config import get_settings
from daytrade_it.shared.errors import error_codes
from daytrade_it.shared.errors.exceptions import TossNewsAPIError
from daytrade_it.shared.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = get_logger(__name__)

# Namespace UUID for generating deterministic article IDs from krx-news-client
# article IDs, so the same Toss article maps to the same domain entity id
# across polls/process restarts (mirrors TiingoClient's TIINGO_NAMESPACE).
TOSS_NEWS_NAMESPACE = UUID("b6a0f6a0-8e3b-4b1a-9b8a-6a2f2f2b9a1a")

#: Fallback used when both the library's content and summary are empty, so
#: the domain NewsArticle's ``content`` (min_length=1) is never violated.
_EMPTY_CONTENT_FALLBACK = "(no content provided by Toss)"


def _new_seen_cache() -> Deduplicator[KrxNewsArticle]:
    """Dedup cache keyed by the library's stable ``NewsArticle.id``.

    The key is pinned explicitly (not the library default) so the dedup key
    and the uuid5 domain id stay derived from the same value.
    """
    return Deduplicator(key=lambda a: a.id)


def map_toss_article_to_domain(article: KrxNewsArticle) -> NewsArticle:
    """Convert a ``krx_news_client`` Toss ``NewsArticle`` to the domain entity.

    Generates a deterministic UUID from the library's stable article id
    (``source:url-hash``) for consistency across polls, same pattern as
    ``TiingoClient.NewsItem.to_domain_entity()``.
    """
    article_id = uuid5(TOSS_NEWS_NAMESPACE, article.id)

    content = article.content or article.summary or article.title or _EMPTY_CONTENT_FALLBACK
    summary = article.summary or (article.content[:500] if article.content else None)

    valid_tickers: list[Ticker] = []
    for ticker_str in article.tickers:
        try:
            valid_tickers.append(Ticker(value=ticker_str))
        except ValueError:
            logger.debug(
                "Skipping invalid ticker from Toss article",
                ticker=ticker_str,
                article_id=article.id,
            )

    return NewsArticle(
        id=article_id,
        title=article.title,
        content=content,
        summary=summary,
        source=article.source.value,
        url=article.url,
        published_at=article.published_at,
        tickers=valid_tickers,
        category=article.category.value,
    )


class TossNewsClient:
    """Poll loop + dedup cache over ``krx_news_client.TossScraper``.

    Example:
        >>> async with TossNewsClient() as client:
        ...     new_articles = await client.poll_once()
        ...     for article in new_articles:
        ...         print(article.title)
    """

    def __init__(
        self,
        poll_interval: float | None = None,
        scraper: TossScraper | None = None,
    ) -> None:
        """Initialize the Toss news client.

        Args:
            poll_interval: Seconds between polls in ``poll_forever``. Falls
                back to ``settings.krx_news.toss_poll_interval_seconds``
                (conservative, few-minutes-range default -- see that
                setting's docstring for why).
            scraper: Injectable ``TossScraper`` instance, primarily for
                tests (to substitute a fake that raises on demand).
        """
        settings = get_settings().krx_news
        self.poll_interval = (
            poll_interval if poll_interval is not None else settings.toss_poll_interval_seconds
        )
        self._scraper = scraper or TossScraper()
        self._seen = _new_seen_cache()

        logger.info("TossNewsClient initialized", poll_interval=self.poll_interval)

    async def __aenter__(self) -> TossNewsClient:
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying scraper's HTTP client."""
        await self._scraper.close()

    def clear_seen_ids(self) -> None:
        """Clear the dedup cache. Useful for tests or a deliberate resync."""
        self._seen = _new_seen_cache()

    @property
    def seen_count(self) -> int:
        """Number of distinct article ids seen so far this session."""
        return len(self._seen)

    async def poll_once(self) -> list[NewsArticle]:
        """Fetch the current Toss dashboard snapshot and return only new articles.

        Raises:
            TossNewsAPIError: If the underlying scraper call raises. Per-feed
                errors inside ``TossScraper.scrape_news()`` are already
                caught and logged by the library itself (it degrades
                gracefully per feed type) -- this only fires for failures
                that escape that internal handling, e.g. an unexpected
                exception constructing the request.
        """
        try:
            raw_articles = await self._scraper.scrape_news()
        except Exception as exc:  # noqa: BLE001 -- wrap any failure shape uniformly
            logger.warning("Toss news poll failed", error=str(exc))
            raise TossNewsAPIError(
                message=f"Toss dashboard news request failed: {exc}",
                error_code=error_codes.TOSS_NEWS_API_ERROR,
            ) from exc

        new_articles = self._seen.filter_new(raw_articles)

        logger.info(
            "Toss news poll complete",
            fetched=len(raw_articles),
            new=len(new_articles),
            seen_total=len(self._seen),
        )

        return [map_toss_article_to_domain(a) for a in new_articles]

    async def poll_forever(
        self,
        max_iterations: int | None = None,
    ) -> AsyncIterator[list[NewsArticle]]:
        """Poll on ``self.poll_interval``, yielding newly-seen articles each time.

        Args:
            max_iterations: Stop after this many polls (mainly for tests).
                ``None`` polls indefinitely until the caller stops iterating
                or an exception propagates.

        Yields:
            The list of new (not previously seen) domain articles for each poll.
        """
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            yield await self.poll_once()
            iteration += 1
            if max_iterations is None or iteration < max_iterations:
                await asyncio.sleep(self.poll_interval)


# --- Article detail (full body) ---------------------------------------------

#: Toss's article-modal endpoint (owned by ``krx_news_client`` since 0.4.0).
#: Unlike the dashboard feed (~300-char summaries), it returns the full body
#: as typed blocks plus the publisher's original URL, Toss's own sentiment
#: label, and all related stock codes -- the same shape for every publisher.
TOSS_NEWS_DETAIL_URL = NEWS_DETAIL_URL

#: Toss newsId (e.g. "yna_AKR20260913056700017") from an article URL, or None.
news_id_from_article_url = news_id_from_url


async def fetch_article_detail(client: httpx.AsyncClient, url: str) -> dict | None:
    """Raw Toss article detail payload for an article URL, or None if unavailable.

    Deliberately returns the *raw* ``result.kr`` dict (plus ``news_id``) rather
    than ``TossScraper.fetch_article_detail``'s parsed ``NewsArticle``: the
    live poller archives this payload under ``data/toss_detail/`` in the same
    shape as the eval cache (typed body blocks included), which the parsed
    model does not preserve. It also keeps the live path free of the
    scraper's request throttle and retry sleeps. URL/id parsing and the
    endpoint come from the library.
    """
    news_id = news_id_from_url(url)
    if news_id is None:
        return None
    try:
        resp = await client.get(NEWS_DETAIL_URL.format(news_id=news_id))
    except httpx.HTTPError as exc:
        logger.warning("Toss news detail request failed", news_id=news_id, error=str(exc))
        return None
    if not resp.is_success:
        return None
    detail = ((resp.json() or {}).get("result") or {}).get("kr")
    return {"news_id": news_id, **detail} if detail else None


def extract_full_body(detail: dict, max_chars: int = 3000) -> str:
    """Article text from a detail payload: body blocks only, HTML tags stripped."""
    return extract_body_text(detail.get("content"))[:max_chars]
