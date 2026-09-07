"""Unit tests for TossNewsClient.

No network access — the underlying `TossScraper` is replaced by a fake that
returns canned `krx_news_client.NewsArticle` objects (or raises), per
Task 7's brief. Covers:
- Domain mapping correctness (`map_toss_article_to_domain`).
- Dedup-by-id across polls (a duplicate article id in a later poll must not
  re-emit).
- `poll_once` propagating scraper failures as `TossNewsAPIError`.
"""

from __future__ import annotations

import datetime as dt

import pytest
from krx_news_client import NewsArticle as KrxNewsArticle
from krx_news_client import NewsCategory, NewsSource

from gpt_quant_v2.infrastructure.external.toss_news_client import (
    TossNewsClient,
    map_toss_article_to_domain,
)
from gpt_quant_v2.shared.errors.exceptions import TossNewsAPIError


def make_krx_article(
    news_id: str = "1",
    title: str = "Test Article",
    tickers: list[str] | None = None,
    content: str = "some content",
    summary: str = "a summary",
    category: NewsCategory = NewsCategory.MARKET,
) -> KrxNewsArticle:
    url = f"https://tossinvest.com/news?id={news_id}"
    return KrxNewsArticle(
        id=f"toss:{news_id}",
        source=NewsSource.TOSS,
        category=category,
        title=title,
        url=url,
        content=content,
        summary=summary,
        tickers=tickers or ["005930"],
        author="Toss",
        published_at=dt.datetime(2026, 9, 7, 9, 0, tzinfo=dt.UTC),
    )


class FakeTossScraper:
    """Stands in for `krx_news_client.TossScraper` in tests."""

    def __init__(self, responses: list[list[KrxNewsArticle] | Exception]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.closed = False

    async def scrape_news(self) -> list[KrxNewsArticle]:
        self.calls += 1
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def close(self) -> None:
        self.closed = True


# =============================================================================
# Domain mapping
# =============================================================================


class TestMapTossArticleToDomain:
    def test_maps_core_fields(self) -> None:
        article = make_krx_article(news_id="42", title="Samsung rallies")
        domain = map_toss_article_to_domain(article)

        assert domain.title == "Samsung rallies"
        assert domain.content == "some content"
        assert domain.summary == "a summary"
        assert domain.source == "toss"
        assert str(domain.url).startswith("https://tossinvest.com/news")
        assert domain.published_at == article.published_at
        assert domain.category == "market"

    def test_maps_valid_6_digit_ticker(self) -> None:
        article = make_krx_article(tickers=["005930"])
        domain = map_toss_article_to_domain(article)

        assert len(domain.tickers) == 1
        assert domain.tickers[0].value == "005930"

    def test_skips_invalid_ticker_without_raising(self) -> None:
        article = make_krx_article(tickers=["NOTATICKER", "005930"])
        domain = map_toss_article_to_domain(article)

        assert [t.value for t in domain.tickers] == ["005930"]

    def test_same_source_article_maps_to_same_id_across_calls(self) -> None:
        """Deterministic UUID from the library's stable article id — same
        article maps to the same domain entity id across polls/restarts."""
        article = make_krx_article(news_id="99")
        first = map_toss_article_to_domain(article)
        second = map_toss_article_to_domain(article)
        assert first.id == second.id

    def test_empty_content_and_summary_falls_back_to_title(self) -> None:
        article = make_krx_article(title="Headline only", content="", summary="")
        domain = map_toss_article_to_domain(article)
        assert domain.content == "Headline only"
        assert len(domain.content) >= 1  # domain NewsArticle requires min_length=1


# =============================================================================
# Dedup logic
# =============================================================================


class TestTossNewsClientDedup:
    async def test_first_poll_returns_all_articles(self) -> None:
        articles = [make_krx_article(news_id=str(i)) for i in range(3)]
        scraper = FakeTossScraper(responses=[articles])
        client = TossNewsClient(scraper=scraper)

        result = await client.poll_once()

        assert len(result) == 3

    async def test_duplicate_ids_across_polls_are_not_re_emitted(self) -> None:
        first_batch = [make_krx_article(news_id=str(i)) for i in range(3)]
        # Second poll returns the same 3 (duplicates) plus 1 genuinely new.
        second_batch = [*first_batch, make_krx_article(news_id="new")]
        scraper = FakeTossScraper(responses=[first_batch, second_batch])
        client = TossNewsClient(scraper=scraper)

        first_result = await client.poll_once()
        second_result = await client.poll_once()

        assert len(first_result) == 3
        assert len(second_result) == 1
        assert second_result[0].title == make_krx_article(news_id="new").title

    async def test_seen_count_accumulates(self) -> None:
        first_batch = [make_krx_article(news_id=str(i)) for i in range(2)]
        second_batch = [make_krx_article(news_id=str(i)) for i in range(3)]
        scraper = FakeTossScraper(responses=[first_batch, second_batch])
        client = TossNewsClient(scraper=scraper)

        await client.poll_once()
        assert client.seen_count == 2
        await client.poll_once()
        assert client.seen_count == 3  # only 1 new id in second batch

    async def test_clear_seen_ids_allows_re_emission(self) -> None:
        batch = [make_krx_article(news_id="1")]
        scraper = FakeTossScraper(responses=[batch, batch])
        client = TossNewsClient(scraper=scraper)

        await client.poll_once()
        client.clear_seen_ids()
        second_result = await client.poll_once()

        assert len(second_result) == 1


# =============================================================================
# Failure propagation
# =============================================================================


class TestTossNewsClientFailures:
    async def test_scraper_exception_raises_toss_news_api_error(self) -> None:
        scraper = FakeTossScraper(responses=[RuntimeError("boom")])
        client = TossNewsClient(scraper=scraper)

        with pytest.raises(TossNewsAPIError):
            await client.poll_once()

    async def test_close_closes_underlying_scraper(self) -> None:
        scraper = FakeTossScraper(responses=[])
        client = TossNewsClient(scraper=scraper)

        await client.close()

        assert scraper.closed is True
