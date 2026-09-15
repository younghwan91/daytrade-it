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

from daytrade_it.infrastructure.external.toss_news_client import (
    TossNewsClient,
    map_toss_article_to_domain,
)
from daytrade_it.shared.errors.exceptions import TossNewsAPIError


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

    async def test_same_id_twice_in_one_snapshot_is_emitted_once(self) -> None:
        """Toss's three dashboard feeds overlap; one poll must not emit the
        same article (and so the same uuid5 domain id) twice."""
        article = make_krx_article(news_id="dup")
        scraper = FakeTossScraper(responses=[[article, make_krx_article(news_id="x"), article]])
        client = TossNewsClient(scraper=scraper)

        result = await client.poll_once()

        assert len(result) == 2
        assert len({a.id for a in result}) == 2
        assert client.seen_count == 2


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


def test_news_id_from_article_url() -> None:
    from daytrade_it.infrastructure.external.toss_news_client import news_id_from_article_url

    url = (
        "https://tossinvest.com/news?contentType=news&contentParams="
        "%7B%22id%22%3A%22yna_AKR20260913056700017%22%7D"
    )
    assert news_id_from_article_url(url) == "yna_AKR20260913056700017"
    assert news_id_from_article_url("https://tossinvest.com/news") is None


def test_extract_full_body_keeps_text_blocks_and_strips_tags() -> None:
    from daytrade_it.infrastructure.external.toss_news_client import extract_full_body

    detail = {
        "content": [
            {"type": "summary", "content": "요약 문장"},
            {"type": "image", "content": "https://img/x.jpg", "caption": "사진"},
            {"type": "sub_headline", "content": " 소제목 "},
            {"type": "text", "content": "본문 <b>강조</b> 끝"},
            {"type": "text", "content": ""},
        ]
    }
    assert extract_full_body(detail) == "요약 문장\n소제목\n본문 강조 끝"
    assert extract_full_body(detail, max_chars=4) == "요약 문장"[:4]


def test_extract_full_body_drops_tag_only_blocks() -> None:
    from daytrade_it.infrastructure.external.toss_news_client import extract_full_body

    detail = {
        "content": [
            {"type": "text", "content": "<br/>"},
            {"type": "text", "content": "<p> 첫 문단 </p>"},
            {"type": "text", "content": "둘째 문단"},
        ]
    }
    assert extract_full_body(detail) == "첫 문단\n둘째 문단"
    assert extract_full_body({}) == ""


def test_detail_helpers_delegate_to_library() -> None:
    from krx_news_client.scrapers import toss

    from daytrade_it.infrastructure.external import toss_news_client

    assert toss_news_client.news_id_from_article_url is toss.news_id_from_url
    assert toss_news_client.TOSS_NEWS_DETAIL_URL == toss.NEWS_DETAIL_URL


class TestFetchArticleDetail:
    URL = (
        "https://tossinvest.com/news?contentType=news&contentParams="
        "%7B%22id%22%3A%22yna_AKR20260913056700017%22%7D"
    )

    @staticmethod
    def _client(handler):  # type: ignore[no-untyped-def]
        import httpx

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def test_returns_raw_kr_payload_with_news_id(self) -> None:
        import httpx

        from daytrade_it.infrastructure.external.toss_news_client import fetch_article_detail

        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            blocks = [{"type": "text", "content": "본문"}]
            return httpx.Response(200, json={"result": {"kr": {"title": "t", "content": blocks}}})

        async with self._client(handler) as client:
            detail = await fetch_article_detail(client, self.URL)

        assert seen == ["https://wts-info-api.tossinvest.com/api/v2/news/yna_AKR20260913056700017"]
        assert detail == {
            "news_id": "yna_AKR20260913056700017",
            "title": "t",
            "content": [{"type": "text", "content": "본문"}],
        }

    async def test_missing_article_and_errors_return_none(self) -> None:
        import httpx

        from daytrade_it.infrastructure.external.toss_news_client import fetch_article_detail

        responses = iter(
            [httpx.Response(200, json={"result": None}), httpx.Response(404), httpx.Response(500)]
        )

        async with self._client(lambda _req: next(responses)) as client:
            assert await fetch_article_detail(client, self.URL) is None
            assert await fetch_article_detail(client, self.URL) is None
            assert await fetch_article_detail(client, self.URL) is None
            # No newsId in the URL: no request at all.
            assert await fetch_article_detail(client, "https://tossinvest.com/news") is None

    async def test_transport_error_returns_none(self) -> None:
        import httpx

        from daytrade_it.infrastructure.external.toss_news_client import fetch_article_detail

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        async with self._client(handler) as client:
            assert await fetch_article_detail(client, self.URL) is None
