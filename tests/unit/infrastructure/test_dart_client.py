"""Unit tests for DartClient.

No network access — the underlying `DartScraper` is replaced by a fake, per
Task 7's brief. Covers domain mapping (including `disclosure_type` surfaced
for Phase 4, not classified here) and dedup-by-id across polls.
"""

from __future__ import annotations

import datetime as dt

import pytest
from krx_news_client import DartQuotaExceededError, NewsSource
from krx_news_client import Disclosure as KrxDisclosure

from daytrade_it.infrastructure.external.dart_client import (
    DartClient,
    map_dart_disclosure_to_domain,
)
from daytrade_it.shared.errors.exceptions import DartAPIError


def make_krx_disclosure(
    rcept_no: str = "1",
    company: str = "삼성전자",
    ticker: str = "005930",
    disclosure_type: str = "주요사항보고서",
    title: str = "주요사항보고서(자기주식취득결정)",
) -> KrxDisclosure:
    url = f"https://dart.fss.or.kr/dsaf001/main.do?rcept_no={rcept_no}"
    return KrxDisclosure(
        id=f"dart:{rcept_no}",
        source=NewsSource.DART,
        title=title,
        url=url,
        company=company,
        ticker=ticker,
        disclosure_type=disclosure_type,
        published_at=dt.datetime(2026, 9, 7, 9, 0, tzinfo=dt.UTC),
    )


class FakeDartScraper:
    """Stands in for `krx_news_client.DartScraper` in tests."""

    def __init__(self, responses: list[list[KrxDisclosure] | Exception]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.closed = False

    async def scrape_disclosures(self) -> list[KrxDisclosure]:
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


class TestMapDartDisclosureToDomain:
    def test_maps_core_fields_and_surfaces_disclosure_type(self) -> None:
        disclosure = make_krx_disclosure(
            disclosure_type="주요사항보고서", title="주요사항보고서(자기주식취득결정)"
        )
        domain = map_dart_disclosure_to_domain(disclosure)

        assert domain.title == "주요사항보고서(자기주식취득결정)"
        assert domain.disclosure_type == "주요사항보고서"
        assert domain.source == "dart"
        assert domain.category is None  # DART maps disclosure_type, not category
        assert str(domain.url).startswith("https://dart.fss.or.kr")

    def test_maps_valid_ticker(self) -> None:
        disclosure = make_krx_disclosure(ticker="005930")
        domain = map_dart_disclosure_to_domain(disclosure)
        assert [t.value for t in domain.tickers] == ["005930"]

    def test_skips_invalid_ticker_without_raising(self) -> None:
        disclosure = make_krx_disclosure(ticker="")
        domain = map_dart_disclosure_to_domain(disclosure)
        assert domain.tickers == []

    def test_content_synthesized_non_empty(self) -> None:
        """Disclosure has no free-text content field — content must still
        satisfy the domain entity's min_length=1."""
        disclosure = make_krx_disclosure(company="삼성전자", disclosure_type="주요사항보고서")
        domain = map_dart_disclosure_to_domain(disclosure)
        assert len(domain.content) >= 1
        assert "삼성전자" in domain.content

    def test_same_disclosure_maps_to_same_id_across_calls(self) -> None:
        disclosure = make_krx_disclosure(rcept_no="777")
        first = map_dart_disclosure_to_domain(disclosure)
        second = map_dart_disclosure_to_domain(disclosure)
        assert first.id == second.id


# =============================================================================
# Dedup logic
# =============================================================================


class TestDartClientDedup:
    async def test_first_poll_returns_all_disclosures(self) -> None:
        disclosures = [make_krx_disclosure(rcept_no=str(i)) for i in range(3)]
        scraper = FakeDartScraper(responses=[disclosures])
        client = DartClient(api_key="fake-key", scraper=scraper)

        result = await client.poll_once()

        assert len(result) == 3

    async def test_duplicate_ids_across_polls_are_not_re_emitted(self) -> None:
        first_batch = [make_krx_disclosure(rcept_no=str(i)) for i in range(2)]
        second_batch = [*first_batch, make_krx_disclosure(rcept_no="new")]
        scraper = FakeDartScraper(responses=[first_batch, second_batch])
        client = DartClient(api_key="fake-key", scraper=scraper)

        first_result = await client.poll_once()
        second_result = await client.poll_once()

        assert len(first_result) == 2
        assert len(second_result) == 1


# =============================================================================
# Failure propagation
# =============================================================================


class TestDartClientFailures:
    async def test_scraper_exception_raises_dart_api_error(self) -> None:
        scraper = FakeDartScraper(responses=[RuntimeError("boom")])
        client = DartClient(api_key="fake-key", scraper=scraper)

        with pytest.raises(DartAPIError):
            await client.poll_once()

    async def test_quota_exceeded_propagates_unwrapped(self) -> None:
        """`DartQuotaExceededError` must propagate as-is (not wrapped in
        `DartAPIError`) so callers like the circuit breaker can distinguish
        quota exhaustion from a generic failure."""
        scraper = FakeDartScraper(responses=[DartQuotaExceededError("quota gone")])
        client = DartClient(api_key="fake-key", scraper=scraper)

        with pytest.raises(DartQuotaExceededError):
            await client.poll_once()

    async def test_close_closes_underlying_scraper(self) -> None:
        scraper = FakeDartScraper(responses=[])
        client = DartClient(api_key="fake-key", scraper=scraper)

        await client.close()

        assert scraper.closed is True
