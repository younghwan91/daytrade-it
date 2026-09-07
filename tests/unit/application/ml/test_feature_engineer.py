"""Unit tests for Korean-language feature keywords and KRX market hours
(Task 12 / Phase 4a cont'd).

Covers the swap of `feature_engineer.py`'s English-language
`DEFAULT_KEYWORDS`/US-Eastern `MARKET_OPEN`/`MARKET_CLOSE` for Korean
financial-news vocabulary and KRX regular-session hours (09:00-15:30 KST).
"""

from datetime import UTC, datetime, time

from krx_signal_engine.application.ml.feature_engineer import (
    DEFAULT_KEYWORDS,
    MARKET_CLOSE,
    MARKET_OPEN,
    FeatureEngineer,
)


class _Article:
    def __init__(self, title: str, content: str, published_at: datetime) -> None:
        self.title = title
        self.content = content
        self.published_at = published_at


class _Sentiment:
    def __init__(self, score: float = 0.0, confidence: float = 0.5) -> None:
        self.sentiment_score = score
        self.confidence = confidence


class TestMarketHours:
    """MARKET_OPEN/MARKET_CLOSE must be KRX regular-session hours, not US Eastern."""

    def test_market_open_is_krx_hours(self) -> None:
        assert time(9, 0) == MARKET_OPEN

    def test_market_close_is_krx_hours(self) -> None:
        assert time(15, 30) == MARKET_CLOSE

    def test_market_hours_are_not_us_eastern(self) -> None:
        # Previously hardcoded to US Eastern (9:30/16:00) -- guard against
        # regression back to those values.
        assert time(9, 30) != MARKET_OPEN
        assert time(16, 0) != MARKET_CLOSE


class TestDefaultKeywords:
    """DEFAULT_KEYWORDS must be Korean financial/disclosure vocabulary."""

    def test_keywords_are_korean(self) -> None:
        assert DEFAULT_KEYWORDS, "keyword list must not be empty"
        for kw in DEFAULT_KEYWORDS:
            assert any("가" <= ch <= "힣" for ch in kw), f"keyword {kw!r} should contain Hangul"

    def test_no_stale_english_keywords(self) -> None:
        stale_english = {"earnings", "fda", "guidance", "buyback", "upgrade", "downgrade"}
        assert not (stale_english & set(DEFAULT_KEYWORDS))

    def test_expected_krx_vocabulary_present(self) -> None:
        expected = {"실적", "영업이익", "상한가", "하한가", "유상증자", "무상증자", "자사주매입"}
        assert expected.issubset(set(DEFAULT_KEYWORDS))


class TestKeywordExtractionWithKoreanText:
    """extract_news_features must count Korean keywords in Korean-language content."""

    def test_counts_korean_keywords_in_combined_text(self) -> None:
        engineer = FeatureEngineer()
        article = _Article(
            title="삼성전자 영업이익 실적 서프라이즈",
            content="삼성전자가 상한가를 기록했다. 유상증자 발표.",
            published_at=datetime(2026, 9, 7, 9, 30, tzinfo=UTC),
        )
        features = engineer.extract_news_features(article, _Sentiment())

        assert features["keyword_count_실적"] == 1.0
        assert features["keyword_count_영업이익"] == 1.0
        assert features["keyword_count_상한가"] == 1.0
        assert features["keyword_count_유상증자"] == 1.0
        # A keyword absent from the text counts zero.
        assert features["keyword_count_하한가"] == 0.0

    def test_time_since_market_open_uses_krx_hours(self) -> None:
        engineer = FeatureEngineer()
        # Published exactly at KRX open (09:00) -- naive interpretation
        # (module treats time-of-day components directly).
        article = _Article(
            title="",
            content="",
            published_at=datetime(2026, 9, 7, 9, 0, tzinfo=UTC),
        )
        features = engineer.extract_news_features(article, _Sentiment())
        assert features["time_since_market_open"] == 0.0
