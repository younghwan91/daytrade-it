"""NewsArticle entity representing financial news content."""

from datetime import datetime
from typing import Any

from pydantic import Field, HttpUrl

from gpt_quant_v2.domain.entities.base import Entity
from gpt_quant_v2.domain.value_objects import Sentiment, Ticker


class NewsArticle(Entity):
    """Domain entity representing a financial news article.

    Attributes:
        title: Article headline
        content: Full article text
        summary: Brief summary of the article
        source: News source/publisher name
        url: Original article URL
        published_at: When the article was published
        tickers: List of stock tickers mentioned in the article
        sentiment: Analyzed sentiment (optional)
        analysis_result: Additional analysis data (optional)
    """

    title: str = Field(min_length=1, max_length=1000)
    content: str = Field(min_length=1)
    summary: str | None = Field(default=None, max_length=5000)
    source: str = Field(min_length=1, max_length=200)
    url: HttpUrl
    published_at: datetime
    tickers: list[Ticker] = Field(default_factory=list)
    sentiment: Sentiment | None = Field(default=None)
    analysis_result: dict[str, Any] | None = Field(default=None)

    def __str__(self) -> str:
        return f"NewsArticle({self.title[:50]}...)"

    def __repr__(self) -> str:
        return (
            f"NewsArticle(id={self.id!r}, title={self.title[:30]!r}..., "
            f"source={self.source!r}, tickers={self.tickers!r})"
        )

    @property
    def is_analyzed(self) -> bool:
        """Check if the article has been analyzed for sentiment."""
        return self.sentiment is not None

    @property
    def has_tickers(self) -> bool:
        """Check if the article mentions any stock tickers."""
        return len(self.tickers) > 0

    @property
    def ticker_count(self) -> int:
        """Get the number of tickers mentioned."""
        return len(self.tickers)

    def mentions_ticker(self, ticker: Ticker) -> bool:
        """Check if the article mentions a specific ticker."""
        return ticker in self.tickers

    def add_ticker(self, ticker: Ticker) -> None:
        """Add a ticker to the article if not already present."""
        if ticker not in self.tickers:
            self.tickers.append(ticker)
            self.touch()

    def set_sentiment(self, sentiment: Sentiment) -> None:
        """Set the sentiment analysis result."""
        self.sentiment = sentiment
        self.touch()

    def set_analysis_result(self, result: dict[str, Any]) -> None:
        """Set the analysis result dictionary."""
        self.analysis_result = result
        self.touch()
