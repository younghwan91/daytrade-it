"""
News Analysis Handler.

Orchestrates the news analysis pipeline: fetch news from Tiingo,
analyze sentiment with OpenAI, and store results.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from gpt_quant_v2.application.cqrs.bus import CommandBus
from gpt_quant_v2.domain.entities.news import NewsArticle, SentimentAnalysis
from gpt_quant_v2.domain.events.trading_events import NewsAnalyzedEvent
from gpt_quant_v2.domain.value_objects import Ticker

if TYPE_CHECKING:
    from gpt_quant_v2.infrastructure.external.openai_client import OpenAIClient
    from gpt_quant_v2.infrastructure.external.tiingo_client import TiingoClient
    from gpt_quant_v2.infrastructure.messaging.redis_pubsub import RedisPubSub
    from gpt_quant_v2.infrastructure.persistence.repositories import NewsRepository

logger = structlog.get_logger(__name__)


@dataclass
class AnalyzeNewsCommand:
    """Command to analyze news for a ticker."""

    ticker: str
    hours_back: int = 24
    correlation_id: str | None = None


@dataclass
class NewsAnalysisResult:
    """Result of news analysis."""

    ticker: str
    articles_fetched: int
    articles_analyzed: int
    average_sentiment: float | None
    sentiment_direction: str  # "bullish", "bearish", "neutral"
    confidence: float
    top_articles: list[dict]


class NewsAnalysisHandler:
    """
    Handles news analysis workflow.

    Pipeline:
    1. Fetch recent news from Tiingo API
    2. Filter for relevant articles
    3. Analyze sentiment using OpenAI
    4. Store results in database
    5. Publish NewsAnalyzedEvent
    """

    def __init__(
        self,
        tiingo_client: TiingoClient,
        openai_client: OpenAIClient,
        news_repository: NewsRepository,
        pubsub: RedisPubSub,
    ) -> None:
        """Initialize the handler with required dependencies."""
        self._tiingo = tiingo_client
        self._openai = openai_client
        self._news_repo = news_repository
        self._pubsub = pubsub

    async def handle(self, command: AnalyzeNewsCommand) -> NewsAnalysisResult:
        """
        Execute news analysis for a ticker.

        Args:
            command: The analysis command with ticker and time range.

        Returns:
            NewsAnalysisResult with analysis summary.
        """
        log = logger.bind(
            ticker=command.ticker,
            hours_back=command.hours_back,
            correlation_id=command.correlation_id,
        )
        log.info("Starting news analysis")

        ticker = Ticker(value=command.ticker.upper())

        # Step 1: Fetch news from Tiingo
        start_date = datetime.now(UTC) - timedelta(hours=command.hours_back)
        raw_articles = await self._tiingo.get_news(
            ticker=ticker.value,
            start_date=start_date,
        )
        log.info("Fetched news articles", count=len(raw_articles))

        if not raw_articles:
            return NewsAnalysisResult(
                ticker=ticker.value,
                articles_fetched=0,
                articles_analyzed=0,
                average_sentiment=None,
                sentiment_direction="neutral",
                confidence=0.0,
                top_articles=[],
            )

        # Step 2: Convert to domain entities and filter
        articles = []
        for raw in raw_articles:
            article = NewsArticle(
                ticker=ticker,
                title=raw.get("title", ""),
                content=raw.get("description", ""),
                source=raw.get("source", "unknown"),
                url=raw.get("url", ""),
                published_at=self._parse_date(raw.get("publishedDate")),
                raw_data=raw,
            )
            articles.append(article)

        # Step 3: Analyze sentiment for each article (with concurrency limit)
        analyzed_articles = await self._analyze_articles(articles, log)
        log.info("Analyzed articles", count=len(analyzed_articles))

        # Step 4: Calculate aggregate sentiment
        sentiments = [a.sentiment for a in analyzed_articles if a.sentiment]
        if sentiments:
            avg_score = sum(s.score for s in sentiments) / len(sentiments)
            avg_confidence = sum(s.confidence for s in sentiments) / len(sentiments)

            if avg_score > 0.2:
                direction = "bullish"
            elif avg_score < -0.2:
                direction = "bearish"
            else:
                direction = "neutral"
        else:
            avg_score = 0.0
            avg_confidence = 0.0
            direction = "neutral"

        # Step 5: Store in database
        for article in analyzed_articles:
            await self._news_repo.save(article)

        # Step 6: Publish event
        event = NewsAnalyzedEvent(
            ticker=ticker.value,
            article_count=len(analyzed_articles),
            average_sentiment=Decimal(str(round(avg_score, 4))),
            sentiment_direction=direction,
        )
        await self._pubsub.publish("news.analyzed", event.model_dump())

        log.info(
            "News analysis complete",
            direction=direction,
            avg_sentiment=avg_score,
            confidence=avg_confidence,
        )

        # Build top articles summary
        top_articles = sorted(
            [a for a in analyzed_articles if a.sentiment],
            key=lambda x: abs(x.sentiment.score) if x.sentiment else 0,
            reverse=True,
        )[:5]

        return NewsAnalysisResult(
            ticker=ticker.value,
            articles_fetched=len(raw_articles),
            articles_analyzed=len(analyzed_articles),
            average_sentiment=avg_score,
            sentiment_direction=direction,
            confidence=avg_confidence,
            top_articles=[
                {
                    "title": a.title,
                    "source": a.source,
                    "sentiment_score": a.sentiment.score if a.sentiment else None,
                    "sentiment_label": a.sentiment.label if a.sentiment else None,
                    "url": a.url,
                }
                for a in top_articles
            ],
        )

    async def _analyze_articles(
        self,
        articles: list[NewsArticle],
        log: structlog.BoundLogger,
    ) -> list[NewsArticle]:
        """Analyze sentiment for articles with concurrency limit."""
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent OpenAI calls

        async def analyze_one(article: NewsArticle) -> NewsArticle:
            async with semaphore:
                try:
                    result = await self._openai.analyze_sentiment(
                        text=f"{article.title}\n\n{article.content}",
                        context=f"Stock ticker: {article.ticker.value}",
                    )
                    article.sentiment = SentimentAnalysis(
                        score=result["score"],
                        label=result["label"],
                        confidence=result["confidence"],
                        reasoning=result.get("reasoning", ""),
                    )
                except Exception as e:
                    log.warning(
                        "Failed to analyze article",
                        article_id=str(article.id),
                        error=str(e),
                    )
            return article

        analyzed = await asyncio.gather(*[analyze_one(a) for a in articles])
        return list(analyzed)

    def _parse_date(self, date_str: str | None) -> datetime:
        """Parse date string from Tiingo format."""
        if not date_str:
            return datetime.now(UTC)

        # Try common formats
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return dt
            except ValueError:
                continue

        return datetime.now(UTC)


# Register with command bus
def register_news_handlers(bus: CommandBus, handler: NewsAnalysisHandler) -> None:
    """Register news analysis handlers with the command bus."""
    bus.register(AnalyzeNewsCommand, handler.handle)
