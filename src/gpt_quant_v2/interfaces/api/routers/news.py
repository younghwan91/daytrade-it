"""News API router."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, HttpUrl

from gpt_quant_v2.domain.value_objects.sentiment import SentimentLabel
from gpt_quant_v2.interfaces.api.dependencies import (
    CorrelationIdDep,
    CurrentUserDep,
    DbSessionDep,
)
from gpt_quant_v2.interfaces.api.schemas import PaginatedResponse

router = APIRouter()


# ============================================================================
# Request/Response Schemas
# ============================================================================


class SentimentResponse(BaseModel):
    """Sentiment analysis response schema."""

    score: float = Field(ge=-1.0, le=1.0, description="Sentiment score (-1 to 1)")
    label: SentimentLabel = Field(description="Sentiment label")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level")


class NewsArticleResponse(BaseModel):
    """News article response schema."""

    id: UUID
    title: str
    content: str
    summary: str | None = None
    source: str
    url: HttpUrl
    published_at: datetime
    tickers: list[str] = Field(default_factory=list)
    sentiment: SentimentResponse | None = None
    is_analyzed: bool
    analysis_result: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class NewsAnalyzeRequest(BaseModel):
    """Request to trigger news analysis for a ticker."""

    ticker: str = Field(
        min_length=1,
        max_length=5,
        description="Stock ticker to analyze news for",
    )
    hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Hours of news to analyze (1-168)",
    )
    force_reanalyze: bool = Field(
        default=False,
        description="Force re-analysis of already analyzed articles",
    )


class NewsAnalyzeResponse(BaseModel):
    """Response after triggering news analysis."""

    ticker: str
    articles_found: int
    articles_analyzed: int
    articles_skipped: int
    average_sentiment: SentimentResponse | None = None
    analysis_id: UUID | None = None
    message: str


class NewsListSummary(BaseModel):
    """Summary statistics for news list."""

    total_articles: int
    analyzed_count: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    average_sentiment_score: float | None = None


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "",
    response_model=PaginatedResponse[NewsArticleResponse],
    summary="Get analyzed news",
    description="Retrieve a paginated list of analyzed news articles with optional filters.",
)
async def list_news(
    db: DbSessionDep,
    current_user: CurrentUserDep,
    ticker: str | None = Query(
        default=None,
        description="Filter by stock ticker",
        min_length=1,
        max_length=5,
    ),
    hours: int = Query(
        default=24,
        ge=1,
        le=720,
        description="Hours of news to retrieve (1-720)",
    ),
    sentiment: SentimentLabel | None = Query(
        default=None,
        description="Filter by sentiment label",
    ),
    analyzed_only: bool = Query(
        default=False,
        description="Only return analyzed articles",
    ),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
) -> PaginatedResponse[NewsArticleResponse]:
    """List news articles with optional filtering.

    Filters:
    - ticker: Filter by stock ticker symbol mentioned
    - hours: Retrieve news from the last N hours
    - sentiment: Filter by sentiment label (BULLISH/BEARISH/NEUTRAL)
    - analyzed_only: Only return articles that have been analyzed
    """
    # TODO: Implement actual database query
    return PaginatedResponse.create(
        items=[],
        total=0,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{article_id}",
    response_model=NewsArticleResponse,
    summary="Get news article by ID",
    description="Retrieve a specific news article by its unique identifier.",
)
async def get_news_article(
    article_id: UUID,
    db: DbSessionDep,
    current_user: CurrentUserDep,
) -> NewsArticleResponse:
    """Get a specific news article by ID."""
    # TODO: Implement actual database lookup
    raise HTTPException(status_code=404, detail=f"News article {article_id} not found")


@router.post(
    "/analyze",
    response_model=NewsAnalyzeResponse,
    status_code=202,
    summary="Trigger news analysis",
    description="Trigger sentiment analysis for news articles related to a ticker.",
)
async def analyze_news(
    request: NewsAnalyzeRequest,
    db: DbSessionDep,
    current_user: CurrentUserDep,
    correlation_id: CorrelationIdDep,
) -> NewsAnalyzeResponse:
    """Trigger news analysis for a ticker.

    This endpoint initiates sentiment and impact analysis
    for recent news articles mentioning the specified ticker.

    The analysis runs asynchronously and updates the
    sentiment and analysis_result fields of each article.
    """
    ticker = request.ticker.upper()

    # Validate ticker format
    if not ticker.isalpha():
        raise HTTPException(
            status_code=400,
            detail="Invalid ticker format. Must be 1-5 uppercase letters.",
        )

    # TODO: Implement actual news fetching and analysis
    # This would typically:
    # 1. Fetch recent news for the ticker
    # 2. Queue analysis jobs or run synchronously
    # 3. Update articles with sentiment scores

    return NewsAnalyzeResponse(
        ticker=ticker,
        articles_found=0,
        articles_analyzed=0,
        articles_skipped=0,
        average_sentiment=None,
        analysis_id=None,
        message=f"News analysis initiated for {ticker}. No articles found in the last {request.hours} hours.",
    )
