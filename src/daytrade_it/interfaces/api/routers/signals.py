"""Trading signals API router."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from daytrade_it.domain.entities.signal import SignalDirection, SignalSource
from daytrade_it.interfaces.api.dependencies import (
    CorrelationIdDep,
    CurrentUserDep,
    DbSessionDep,
)
from daytrade_it.interfaces.api.schemas import PaginatedResponse

router = APIRouter()


# ============================================================================
# Request/Response Schemas
# ============================================================================


class SignalResponse(BaseModel):
    """Trading signal response schema."""

    id: UUID
    ticker: str
    direction: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0)
    source: SignalSource
    predicted_return: Decimal | None = None
    predicted_drawdown: Decimal | None = None
    risk_reward_ratio: Decimal | None = None
    expires_at: datetime | None = None
    is_expired: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
    news_article_id: UUID | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SignalCreateRequest(BaseModel):
    """Request to create a manual trading signal."""

    ticker: str = Field(min_length=1, max_length=5, description="Stock ticker symbol")
    direction: SignalDirection = Field(description="BUY or SELL direction")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level 0.0-1.0")
    source: SignalSource = Field(
        default=SignalSource.COMPOSITE,
        description="Signal source type",
    )
    predicted_return: Decimal | None = Field(
        default=None,
        description="Expected return percentage",
    )
    predicted_drawdown: Decimal | None = Field(
        default=None,
        le=0,
        description="Expected max drawdown (negative)",
    )
    risk_reward_ratio: Decimal | None = Field(
        default=None,
        gt=0,
        description="Risk/reward ratio",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Signal expiration timestamp",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional signal metadata",
    )
    news_article_id: UUID | None = Field(
        default=None,
        description="Related news article ID",
    )


class SignalDeleteResponse(BaseModel):
    """Response when deleting/cancelling a signal."""

    id: UUID
    deleted: bool
    message: str


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "",
    response_model=PaginatedResponse[SignalResponse],
    summary="List active trading signals",
    description="Retrieve a paginated list of active trading signals with optional filters.",
)
async def list_signals(
    db: DbSessionDep,
    current_user: CurrentUserDep,
    ticker: str | None = Query(
        default=None,
        description="Filter by stock ticker",
        min_length=1,
        max_length=5,
    ),
    min_confidence: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum confidence level",
    ),
    direction: SignalDirection | None = Query(
        default=None,
        description="Filter by signal direction (BUY/SELL)",
    ),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
) -> PaginatedResponse[SignalResponse]:
    """List active trading signals with optional filtering.

    Filters:
    - ticker: Filter by stock ticker symbol
    - min_confidence: Minimum confidence threshold
    - direction: Filter by BUY or SELL direction
    """
    # TODO: Implement actual database query
    # This is a placeholder that returns empty results
    return PaginatedResponse.create(
        items=[],
        total=0,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{signal_id}",
    response_model=SignalResponse,
    summary="Get signal by ID",
    description="Retrieve a specific trading signal by its unique identifier.",
)
async def get_signal(
    signal_id: UUID,
    db: DbSessionDep,
    current_user: CurrentUserDep,
) -> SignalResponse:
    """Get a specific trading signal by ID."""
    # TODO: Implement actual database lookup
    raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")


@router.post(
    "",
    response_model=SignalResponse,
    status_code=201,
    summary="Create manual signal",
    description="Create a new trading signal manually.",
)
async def create_signal(
    request: SignalCreateRequest,
    db: DbSessionDep,
    current_user: CurrentUserDep,
    correlation_id: CorrelationIdDep,
) -> SignalResponse:
    """Create a new manual trading signal.

    This allows users to manually create trading signals
    outside of automated analysis pipelines.
    """

    from daytrade_it.domain.entities.signal import TradingSignal
    from daytrade_it.domain.value_objects import Confidence, Ticker

    # Create the signal entity
    signal = TradingSignal(
        ticker=Ticker(value=request.ticker.upper()),
        direction=request.direction,
        confidence=Confidence(value=request.confidence),
        source=request.source,
        predicted_return=request.predicted_return,
        predicted_drawdown=request.predicted_drawdown,
        risk_reward_ratio=request.risk_reward_ratio,
        expires_at=request.expires_at,
        metadata=request.metadata,
        news_article_id=request.news_article_id,
    )

    # TODO: Persist to database
    # await db.add(signal)
    # await db.commit()

    return SignalResponse(
        id=signal.id,
        ticker=signal.ticker.value,
        direction=signal.direction,
        confidence=signal.confidence.value,
        source=signal.source,
        predicted_return=signal.predicted_return,
        predicted_drawdown=signal.predicted_drawdown,
        risk_reward_ratio=signal.risk_reward_ratio,
        expires_at=signal.expires_at,
        is_expired=signal.is_expired,
        metadata=signal.metadata,
        news_article_id=signal.news_article_id,
        created_at=signal.created_at,
        updated_at=signal.updated_at,
    )


@router.delete(
    "/{signal_id}",
    response_model=SignalDeleteResponse,
    summary="Delete/cancel signal",
    description="Delete or cancel a trading signal by its unique identifier.",
)
async def delete_signal(
    signal_id: UUID,
    db: DbSessionDep,
    current_user: CurrentUserDep,
) -> SignalDeleteResponse:
    """Delete or cancel a trading signal.

    This removes the signal from the active signal list
    and prevents any further trade execution based on it.
    """
    # TODO: Implement actual database deletion
    # For now, we'll just verify the signal exists (placeholder)
    raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")
