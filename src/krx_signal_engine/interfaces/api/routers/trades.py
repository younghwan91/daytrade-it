"""Trades API router."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from krx_signal_engine.domain.entities.signal import SignalDirection
from krx_signal_engine.domain.entities.trade import TradeStatus
from krx_signal_engine.interfaces.api.dependencies import (
    CorrelationIdDep,
    CurrentUserDep,
    DbSessionDep,
)
from krx_signal_engine.interfaces.api.schemas import PaginatedResponse

router = APIRouter()


# ============================================================================
# Request/Response Schemas
# ============================================================================


class TradeResponse(BaseModel):
    """Trade response schema."""

    id: UUID
    signal_id: UUID | None = None
    ticker: str
    direction: SignalDirection
    quantity: Decimal
    entry_price: Decimal | None = None
    exit_price: Decimal | None = None
    status: TradeStatus
    pnl: Decimal | None = None
    fees: Decimal = Decimal("0")
    filled_at: datetime | None = None
    closed_at: datetime | None = None
    notional_value: Decimal | None = None
    return_pct: Decimal | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TradeCreateRequest(BaseModel):
    """Request to execute a trade from a signal."""

    signal_id: UUID = Field(description="ID of the signal to execute")
    quantity: Decimal = Field(gt=0, description="Number of shares to trade")
    limit_price: Decimal | None = Field(
        default=None,
        gt=0,
        description="Limit price (optional, market order if not specified)",
    )


class TradeManualCreateRequest(BaseModel):
    """Request to create a manual trade (without signal)."""

    ticker: str = Field(min_length=1, max_length=5, description="Stock ticker symbol")
    direction: SignalDirection = Field(description="BUY or SELL direction")
    quantity: Decimal = Field(gt=0, description="Number of shares to trade")
    limit_price: Decimal | None = Field(
        default=None,
        gt=0,
        description="Limit price (optional)",
    )


class TradeSummary(BaseModel):
    """P&L summary response schema."""

    total_trades: int = Field(description="Total number of trades")
    open_trades: int = Field(description="Number of open trades")
    closed_trades: int = Field(description="Number of closed trades")
    winning_trades: int = Field(description="Number of winning trades")
    losing_trades: int = Field(description="Number of losing trades")
    total_pnl: Decimal = Field(description="Total realized P&L")
    total_fees: Decimal = Field(description="Total trading fees")
    net_pnl: Decimal = Field(description="Net P&L after fees")
    win_rate: float = Field(ge=0.0, le=1.0, description="Win rate percentage")
    average_win: Decimal | None = Field(
        default=None,
        description="Average winning trade P&L",
    )
    average_loss: Decimal | None = Field(
        default=None,
        description="Average losing trade P&L",
    )
    profit_factor: Decimal | None = Field(
        default=None,
        description="Gross profit / Gross loss",
    )
    period_start: datetime | None = Field(
        default=None,
        description="Start of summary period",
    )
    period_end: datetime | None = Field(
        default=None,
        description="End of summary period",
    )


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "",
    response_model=PaginatedResponse[TradeResponse],
    summary="List trades",
    description="Retrieve a paginated list of trades with optional filters.",
)
async def list_trades(
    db: DbSessionDep,
    current_user: CurrentUserDep,
    ticker: str | None = Query(
        default=None,
        description="Filter by stock ticker",
        min_length=1,
        max_length=5,
    ),
    status: TradeStatus | None = Query(
        default=None,
        description="Filter by trade status",
    ),
    date_from: datetime | None = Query(
        default=None,
        description="Filter trades from this date",
    ),
    date_to: datetime | None = Query(
        default=None,
        description="Filter trades until this date",
    ),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
) -> PaginatedResponse[TradeResponse]:
    """List trades with optional filtering.

    Filters:
    - ticker: Filter by stock ticker symbol
    - status: Filter by trade status
    - date_from/date_to: Filter by date range
    """
    # TODO: Implement actual database query
    return PaginatedResponse.create(
        items=[],
        total=0,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/summary",
    response_model=TradeSummary,
    summary="Get P&L summary",
    description="Get profit/loss summary and trading statistics.",
)
async def get_trade_summary(
    db: DbSessionDep,
    current_user: CurrentUserDep,
    date_from: datetime | None = Query(
        default=None,
        description="Summary period start date",
    ),
    date_to: datetime | None = Query(
        default=None,
        description="Summary period end date",
    ),
    ticker: str | None = Query(
        default=None,
        description="Filter summary by ticker",
        min_length=1,
        max_length=5,
    ),
) -> TradeSummary:
    """Get P&L summary and trading statistics.

    Returns aggregate statistics including:
    - Total, open, and closed trade counts
    - Win/loss counts and win rate
    - Total and net P&L
    - Average win/loss amounts
    - Profit factor
    """
    # TODO: Implement actual database aggregation
    return TradeSummary(
        total_trades=0,
        open_trades=0,
        closed_trades=0,
        winning_trades=0,
        losing_trades=0,
        total_pnl=Decimal("0"),
        total_fees=Decimal("0"),
        net_pnl=Decimal("0"),
        win_rate=0.0,
        average_win=None,
        average_loss=None,
        profit_factor=None,
        period_start=date_from,
        period_end=date_to,
    )


@router.get(
    "/{trade_id}",
    response_model=TradeResponse,
    summary="Get trade by ID",
    description="Retrieve a specific trade by its unique identifier.",
)
async def get_trade(
    trade_id: UUID,
    db: DbSessionDep,
    current_user: CurrentUserDep,
) -> TradeResponse:
    """Get a specific trade by ID."""
    # TODO: Implement actual database lookup
    raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")


@router.post(
    "",
    response_model=TradeResponse,
    status_code=201,
    summary="Execute trade from signal",
    description="Execute a new trade based on a trading signal.",
)
async def create_trade(
    request: TradeCreateRequest,
    db: DbSessionDep,
    current_user: CurrentUserDep,
    correlation_id: CorrelationIdDep,
) -> TradeResponse:
    """Execute a trade based on a trading signal.

    This creates a new trade linked to the specified signal
    and submits it for execution.
    """

    # TODO: Lookup the signal to get ticker and direction
    # For now, we'll raise an error
    raise HTTPException(
        status_code=404,
        detail=f"Signal {request.signal_id} not found",
    )
