"""Specific repository implementations for domain entities."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import Integer, and_, func, select

from krx_signal_engine.domain.entities.news import NewsArticle
from krx_signal_engine.domain.entities.position import Position
from krx_signal_engine.domain.entities.signal import TradingSignal
from krx_signal_engine.domain.entities.trade import Trade, TradeStatus
from krx_signal_engine.domain.value_objects import Ticker
from krx_signal_engine.infrastructure.persistence.base_repository import BaseRepository
from krx_signal_engine.infrastructure.persistence.models import (
    NewsArticleModel,
    PositionModel,
    PriceBarModel,
    TradeModel,
    TradingSignalModel,
)


class NewsRepository(BaseRepository[NewsArticle, NewsArticleModel]):
    """Repository for NewsArticle entities."""

    @property
    def _model_class(self) -> type[NewsArticleModel]:
        return NewsArticleModel

    def _to_entity(self, model: NewsArticleModel) -> NewsArticle:
        return model.to_entity()

    def _from_entity(self, entity: NewsArticle) -> NewsArticleModel:
        return NewsArticleModel.from_entity(entity)

    async def find_by_ticker(
        self,
        ticker: Ticker | str,
        page: int = 1,
        page_size: int = 50,
    ) -> list[NewsArticle]:
        """Find news articles mentioning a specific ticker.

        Args:
            ticker: Stock ticker to search for
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            List of news articles mentioning the ticker
        """
        ticker_value = ticker.value if isinstance(ticker, Ticker) else ticker.upper()
        query = (
            select(NewsArticleModel)
            .where(NewsArticleModel.tickers.contains([ticker_value]))
            .order_by(NewsArticleModel.published_at.desc())
        )
        return await self._execute_query(query, page, page_size)

    async def find_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        page: int = 1,
        page_size: int = 50,
    ) -> list[NewsArticle]:
        """Find news articles within a date range.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            List of news articles in the date range
        """
        query = (
            select(NewsArticleModel)
            .where(
                and_(
                    NewsArticleModel.published_at >= start_date,
                    NewsArticleModel.published_at <= end_date,
                )
            )
            .order_by(NewsArticleModel.published_at.desc())
        )
        return await self._execute_query(query, page, page_size)

    async def find_by_source(
        self,
        source: str,
        page: int = 1,
        page_size: int = 50,
    ) -> list[NewsArticle]:
        """Find news articles from a specific source.

        Args:
            source: News source name
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            List of news articles from the source
        """
        query = (
            select(NewsArticleModel)
            .where(NewsArticleModel.source == source)
            .order_by(NewsArticleModel.published_at.desc())
        )
        return await self._execute_query(query, page, page_size)

    async def find_unanalyzed(
        self,
        page: int = 1,
        page_size: int = 50,
    ) -> list[NewsArticle]:
        """Find news articles that haven't been analyzed yet.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            List of unanalyzed news articles
        """
        query = (
            select(NewsArticleModel)
            .where(NewsArticleModel.sentiment_score.is_(None))
            .order_by(NewsArticleModel.published_at.desc())
        )
        return await self._execute_query(query, page, page_size)


class SignalRepository(BaseRepository[TradingSignal, TradingSignalModel]):
    """Repository for TradingSignal entities."""

    @property
    def _model_class(self) -> type[TradingSignalModel]:
        return TradingSignalModel

    def _to_entity(self, model: TradingSignalModel) -> TradingSignal:
        return model.to_entity()

    def _from_entity(self, entity: TradingSignal) -> TradingSignalModel:
        return TradingSignalModel.from_entity(entity)

    async def find_active(
        self,
        page: int = 1,
        page_size: int = 50,
    ) -> list[TradingSignal]:
        """Find active (non-expired) signals.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            List of active trading signals
        """
        now = datetime.now(UTC)
        query = (
            select(TradingSignalModel)
            .where(
                (TradingSignalModel.expires_at.is_(None)) | (TradingSignalModel.expires_at > now)
            )
            .order_by(TradingSignalModel.created_at.desc())
        )
        return await self._execute_query(query, page, page_size)

    async def find_by_ticker(
        self,
        ticker: Ticker | str,
        include_expired: bool = False,
        page: int = 1,
        page_size: int = 50,
    ) -> list[TradingSignal]:
        """Find signals for a specific ticker.

        Args:
            ticker: Stock ticker to search for
            include_expired: Whether to include expired signals
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            List of trading signals for the ticker
        """
        ticker_value = ticker.value if isinstance(ticker, Ticker) else ticker.upper()
        query = select(TradingSignalModel).where(TradingSignalModel.ticker == ticker_value)

        if not include_expired:
            now = datetime.now(UTC)
            query = query.where(
                (TradingSignalModel.expires_at.is_(None)) | (TradingSignalModel.expires_at > now)
            )

        query = query.order_by(TradingSignalModel.created_at.desc())
        return await self._execute_query(query, page, page_size)

    async def find_high_confidence(
        self,
        min_confidence: float = 0.7,
        page: int = 1,
        page_size: int = 50,
    ) -> list[TradingSignal]:
        """Find high confidence signals.

        Args:
            min_confidence: Minimum confidence threshold
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            List of high confidence trading signals
        """
        now = datetime.now(UTC)
        query = (
            select(TradingSignalModel)
            .where(
                and_(
                    TradingSignalModel.confidence >= min_confidence,
                    (TradingSignalModel.expires_at.is_(None))
                    | (TradingSignalModel.expires_at > now),
                )
            )
            .order_by(TradingSignalModel.confidence.desc())
        )
        return await self._execute_query(query, page, page_size)

    async def find_by_news_article(
        self,
        news_article_id: UUID,
    ) -> list[TradingSignal]:
        """Find signals generated from a specific news article.

        Args:
            news_article_id: News article UUID

        Returns:
            List of trading signals from the article
        """
        query = (
            select(TradingSignalModel)
            .where(TradingSignalModel.news_article_id == news_article_id)
            .order_by(TradingSignalModel.created_at.desc())
        )
        return await self._execute_query(query)


class TradeRepository(BaseRepository[Trade, TradeModel]):
    """Repository for Trade entities."""

    @property
    def _model_class(self) -> type[TradeModel]:
        return TradeModel

    def _to_entity(self, model: TradeModel) -> Trade:
        return model.to_entity()

    def _from_entity(self, entity: Trade) -> TradeModel:
        return TradeModel.from_entity(entity)

    async def find_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        status: TradeStatus | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> list[Trade]:
        """Find trades within a date range.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            status: Optional filter by trade status
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            List of trades in the date range
        """
        query = select(TradeModel).where(
            and_(
                TradeModel.created_at >= start_date,
                TradeModel.created_at <= end_date,
            )
        )

        if status is not None:
            query = query.where(TradeModel.status == status.value)

        query = query.order_by(TradeModel.created_at.desc())
        return await self._execute_query(query, page, page_size)

    async def find_open_trades(
        self,
        page: int = 1,
        page_size: int = 50,
    ) -> list[Trade]:
        """Find all open trades (filled but not closed).

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            List of open trades
        """
        query = (
            select(TradeModel)
            .where(
                and_(
                    TradeModel.status == TradeStatus.FILLED.value,
                    TradeModel.exit_price.is_(None),
                )
            )
            .order_by(TradeModel.filled_at.desc())
        )
        return await self._execute_query(query, page, page_size)

    async def find_by_ticker(
        self,
        ticker: Ticker | str,
        page: int = 1,
        page_size: int = 50,
    ) -> list[Trade]:
        """Find trades for a specific ticker.

        Args:
            ticker: Stock ticker to search for
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            List of trades for the ticker
        """
        ticker_value = ticker.value if isinstance(ticker, Ticker) else ticker.upper()
        query = (
            select(TradeModel)
            .where(TradeModel.ticker == ticker_value)
            .order_by(TradeModel.created_at.desc())
        )
        return await self._execute_query(query, page, page_size)

    async def find_by_signal(self, signal_id: UUID) -> list[Trade]:
        """Find trades generated from a specific signal.

        Args:
            signal_id: Signal UUID

        Returns:
            List of trades from the signal
        """
        query = (
            select(TradeModel)
            .where(TradeModel.signal_id == signal_id)
            .order_by(TradeModel.created_at.desc())
        )
        return await self._execute_query(query)

    async def calculate_pnl(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        ticker: Ticker | str | None = None,
    ) -> dict[str, Decimal]:
        """Calculate PnL statistics for closed trades.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            ticker: Optional ticker filter

        Returns:
            Dictionary with PnL statistics:
            - total_pnl: Total realized PnL
            - total_fees: Total trading fees
            - winning_trades: Number of profitable trades
            - losing_trades: Number of losing trades
            - total_trades: Total number of closed trades
        """
        query = select(
            func.sum(TradeModel.pnl).label("total_pnl"),
            func.sum(TradeModel.fees).label("total_fees"),
            func.count(TradeModel.id).label("total_trades"),
            func.sum(func.cast(TradeModel.pnl > 0, Integer)).label("winning_trades"),
            func.sum(func.cast(TradeModel.pnl < 0, Integer)).label("losing_trades"),
        ).where(TradeModel.pnl.is_not(None))

        if start_date is not None:
            query = query.where(TradeModel.closed_at >= start_date)
        if end_date is not None:
            query = query.where(TradeModel.closed_at <= end_date)
        if ticker is not None:
            ticker_value = ticker.value if isinstance(ticker, Ticker) else ticker.upper()
            query = query.where(TradeModel.ticker == ticker_value)

        result = await self._session.execute(query)
        row = result.one()

        return {
            "total_pnl": row.total_pnl or Decimal("0"),
            "total_fees": row.total_fees or Decimal("0"),
            "winning_trades": row.winning_trades or 0,
            "losing_trades": row.losing_trades or 0,
            "total_trades": row.total_trades or 0,
        }


class PositionRepository(BaseRepository[Position, PositionModel]):
    """Repository for Position entities."""

    @property
    def _model_class(self) -> type[PositionModel]:
        return PositionModel

    def _to_entity(self, model: PositionModel) -> Position:
        return model.to_entity()

    def _from_entity(self, entity: Position) -> PositionModel:
        return PositionModel.from_entity(entity)

    async def find_by_ticker(self, ticker: Ticker | str) -> Position | None:
        """Find position for a specific ticker.

        Args:
            ticker: Stock ticker

        Returns:
            Position if found, None otherwise
        """
        ticker_value = ticker.value if isinstance(ticker, Ticker) else ticker.upper()
        query = select(PositionModel).where(PositionModel.ticker == ticker_value)
        result = await self._session.execute(query)
        model = result.scalar_one_or_none()
        return model.to_entity() if model else None

    async def find_open_positions(
        self,
        page: int = 1,
        page_size: int = 50,
    ) -> list[Position]:
        """Find all open (non-zero) positions.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            List of open positions
        """
        query = (
            select(PositionModel)
            .where(PositionModel.quantity != 0)
            .order_by(PositionModel.opened_at.desc())
        )
        return await self._execute_query(query, page, page_size)

    async def find_long_positions(
        self,
        page: int = 1,
        page_size: int = 50,
    ) -> list[Position]:
        """Find all long positions.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            List of long positions
        """
        query = (
            select(PositionModel)
            .where(PositionModel.quantity > 0)
            .order_by(PositionModel.opened_at.desc())
        )
        return await self._execute_query(query, page, page_size)

    async def find_short_positions(
        self,
        page: int = 1,
        page_size: int = 50,
    ) -> list[Position]:
        """Find all short positions.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            List of short positions
        """
        query = (
            select(PositionModel)
            .where(PositionModel.quantity < 0)
            .order_by(PositionModel.opened_at.desc())
        )
        return await self._execute_query(query, page, page_size)

    async def get_total_exposure(self) -> dict[str, Decimal]:
        """Calculate total portfolio exposure.

        Returns:
            Dictionary with exposure metrics:
            - total_long: Total long exposure (market value)
            - total_short: Total short exposure (absolute market value)
            - net_exposure: Net exposure (long - short)
            - gross_exposure: Gross exposure (long + short)
        """
        # Get all positions with current prices
        positions = await self.find_open_positions(page_size=1000)

        total_long = Decimal("0")
        total_short = Decimal("0")

        for pos in positions:
            market_value = pos.market_value
            if market_value is None:
                continue

            if pos.is_long:
                total_long += market_value
            else:
                total_short += abs(market_value)

        return {
            "total_long": total_long,
            "total_short": total_short,
            "net_exposure": total_long - total_short,
            "gross_exposure": total_long + total_short,
        }


class PriceBarRepository:
    """Repository for PriceBar time-series data."""

    def __init__(self, session: Any) -> None:
        """Initialize repository with database session."""
        self._session = session

    async def get_bars(
        self,
        ticker: Ticker | str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> list[dict[str, Any]]:
        """Get price bars for a ticker within a date range.

        Args:
            ticker: Stock ticker
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            timeframe: Bar timeframe (e.g., "1m", "5m", "1h", "1d")

        Returns:
            List of price bar dictionaries
        """
        ticker_value = ticker.value if isinstance(ticker, Ticker) else ticker.upper()
        query = (
            select(PriceBarModel)
            .where(
                and_(
                    PriceBarModel.ticker == ticker_value,
                    PriceBarModel.timeframe == timeframe,
                    PriceBarModel.timestamp >= start,
                    PriceBarModel.timestamp <= end,
                )
            )
            .order_by(PriceBarModel.timestamp)
        )
        result = await self._session.execute(query)
        models = result.scalars().all()
        return [m.to_dict() for m in models]

    async def get_latest_bar(
        self,
        ticker: Ticker | str,
        timeframe: str = "1d",
    ) -> dict[str, Any] | None:
        """Get the most recent price bar for a ticker.

        Args:
            ticker: Stock ticker
            timeframe: Bar timeframe

        Returns:
            Latest price bar dictionary or None
        """
        ticker_value = ticker.value if isinstance(ticker, Ticker) else ticker.upper()
        query = (
            select(PriceBarModel)
            .where(
                and_(
                    PriceBarModel.ticker == ticker_value,
                    PriceBarModel.timeframe == timeframe,
                )
            )
            .order_by(PriceBarModel.timestamp.desc())
            .limit(1)
        )
        result = await self._session.execute(query)
        model = result.scalar_one_or_none()
        return model.to_dict() if model else None

    async def add_bar(self, bar_data: dict[str, Any]) -> dict[str, Any]:
        """Add a single price bar.

        Args:
            bar_data: Price bar data dictionary

        Returns:
            Added price bar dictionary
        """
        model = PriceBarModel.from_dict(bar_data)
        self._session.add(model)
        await self._session.flush()
        return model.to_dict()

    async def add_bars(self, bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add multiple price bars.

        Args:
            bars: List of price bar data dictionaries

        Returns:
            List of added price bar dictionaries
        """
        models = [PriceBarModel.from_dict(b) for b in bars]
        self._session.add_all(models)
        await self._session.flush()
        return [m.to_dict() for m in models]

    async def get_bars_count(
        self,
        ticker: Ticker | str,
        timeframe: str = "1d",
    ) -> int:
        """Get count of bars for a ticker.

        Args:
            ticker: Stock ticker
            timeframe: Bar timeframe

        Returns:
            Number of bars
        """
        ticker_value = ticker.value if isinstance(ticker, Ticker) else ticker.upper()
        query = (
            select(func.count())
            .select_from(PriceBarModel)
            .where(
                and_(
                    PriceBarModel.ticker == ticker_value,
                    PriceBarModel.timeframe == timeframe,
                )
            )
        )
        result = await self._session.execute(query)
        return result.scalar() or 0

    async def delete_bars(
        self,
        ticker: Ticker | str,
        start: datetime | None = None,
        end: datetime | None = None,
        timeframe: str = "1d",
    ) -> int:
        """Delete price bars for a ticker.

        Args:
            ticker: Stock ticker
            start: Optional start datetime
            end: Optional end datetime
            timeframe: Bar timeframe

        Returns:
            Number of deleted bars
        """
        from sqlalchemy import delete

        ticker_value = ticker.value if isinstance(ticker, Ticker) else ticker.upper()
        stmt = delete(PriceBarModel).where(
            and_(
                PriceBarModel.ticker == ticker_value,
                PriceBarModel.timeframe == timeframe,
            )
        )

        if start is not None:
            stmt = stmt.where(PriceBarModel.timestamp >= start)
        if end is not None:
            stmt = stmt.where(PriceBarModel.timestamp <= end)

        result = await self._session.execute(stmt)
        await self._session.flush()
        return result.rowcount
