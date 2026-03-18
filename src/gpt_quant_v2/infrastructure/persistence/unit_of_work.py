"""Unit of Work pattern for managing database transactions."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Self

from sqlalchemy.ext.asyncio import AsyncSession

from gpt_quant_v2.infrastructure.persistence.database import DatabaseManager
from gpt_quant_v2.infrastructure.persistence.repositories import (
    NewsRepository,
    PositionRepository,
    PriceBarRepository,
    SignalRepository,
    TradeRepository,
)


class UnitOfWork:
    """Unit of Work pattern for managing transaction scope.

    Provides a single point of control for database transactions and
    access to all repositories. Ensures that all changes within a
    unit of work are committed or rolled back together.

    Example:
        async with UnitOfWork(db_manager) as uow:
            article = await uow.news.get_by_id(article_id)
            article.set_sentiment(sentiment)
            await uow.news.update(article)
            await uow.commit()
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize Unit of Work.

        Args:
            db_manager: Database manager for session creation
        """
        self._db_manager = db_manager
        self._session: AsyncSession | None = None
        self._news: NewsRepository | None = None
        self._signals: SignalRepository | None = None
        self._trades: TradeRepository | None = None
        self._positions: PositionRepository | None = None
        self._price_bars: PriceBarRepository | None = None

    @property
    def session(self) -> AsyncSession:
        """Get the current session."""
        if self._session is None:
            raise RuntimeError("UnitOfWork not started. Use 'async with' context manager.")
        return self._session

    @property
    def news(self) -> NewsRepository:
        """Get the news repository."""
        if self._news is None:
            self._news = NewsRepository(self.session)
        return self._news

    @property
    def signals(self) -> SignalRepository:
        """Get the signals repository."""
        if self._signals is None:
            self._signals = SignalRepository(self.session)
        return self._signals

    @property
    def trades(self) -> TradeRepository:
        """Get the trades repository."""
        if self._trades is None:
            self._trades = TradeRepository(self.session)
        return self._trades

    @property
    def positions(self) -> PositionRepository:
        """Get the positions repository."""
        if self._positions is None:
            self._positions = PositionRepository(self.session)
        return self._positions

    @property
    def price_bars(self) -> PriceBarRepository:
        """Get the price bars repository."""
        if self._price_bars is None:
            self._price_bars = PriceBarRepository(self.session)
        return self._price_bars

    async def __aenter__(self) -> Self:
        """Enter the async context and start a session."""
        self._session = self._db_manager.session_factory()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the async context, rolling back on error."""
        if self._session is not None:
            if exc_type is not None:
                await self.rollback()
            await self._session.close()
            self._session = None
            self._clear_repositories()

    def _clear_repositories(self) -> None:
        """Clear all repository instances."""
        self._news = None
        self._signals = None
        self._trades = None
        self._positions = None
        self._price_bars = None

    async def commit(self) -> None:
        """Commit the current transaction.

        Commits all pending changes to the database.
        """
        if self._session is not None:
            await self._session.commit()

    async def rollback(self) -> None:
        """Rollback the current transaction.

        Discards all pending changes since the last commit.
        """
        if self._session is not None:
            await self._session.rollback()

    async def refresh(self, entity: object) -> None:
        """Refresh an entity from the database.

        Reloads the entity's data from the database, discarding
        any local changes.

        Args:
            entity: The ORM model instance to refresh
        """
        if self._session is not None:
            await self._session.refresh(entity)


@asynccontextmanager
async def get_unit_of_work(
    db_manager: DatabaseManager,
) -> AsyncGenerator[UnitOfWork, None]:
    """Get a Unit of Work context manager.

    Convenience function for creating a UnitOfWork.

    Args:
        db_manager: Database manager instance

    Yields:
        UnitOfWork instance

    Example:
        async with get_unit_of_work(db_manager) as uow:
            articles = await uow.news.find_by_ticker("AAPL")
            for article in articles:
                print(article.title)
    """
    async with UnitOfWork(db_manager) as uow:
        yield uow
