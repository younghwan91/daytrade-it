"""Database setup and session management for async SQLAlchemy."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from gpt_quant_v2.infrastructure.persistence.models import Base

# Default database URL (can be overridden via settings)
DEFAULT_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/gpt_quant"


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(
        self,
        database_url: str = DEFAULT_DATABASE_URL,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
    ) -> None:
        """Initialize database manager.

        Args:
            database_url: Async database connection URL
            echo: Whether to echo SQL statements
            pool_size: Connection pool size
            max_overflow: Max connections above pool_size
        """
        self._database_url = database_url
        self._engine: AsyncEngine = create_async_engine(
            database_url,
            echo=echo,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
        )
        self._session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    @property
    def engine(self) -> AsyncEngine:
        """Get the async engine."""
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get the session factory."""
        return self._session_factory

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session.

        Yields:
            AsyncSession: Database session for executing queries

        Example:
            async with db_manager.get_session() as session:
                result = await session.execute(select(User))
        """
        session = self._session_factory()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def init_db(self) -> None:
        """Initialize database by creating all tables.

        Creates all tables defined in the ORM models if they don't exist.
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_db(self) -> None:
        """Drop all tables in the database.

        WARNING: This will delete all data. Use with caution.
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def close(self) -> None:
        """Close the database engine and all connections."""
        await self._engine.dispose()


# Global database manager instance (initialized lazily)
_db_manager: DatabaseManager | None = None


def get_database_manager(
    database_url: str = DEFAULT_DATABASE_URL,
    echo: bool = False,
) -> DatabaseManager:
    """Get or create the global database manager.

    Args:
        database_url: Database connection URL
        echo: Whether to echo SQL statements

    Returns:
        DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(database_url=database_url, echo=echo)
    return _db_manager


@asynccontextmanager
async def get_session(
    database_url: str = DEFAULT_DATABASE_URL,
) -> AsyncGenerator[AsyncSession, None]:
    """Convenience function to get a database session.

    Args:
        database_url: Database connection URL

    Yields:
        AsyncSession: Database session

    Example:
        async with get_session() as session:
            result = await session.execute(select(User))
    """
    db_manager = get_database_manager(database_url)
    async with db_manager.get_session() as session:
        yield session


async def init_db(database_url: str = DEFAULT_DATABASE_URL) -> None:
    """Initialize the database by creating all tables.

    Args:
        database_url: Database connection URL
    """
    db_manager = get_database_manager(database_url)
    await db_manager.init_db()
