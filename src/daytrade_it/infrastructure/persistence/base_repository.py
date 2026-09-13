"""Generic base repository with CRUD operations."""

from abc import ABC, abstractmethod
from typing import Any, TypeVar
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from daytrade_it.infrastructure.persistence.models import Base

# Type variables for generic repository
TEntity = TypeVar("TEntity")
TModel = TypeVar("TModel", bound=Base)


class PaginationResult[TEntity]:
    """Paginated result container."""

    def __init__(
        self,
        items: list[TEntity],
        total: int,
        page: int,
        page_size: int,
    ) -> None:
        self.items = items
        self.total = total
        self.page = page
        self.page_size = page_size

    @property
    def total_pages(self) -> int:
        """Calculate total number of pages."""
        if self.page_size <= 0:
            return 0
        return (self.total + self.page_size - 1) // self.page_size

    @property
    def has_next(self) -> bool:
        """Check if there's a next page."""
        return self.page < self.total_pages

    @property
    def has_previous(self) -> bool:
        """Check if there's a previous page."""
        return self.page > 1


class BaseRepository[TEntity, TModel: Base](ABC):
    """Abstract base repository with CRUD operations.

    Provides generic CRUD operations for domain entities.

    Type Parameters:
        TEntity: Domain entity type
        TModel: SQLAlchemy ORM model type
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session
        """
        self._session = session

    @property
    @abstractmethod
    def _model_class(self) -> type[TModel]:
        """Get the ORM model class."""
        ...

    @abstractmethod
    def _to_entity(self, model: TModel) -> TEntity:
        """Convert ORM model to domain entity."""
        ...

    @abstractmethod
    def _from_entity(self, entity: TEntity) -> TModel:
        """Convert domain entity to ORM model."""
        ...

    async def get_by_id(self, entity_id: UUID) -> TEntity | None:
        """Get entity by ID.

        Args:
            entity_id: Entity UUID

        Returns:
            Entity if found, None otherwise
        """
        result = await self._session.get(self._model_class, entity_id)
        if result is None:
            return None
        return self._to_entity(result)

    async def get_all(
        self,
        page: int = 1,
        page_size: int = 50,
        order_by: str | None = None,
        order_desc: bool = False,
    ) -> PaginationResult[TEntity]:
        """Get all entities with pagination.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page
            order_by: Column name to order by
            order_desc: Whether to order descending

        Returns:
            Paginated result with entities
        """
        # Count total
        count_query = select(func.count()).select_from(self._model_class)
        total_result = await self._session.execute(count_query)
        total = total_result.scalar() or 0

        # Build query
        query = select(self._model_class)

        # Apply ordering
        if order_by:
            column = getattr(self._model_class, order_by, None)
            if column is not None:
                query = query.order_by(column.desc() if order_desc else column)
        else:
            # Default ordering by created_at if available
            if hasattr(self._model_class, "created_at"):
                query = query.order_by(self._model_class.created_at.desc())

        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)

        result = await self._session.execute(query)
        models = result.scalars().all()
        entities = [self._to_entity(m) for m in models]

        return PaginationResult(
            items=entities,
            total=total,
            page=page,
            page_size=page_size,
        )

    async def add(self, entity: TEntity) -> TEntity:
        """Add a new entity.

        Args:
            entity: Domain entity to add

        Returns:
            Added entity
        """
        model = self._from_entity(entity)
        self._session.add(model)
        await self._session.flush()
        return self._to_entity(model)

    async def add_many(self, entities: list[TEntity]) -> list[TEntity]:
        """Add multiple entities.

        Args:
            entities: List of domain entities to add

        Returns:
            List of added entities
        """
        models = [self._from_entity(e) for e in entities]
        self._session.add_all(models)
        await self._session.flush()
        return [self._to_entity(m) for m in models]

    async def update(self, entity: TEntity) -> TEntity:
        """Update an existing entity.

        Args:
            entity: Domain entity with updated data

        Returns:
            Updated entity
        """
        model = self._from_entity(entity)
        merged = await self._session.merge(model)
        await self._session.flush()
        return self._to_entity(merged)

    async def delete(self, entity_id: UUID) -> bool:
        """Delete an entity by ID.

        Args:
            entity_id: Entity UUID to delete

        Returns:
            True if deleted, False if not found
        """
        model = await self._session.get(self._model_class, entity_id)
        if model is None:
            return False
        await self._session.delete(model)
        await self._session.flush()
        return True

    async def exists(self, entity_id: UUID) -> bool:
        """Check if an entity exists by ID.

        Args:
            entity_id: Entity UUID

        Returns:
            True if exists, False otherwise
        """
        query = (
            select(func.count())
            .select_from(self._model_class)
            .where(self._model_class.id == entity_id)
        )
        result = await self._session.execute(query)
        count = result.scalar() or 0
        return count > 0

    async def count(self) -> int:
        """Count total number of entities.

        Returns:
            Total count
        """
        query = select(func.count()).select_from(self._model_class)
        result = await self._session.execute(query)
        return result.scalar() or 0

    async def _execute_query(
        self,
        query: Any,
        page: int | None = None,
        page_size: int | None = None,
    ) -> list[TEntity]:
        """Execute a query and return entities.

        Args:
            query: SQLAlchemy select query
            page: Optional page number for pagination
            page_size: Optional page size for pagination

        Returns:
            List of domain entities
        """
        if page is not None and page_size is not None:
            offset = (page - 1) * page_size
            query = query.offset(offset).limit(page_size)

        result = await self._session.execute(query)
        models = result.scalars().all()
        return [self._to_entity(m) for m in models]
