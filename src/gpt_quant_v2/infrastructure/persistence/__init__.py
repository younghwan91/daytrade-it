"""Database Persistence - Repositories, Models, and Unit of Work."""

from gpt_quant_v2.infrastructure.persistence.base_repository import (
    BaseRepository,
    PaginationResult,
)
from gpt_quant_v2.infrastructure.persistence.database import (
    DatabaseManager,
    get_database_manager,
    get_session,
    init_db,
)
from gpt_quant_v2.infrastructure.persistence.models import (
    Base,
    NewsArticleModel,
    PositionModel,
    PriceBarModel,
    TradingSignalModel,
    TradeModel,
)
from gpt_quant_v2.infrastructure.persistence.repositories import (
    NewsRepository,
    PositionRepository,
    PriceBarRepository,
    SignalRepository,
    TradeRepository,
)
from gpt_quant_v2.infrastructure.persistence.unit_of_work import (
    UnitOfWork,
    get_unit_of_work,
)

__all__ = [
    # Database
    "DatabaseManager",
    "get_database_manager",
    "get_session",
    "init_db",
    # Base
    "Base",
    "BaseRepository",
    "PaginationResult",
    # Models
    "NewsArticleModel",
    "PositionModel",
    "PriceBarModel",
    "TradingSignalModel",
    "TradeModel",
    # Repositories
    "NewsRepository",
    "PositionRepository",
    "PriceBarRepository",
    "SignalRepository",
    "TradeRepository",
    # Unit of Work
    "UnitOfWork",
    "get_unit_of_work",
]
