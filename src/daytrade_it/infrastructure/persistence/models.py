"""SQLAlchemy ORM models for database persistence."""

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import (
    JSON,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from daytrade_it.domain.entities.news import NewsArticle
from daytrade_it.domain.entities.position import Position
from daytrade_it.domain.entities.signal import (
    SignalDirection,
    SignalSource,
    TradingSignal,
)
from daytrade_it.domain.entities.trade import Trade, TradeStatus
from daytrade_it.domain.value_objects import Confidence, Sentiment, SentimentLabel, Ticker


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class NewsArticleModel(Base):
    """ORM model for NewsArticle entity."""

    __tablename__ = "news_articles"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    title: Mapped[str] = mapped_column(String(1000), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(200), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Stored as JSON array of ticker strings
    tickers: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    # Sentiment stored as JSON object
    sentiment_score: Mapped[float | None] = mapped_column(Numeric(4, 3), nullable=True)
    sentiment_label: Mapped[str | None] = mapped_column(String(20), nullable=True)
    sentiment_confidence: Mapped[float | None] = mapped_column(Numeric(4, 3), nullable=True)

    # Analysis result as JSON
    analysis_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_news_articles_published_at", "published_at"),
        Index("ix_news_articles_source", "source"),
        Index("ix_news_articles_tickers", "tickers", postgresql_using="gin"),
    )

    def to_entity(self) -> NewsArticle:
        """Convert ORM model to domain entity."""
        sentiment = None
        if (
            self.sentiment_score is not None
            and self.sentiment_label is not None
            and self.sentiment_confidence is not None
        ):
            sentiment = Sentiment(
                score=float(self.sentiment_score),
                label=SentimentLabel(self.sentiment_label),
                confidence=float(self.sentiment_confidence),
            )

        return NewsArticle(
            id=self.id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=self.version,
            title=self.title,
            content=self.content,
            summary=self.summary,
            source=self.source,
            url=self.url,  # type: ignore[arg-type]
            published_at=self.published_at,
            tickers=[Ticker(value=t) for t in self.tickers],
            sentiment=sentiment,
            analysis_result=self.analysis_result,
        )

    @classmethod
    def from_entity(cls, entity: NewsArticle) -> "NewsArticleModel":
        """Create ORM model from domain entity."""
        return cls(
            id=entity.id,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            version=entity.version,
            title=entity.title,
            content=entity.content,
            summary=entity.summary,
            source=entity.source,
            url=str(entity.url),
            published_at=entity.published_at,
            tickers=[t.value for t in entity.tickers],
            sentiment_score=entity.sentiment.score if entity.sentiment else None,
            sentiment_label=entity.sentiment.label.value if entity.sentiment else None,
            sentiment_confidence=entity.sentiment.confidence if entity.sentiment else None,
            analysis_result=entity.analysis_result,
        )


class TradingSignalModel(Base):
    """ORM model for TradingSignal entity."""

    __tablename__ = "trading_signals"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False)

    predicted_return: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)
    predicted_drawdown: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)
    risk_reward_ratio: Mapped[Decimal | None] = mapped_column(Numeric(8, 4), nullable=True)

    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    news_article_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)

    __table_args__ = (
        Index("ix_trading_signals_ticker", "ticker"),
        Index("ix_trading_signals_direction", "direction"),
        Index("ix_trading_signals_expires_at", "expires_at"),
        Index("ix_trading_signals_created_at", "created_at"),
    )

    def to_entity(self) -> TradingSignal:
        """Convert ORM model to domain entity."""
        return TradingSignal(
            id=self.id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=self.version,
            ticker=Ticker(value=self.ticker),
            direction=SignalDirection(self.direction),
            confidence=Confidence(value=float(self.confidence)),
            source=SignalSource(self.source),
            predicted_return=self.predicted_return,
            predicted_drawdown=self.predicted_drawdown,
            risk_reward_ratio=self.risk_reward_ratio,
            expires_at=self.expires_at,
            metadata=self.metadata_json,
            news_article_id=self.news_article_id,
        )

    @classmethod
    def from_entity(cls, entity: TradingSignal) -> "TradingSignalModel":
        """Create ORM model from domain entity."""
        return cls(
            id=entity.id,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            version=entity.version,
            ticker=entity.ticker.value,
            direction=entity.direction.value,
            confidence=entity.confidence.value,
            source=entity.source.value,
            predicted_return=entity.predicted_return,
            predicted_drawdown=entity.predicted_drawdown,
            risk_reward_ratio=entity.risk_reward_ratio,
            expires_at=entity.expires_at,
            metadata_json=entity.metadata,
            news_article_id=entity.news_article_id,
        )


class TradeModel(Base):
    """ORM model for Trade entity."""

    __tablename__ = "trades"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    signal_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    entry_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    pnl: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    fees: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False, default=Decimal("0"))
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_trades_ticker", "ticker"),
        Index("ix_trades_status", "status"),
        Index("ix_trades_signal_id", "signal_id"),
        Index("ix_trades_filled_at", "filled_at"),
        Index("ix_trades_closed_at", "closed_at"),
    )

    def to_entity(self) -> Trade:
        """Convert ORM model to domain entity."""
        return Trade(
            id=self.id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=self.version,
            signal_id=self.signal_id,
            ticker=Ticker(value=self.ticker),
            direction=SignalDirection(self.direction),
            quantity=self.quantity,
            entry_price=self.entry_price,
            exit_price=self.exit_price,
            status=TradeStatus(self.status),
            pnl=self.pnl,
            fees=self.fees,
            filled_at=self.filled_at,
            closed_at=self.closed_at,
        )

    @classmethod
    def from_entity(cls, entity: Trade) -> "TradeModel":
        """Create ORM model from domain entity."""
        return cls(
            id=entity.id,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            version=entity.version,
            signal_id=entity.signal_id,
            ticker=entity.ticker.value,
            direction=entity.direction.value,
            quantity=entity.quantity,
            entry_price=entity.entry_price,
            exit_price=entity.exit_price,
            status=entity.status.value,
            pnl=entity.pnl,
            fees=entity.fees,
            filled_at=entity.filled_at,
            closed_at=entity.closed_at,
        )


class PositionModel(Base):
    """ORM model for Position entity."""

    __tablename__ = "positions"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    average_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    current_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    unrealized_pnl: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_positions_ticker", "ticker"),
        UniqueConstraint("ticker", name="uq_positions_ticker"),
    )

    def to_entity(self) -> Position:
        """Convert ORM model to domain entity."""
        return Position(
            id=self.id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=self.version,
            ticker=Ticker(value=self.ticker),
            quantity=self.quantity,
            average_price=self.average_price,
            current_price=self.current_price,
            unrealized_pnl=self.unrealized_pnl,
            opened_at=self.opened_at,
        )

    @classmethod
    def from_entity(cls, entity: Position) -> "PositionModel":
        """Create ORM model from domain entity."""
        return cls(
            id=entity.id,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            version=entity.version,
            ticker=entity.ticker.value,
            quantity=entity.quantity,
            average_price=entity.average_price,
            current_price=entity.current_price,
            unrealized_pnl=entity.unrealized_pnl,
            opened_at=entity.opened_at,
        )


class PriceBarModel(Base):
    """ORM model for OHLCV price bars (time-series data)."""

    __tablename__ = "price_bars"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    open: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False)
    vwap: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)

    # Timeframe identifier (e.g., "1m", "5m", "1h", "1d")
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False, default="1d")

    __table_args__ = (
        Index("ix_price_bars_ticker_timestamp", "ticker", "timestamp"),
        Index("ix_price_bars_timestamp", "timestamp"),
        UniqueConstraint("ticker", "timestamp", "timeframe", name="uq_price_bars_ticker_ts_tf"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for easy consumption."""
        return {
            "id": self.id,
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "timeframe": self.timeframe,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PriceBarModel":
        """Create from dictionary."""
        from uuid import uuid4

        return cls(
            id=data.get("id", uuid4()),
            ticker=data["ticker"],
            timestamp=data["timestamp"],
            open=Decimal(str(data["open"])),
            high=Decimal(str(data["high"])),
            low=Decimal(str(data["low"])),
            close=Decimal(str(data["close"])),
            volume=int(data["volume"]),
            vwap=Decimal(str(data["vwap"])) if data.get("vwap") else None,
            timeframe=data.get("timeframe", "1d"),
        )
