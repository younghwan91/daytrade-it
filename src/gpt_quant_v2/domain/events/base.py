"""
Base Domain Event

All domain events inherit from this base class.
Events are immutable records of something that happened in the domain.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DomainEvent(BaseModel):
    """
    Base class for all domain events.
    
    Domain events represent something significant that happened in the domain.
    They are immutable and contain all information needed to understand what happened.
    
    Attributes:
        event_id: Unique identifier for this event instance
        event_type: Name of the event type (auto-set to class name)
        occurred_at: Timestamp when the event occurred
        correlation_id: Optional ID to correlate related events
        causation_id: Optional ID of the event that caused this event
        metadata: Additional context about the event
    """
    
    event_id: UUID = Field(default_factory=uuid4)
    event_type: str = Field(default="")
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: UUID | None = None
    causation_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "frozen": True,  # Events are immutable
        "extra": "forbid",
    }
    
    def __init__(self, **data: Any) -> None:
        if "event_type" not in data or not data["event_type"]:
            data["event_type"] = self.__class__.__name__
        super().__init__(**data)
    
    def with_correlation(self, correlation_id: UUID) -> "DomainEvent":
        """Create a copy with the given correlation ID."""
        return self.model_copy(update={"correlation_id": correlation_id})
    
    def with_causation(self, causation_id: UUID) -> "DomainEvent":
        """Create a copy with the given causation ID."""
        return self.model_copy(update={"causation_id": causation_id})
    
    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        data = self.model_dump(mode="json")
        data["event_id"] = str(self.event_id)
        data["occurred_at"] = self.occurred_at.isoformat()
        if self.correlation_id:
            data["correlation_id"] = str(self.correlation_id)
        if self.causation_id:
            data["causation_id"] = str(self.causation_id)
        return data
