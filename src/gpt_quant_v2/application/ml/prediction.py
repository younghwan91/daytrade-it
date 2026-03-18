"""Prediction result dataclasses for ML model predictions.

This module provides data structures for representing model prediction
results including direction, confidence, and probabilities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class PredictionDirection(StrEnum):
    """Trading signal direction from model prediction."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class PredictionResult:
    """Result of a model prediction for a single ticker.

    Attributes:
        ticker: Stock ticker symbol
        direction: Predicted trading direction (BUY, SELL, HOLD)
        confidence: Confidence score of the prediction (0.0 to 1.0)
        probabilities: Dictionary containing probability for each class
        features_used: Dictionary of feature values used for prediction
        model_version: Version of the model used for prediction
        predicted_at: Timestamp when prediction was made
    """

    ticker: str
    direction: PredictionDirection
    confidence: float
    probabilities: dict[str, float]
    features_used: dict[str, Any]
    model_version: str
    predicted_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate prediction result after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        required_probs = {"buy_prob", "sell_prob", "hold_prob"}
        if not required_probs.issubset(self.probabilities.keys()):
            missing = required_probs - set(self.probabilities.keys())
            raise ValueError(f"Missing required probabilities: {missing}")

        if isinstance(self.direction, str):
            self.direction = PredictionDirection(self.direction)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the prediction result.
        """
        return {
            "ticker": self.ticker,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "features_used": self.features_used,
            "model_version": self.model_version,
            "predicted_at": self.predicted_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PredictionResult":
        """Create PredictionResult from dictionary.

        Args:
            data: Dictionary containing prediction result data.

        Returns:
            PredictionResult instance.
        """
        return cls(
            ticker=data["ticker"],
            direction=PredictionDirection(data["direction"]),
            confidence=data["confidence"],
            probabilities=data["probabilities"],
            features_used=data.get("features_used", {}),
            model_version=data["model_version"],
            predicted_at=datetime.fromisoformat(data["predicted_at"]),
        )

    @property
    def is_actionable(self) -> bool:
        """Check if prediction is actionable (not HOLD with sufficient confidence).

        Returns:
            True if direction is not HOLD and confidence is >= 0.6.
        """
        return self.direction != PredictionDirection.HOLD and self.confidence >= 0.6
