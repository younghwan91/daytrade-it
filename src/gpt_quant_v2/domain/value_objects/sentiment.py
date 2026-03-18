"""Sentiment value object representing market sentiment analysis."""

from enum import StrEnum
from typing import Any, Self

from pydantic import field_validator, model_validator

from gpt_quant_v2.domain.value_objects.base import ValueObject


class SentimentLabel(StrEnum):
    """Enumeration of sentiment labels."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class Sentiment(ValueObject):
    """Represents market sentiment with a score, label, and confidence.

    Sentiment captures the market outlook with:
    - score: float from -1.0 (bearish) to 1.0 (bullish)
    - label: categorical label (BULLISH, BEARISH, NEUTRAL)
    - confidence: float from 0.0 to 1.0 indicating certainty
    """

    score: float
    label: SentimentLabel
    confidence: float

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Validate score is in range [-1.0, 1.0]."""
        if not -1.0 <= v <= 1.0:
            raise ValueError("Score must be between -1.0 and 1.0")
        return v

    @field_validator("label", mode="before")
    @classmethod
    def coerce_label(cls, v: Any) -> SentimentLabel:
        """Coerce label to SentimentLabel enum."""
        if isinstance(v, SentimentLabel):
            return v
        if isinstance(v, str):
            try:
                return SentimentLabel(v.upper())
            except ValueError:
                valid_labels = ", ".join(label.value for label in SentimentLabel)
                raise ValueError(f"Label must be one of: {valid_labels}") from None
        raise ValueError(f"Cannot convert {type(v).__name__} to SentimentLabel")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence is in range [0.0, 1.0]."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def validate_score_label_consistency(self) -> Self:
        """Validate that score and label are consistent."""
        # Allow some flexibility for edge cases
        if self.label == SentimentLabel.BULLISH and self.score < 0:
            raise ValueError("Bullish sentiment cannot have negative score")
        if self.label == SentimentLabel.BEARISH and self.score > 0:
            raise ValueError("Bearish sentiment cannot have positive score")
        return self

    @classmethod
    def bullish(cls, score: float = 0.8, confidence: float = 0.9) -> "Sentiment":
        """Create a bullish sentiment.

        Args:
            score: Bullish score (0.0 to 1.0), defaults to 0.8
            confidence: Confidence level (0.0 to 1.0), defaults to 0.9
        """
        if score < 0:
            raise ValueError("Bullish score must be non-negative")
        return cls(score=score, label=SentimentLabel.BULLISH, confidence=confidence)

    @classmethod
    def bearish(cls, score: float = -0.8, confidence: float = 0.9) -> "Sentiment":
        """Create a bearish sentiment.

        Args:
            score: Bearish score (-1.0 to 0.0), defaults to -0.8
            confidence: Confidence level (0.0 to 1.0), defaults to 0.9
        """
        if score > 0:
            raise ValueError("Bearish score must be non-positive")
        return cls(score=score, label=SentimentLabel.BEARISH, confidence=confidence)

    @classmethod
    def neutral(cls, score: float = 0.0, confidence: float = 0.9) -> "Sentiment":
        """Create a neutral sentiment.

        Args:
            score: Neutral score (typically near 0), defaults to 0.0
            confidence: Confidence level (0.0 to 1.0), defaults to 0.9
        """
        return cls(score=score, label=SentimentLabel.NEUTRAL, confidence=confidence)

    @property
    def is_bullish(self) -> bool:
        """Check if sentiment is bullish."""
        return self.label == SentimentLabel.BULLISH

    @property
    def is_bearish(self) -> bool:
        """Check if sentiment is bearish."""
        return self.label == SentimentLabel.BEARISH

    @property
    def is_neutral(self) -> bool:
        """Check if sentiment is neutral."""
        return self.label == SentimentLabel.NEUTRAL

    def __str__(self) -> str:
        """Return formatted sentiment string."""
        return f"{self.label.value} (score={self.score:.2f}, confidence={self.confidence:.0%})"

    def __hash__(self) -> int:
        """Make sentiment hashable."""
        return hash((self.score, self.label, self.confidence))

    def __eq__(self, other: Any) -> bool:
        """Compare sentiments by all fields."""
        if isinstance(other, Sentiment):
            return (
                self.score == other.score
                and self.label == other.label
                and self.confidence == other.confidence
            )
        return NotImplemented

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Sentiment(score={self.score}, label='{self.label.value}', confidence={self.confidence})"
