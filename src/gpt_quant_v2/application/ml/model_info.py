"""Model information dataclass."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ModelInfo:
    """Information about a registered model."""

    name: str
    version: str
    created_at: datetime
    metrics: dict[str, float] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    stage: str = "development"
    artifact_path: Path = field(default_factory=lambda: Path())

    def __post_init__(self) -> None:
        """Validate model info after initialization."""
        valid_stages = {"development", "staging", "production"}
        if self.stage not in valid_stages:
            raise ValueError(f"Invalid stage '{self.stage}'. Must be one of {valid_stages}")
        if isinstance(self.artifact_path, str):
            self.artifact_path = Path(self.artifact_path)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "metrics": self.metrics,
            "parameters": self.parameters,
            "stage": self.stage,
            "artifact_path": str(self.artifact_path),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelInfo":
        """Create ModelInfo from dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metrics=data.get("metrics", {}),
            parameters=data.get("parameters", {}),
            stage=data.get("stage", "development"),
            artifact_path=Path(data.get("artifact_path", "")),
        )
