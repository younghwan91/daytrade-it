"""Model registry for managing ML models."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from gpt_quant_v2.application.ml.model_info import ModelInfo
from gpt_quant_v2.application.ml.storage import ModelStorage

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing ML models with optional MLflow integration."""

    REGISTRY_FILE = "registry.json"

    def __init__(self, storage_path: Path, mlflow_tracking_uri: str | None = None) -> None:
        """Initialize the model registry.

        Args:
            storage_path: Base path for storing models.
            mlflow_tracking_uri: Optional MLflow tracking server URI.
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self._storage = ModelStorage()
        self._registry_path = self.storage_path / self.REGISTRY_FILE
        self._registry: dict[str, dict[str, ModelInfo]] = self._load_registry()

        if mlflow_tracking_uri:
            self._init_mlflow()

    def _init_mlflow(self) -> None:
        """Initialize MLflow tracking."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            logger.info(f"MLflow tracking initialized: {self.mlflow_tracking_uri}")
        except ImportError:
            logger.warning("MLflow not installed. Skipping MLflow integration.")
            self.mlflow_tracking_uri = None

    def _load_registry(self) -> dict[str, dict[str, ModelInfo]]:
        """Load registry from disk."""
        if not self._registry_path.exists():
            return {}

        try:
            with open(self._registry_path) as f:
                data = json.load(f)
            registry: dict[str, dict[str, ModelInfo]] = {}
            for name, versions in data.items():
                registry[name] = {
                    version: ModelInfo.from_dict(info) for version, info in versions.items()
                }
            return registry
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return {}

    def _save_registry(self) -> None:
        """Save registry to disk."""
        data: dict[str, dict[str, dict[str, Any]]] = {}
        for name, versions in self._registry.items():
            data[name] = {version: info.to_dict() for version, info in versions.items()}

        with open(self._registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        model: Any,
        name: str,
        version: str,
        metrics: dict[str, float],
        parameters: dict[str, Any] | None = None,
    ) -> ModelInfo:
        """Register a new model version.

        Args:
            model: The trained model to register.
            name: Model name.
            version: Model version string.
            metrics: Model metrics (accuracy, precision, recall, f1).
            parameters: Optional model parameters/hyperparameters.

        Returns:
            ModelInfo for the registered model.

        Raises:
            ValueError: If model version already exists.
        """
        if name in self._registry and version in self._registry[name]:
            raise ValueError(f"Model {name} version {version} already exists")

        model_dir = self.storage_path / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = model_dir / "model"

        self._storage.save_model(model, artifact_path)

        model_info = ModelInfo(
            name=name,
            version=version,
            created_at=datetime.now(),
            metrics=metrics,
            parameters=parameters or {},
            stage="development",
            artifact_path=artifact_path,
        )

        if name not in self._registry:
            self._registry[name] = {}
        self._registry[name][version] = model_info
        self._save_registry()

        if self.mlflow_tracking_uri:
            self._log_to_mlflow(model_info, model)

        logger.info(f"Registered model {name} version {version}")
        return model_info

    def _log_to_mlflow(self, model_info: ModelInfo, model: Any) -> None:
        """Log model to MLflow."""
        try:
            import mlflow

            with mlflow.start_run(run_name=f"{model_info.name}-{model_info.version}"):
                mlflow.log_params(model_info.parameters)
                mlflow.log_metrics(model_info.metrics)
                mlflow.log_param("version", model_info.version)
                mlflow.log_param("stage", model_info.stage)

                if hasattr(model, "get_params"):
                    try:
                        import mlflow.sklearn

                        mlflow.sklearn.log_model(model, "model")
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    def get_model(self, name: str, version: str | None = None) -> Any:
        """Get a model by name and optional version.

        Args:
            name: Model name.
            version: Model version. If None, returns latest version.

        Returns:
            The loaded model.

        Raises:
            KeyError: If model not found.
        """
        if name not in self._registry:
            raise KeyError(f"Model {name} not found in registry")

        if version is None:
            version = self.get_latest_version(name)

        if version not in self._registry[name]:
            raise KeyError(f"Model {name} version {version} not found")

        model_info = self._registry[name][version]
        return self._load_model_artifact(model_info)

    def _load_model_artifact(self, model_info: ModelInfo) -> Any:
        """Load model artifact from storage."""
        artifact_path = model_info.artifact_path
        for ext in [".joblib", ".json", ".pkl"]:
            path = artifact_path.with_suffix(ext)
            if path.exists():
                return self._storage.load_model(path)

        raise FileNotFoundError(f"Model artifact not found at {artifact_path}")

    def list_models(self, name: str | None = None) -> list[ModelInfo]:
        """List all models or versions of a specific model.

        Args:
            name: Optional model name to filter by.

        Returns:
            List of ModelInfo objects.
        """
        if name is not None:
            if name not in self._registry:
                return []
            return list(self._registry[name].values())

        result: list[ModelInfo] = []
        for versions in self._registry.values():
            result.extend(versions.values())
        return result

    def get_latest_version(self, name: str) -> str:
        """Get the latest version of a model.

        Args:
            name: Model name.

        Returns:
            Latest version string.

        Raises:
            KeyError: If model not found.
        """
        if name not in self._registry or not self._registry[name]:
            raise KeyError(f"Model {name} not found in registry")

        versions = self._registry[name]
        latest = max(versions.values(), key=lambda m: m.created_at)
        return latest.version

    def promote_model(self, name: str, version: str, stage: str) -> None:
        """Promote a model to a new stage.

        Args:
            name: Model name.
            version: Model version.
            stage: Target stage (staging, production).

        Raises:
            KeyError: If model not found.
            ValueError: If stage is invalid.
        """
        valid_stages = {"development", "staging", "production"}
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {valid_stages}")

        if name not in self._registry:
            raise KeyError(f"Model {name} not found")
        if version not in self._registry[name]:
            raise KeyError(f"Model {name} version {version} not found")

        old_stage = self._registry[name][version].stage
        self._registry[name][version].stage = stage
        self._save_registry()

        if self.mlflow_tracking_uri:
            self._update_mlflow_stage(name, version, stage)

        logger.info(f"Promoted model {name} version {version}: {old_stage} -> {stage}")

    def _update_mlflow_stage(self, name: str, version: str, stage: str) -> None:
        """Update model stage in MLflow."""
        try:
            import mlflow

            client = mlflow.tracking.MlflowClient()
            mlflow_stage_map = {
                "development": "None",
                "staging": "Staging",
                "production": "Production",
            }
            mlflow_stage = mlflow_stage_map.get(stage, "None")
            client.transition_model_version_stage(
                name=name,
                version=version,
                stage=mlflow_stage,
            )
        except Exception as e:
            logger.warning(f"Failed to update MLflow stage: {e}")

    def delete_model(self, name: str, version: str) -> None:
        """Delete a model version.

        Args:
            name: Model name.
            version: Model version.

        Raises:
            KeyError: If model not found.
        """
        if name not in self._registry:
            raise KeyError(f"Model {name} not found")
        if version not in self._registry[name]:
            raise KeyError(f"Model {name} version {version} not found")

        model_info = self._registry[name][version]
        model_dir = model_info.artifact_path.parent

        import shutil

        if model_dir.exists():
            shutil.rmtree(model_dir)

        del self._registry[name][version]
        if not self._registry[name]:
            del self._registry[name]

        self._save_registry()
        logger.info(f"Deleted model {name} version {version}")
