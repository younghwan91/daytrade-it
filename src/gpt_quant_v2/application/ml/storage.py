"""Model storage for serialization and deserialization."""

import logging
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


class ModelStorage:
    """Storage handler for ML models.

    Supports XGBoost and sklearn models using joblib serialization.
    """

    SUPPORTED_EXTENSIONS = {".joblib", ".pkl", ".json"}

    def save_model(self, model: Any, path: Path) -> None:
        """Save a model to the specified path.

        Args:
            model: The model to save (XGBoost or sklearn model).
            path: Path where the model will be saved.

        Raises:
            ValueError: If model type is not supported.
            IOError: If saving fails.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if self._is_xgboost_model(model):
                self._save_xgboost_model(model, path)
            else:
                self._save_sklearn_model(model, path)
            logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Failed to save model to {path}: {e}")
            raise IOError(f"Failed to save model: {e}") from e

    def load_model(self, path: Path) -> Any:
        """Load a model from the specified path.

        Args:
            path: Path to the saved model.

        Returns:
            The loaded model.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            IOError: If loading fails.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            if path.suffix == ".json":
                return self._load_xgboost_model(path)
            else:
                return self._load_sklearn_model(path)
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            raise IOError(f"Failed to load model: {e}") from e

    def _is_xgboost_model(self, model: Any) -> bool:
        """Check if model is an XGBoost model."""
        model_type = type(model).__module__
        return model_type.startswith("xgboost")

    def _save_xgboost_model(self, model: Any, path: Path) -> None:
        """Save XGBoost model using native format."""
        json_path = path.with_suffix(".json")
        model.save_model(str(json_path))
        logger.debug(f"XGBoost model saved to {json_path}")

    def _load_xgboost_model(self, path: Path) -> Any:
        """Load XGBoost model from native format."""
        try:
            import xgboost as xgb

            model = xgb.Booster()
            model.load_model(str(path))
            return model
        except ImportError:
            try:
                import xgboost as xgb

                model = xgb.XGBClassifier()
                model.load_model(str(path))
                return model
            except Exception:
                return joblib.load(path)

    def _save_sklearn_model(self, model: Any, path: Path) -> None:
        """Save sklearn model using joblib."""
        joblib_path = path.with_suffix(".joblib")
        joblib.dump(model, joblib_path)
        logger.debug(f"sklearn model saved to {joblib_path}")

    def _load_sklearn_model(self, path: Path) -> Any:
        """Load sklearn model using joblib."""
        return joblib.load(path)
