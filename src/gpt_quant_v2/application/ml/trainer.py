"""Model training module for ML-based trading signals.

This module provides the ModelTrainer class for training XGBoost
classification models with hyperparameter optimization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from gpt_quant_v2.application.ml.feature_engineer import FeatureEngineer
from gpt_quant_v2.application.ml.hyperparams import (
    DEFAULT_PARAM_SPACE,
    OptunaObjective,
    XGBoostParams,
    create_study,
)


class SignalLabel(IntEnum):
    """Trading signal labels for classification."""

    SELL = 0
    HOLD = 1
    BUY = 2


# Label mapping for string conversion
LABEL_NAMES: dict[int, str] = {
    SignalLabel.SELL: "SELL",
    SignalLabel.HOLD: "HOLD",
    SignalLabel.BUY: "BUY",
}


@dataclass
class TrainedModel:
    """Container for a trained model and its metadata.

    Attributes:
        model: The trained XGBoost classifier
        feature_engineer: FeatureEngineer used for preprocessing
        params: Training parameters used
        metrics: Evaluation metrics
        feature_names: List of feature names
        label_names: Mapping of label indices to names
        created_at: Timestamp when model was trained
        model_version: Model version string
    """

    model: XGBClassifier
    feature_engineer: FeatureEngineer
    params: dict[str, Any]
    metrics: dict[str, float]
    feature_names: list[str]
    label_names: dict[int, str] = field(default_factory=lambda: dict(LABEL_NAMES))
    created_at: datetime = field(default_factory=datetime.utcnow)
    model_version: str = "1.0.0"

    def predict(self, features: pd.DataFrame) -> np.ndarray[Any, np.dtype[Any]]:
        """Make predictions on new features.

        Args:
            features: DataFrame with feature columns

        Returns:
            Array of predicted labels
        """
        # Ensure columns are in correct order
        features = features.reindex(columns=self.feature_names, fill_value=0.0)
        result: np.ndarray[Any, np.dtype[Any]] = self.model.predict(features)
        return result

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray[Any, np.dtype[Any]]:
        """Get prediction probabilities.

        Args:
            features: DataFrame with feature columns

        Returns:
            Array of class probabilities
        """
        features = features.reindex(columns=self.feature_names, fill_value=0.0)
        result: np.ndarray[Any, np.dtype[Any]] = self.model.predict_proba(features)
        return result

    def predict_label(self, features: pd.DataFrame) -> list[str]:
        """Get predicted label names.

        Args:
            features: DataFrame with feature columns

        Returns:
            List of label names (BUY, SELL, HOLD)
        """
        predictions = self.predict(features)
        return [self.label_names[int(p)] for p in predictions]

    def save(self, path: Path) -> None:
        """Save the trained model to disk.

        Args:
            path: Directory path to save model files
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save XGBoost model
        self.model.save_model(path / "model.ubj")

        # Save metadata and feature engineer
        metadata = {
            "params": self.params,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "created_at": self.created_at.isoformat(),
            "model_version": self.model_version,
        }
        joblib.dump(metadata, path / "metadata.joblib")
        joblib.dump(self.feature_engineer, path / "feature_engineer.joblib")

    @classmethod
    def load(cls, path: Path) -> "TrainedModel":
        """Load a trained model from disk.

        Args:
            path: Directory path containing model files

        Returns:
            TrainedModel instance
        """
        # Load XGBoost model
        model = XGBClassifier()
        model.load_model(path / "model.ubj")

        # Load metadata
        metadata = joblib.load(path / "metadata.joblib")
        feature_engineer = joblib.load(path / "feature_engineer.joblib")

        return cls(
            model=model,
            feature_engineer=feature_engineer,
            params=metadata["params"],
            metrics=metadata["metrics"],
            feature_names=metadata["feature_names"],
            label_names=metadata["label_names"],
            created_at=datetime.fromisoformat(metadata["created_at"]),
            model_version=metadata["model_version"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert model metadata to dictionary for serialization."""
        return {
            "params": self.params,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "created_at": self.created_at.isoformat(),
            "model_version": self.model_version,
        }


@dataclass
class ModelTrainer:
    """Trainer for XGBoost classification models.

    Handles model training with hyperparameter optimization,
    cross-validation using time-series splits, and model evaluation.

    Attributes:
        feature_engineer: FeatureEngineer for data preprocessing
        model_dir: Directory for saving models
        n_cv_splits: Number of cross-validation splits
        optimization_trials: Number of Optuna trials
    """

    feature_engineer: FeatureEngineer
    model_dir: Path
    n_cv_splits: int = 5
    optimization_trials: int = 50

    def __post_init__(self) -> None:
        """Initialize model directory."""
        self.model_dir = Path(self.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        training_data: pd.DataFrame,
        params: dict[str, Any] | None = None,
        target_column: str = "label",
        optimize_hyperparams: bool = True,
    ) -> TrainedModel:
        """Train a model on the provided data.

        Args:
            training_data: DataFrame with features and target label
            params: Optional hyperparameters (if None, will optimize)
            target_column: Name of the target column
            optimize_hyperparams: Whether to run hyperparameter optimization

        Returns:
            TrainedModel with trained model and metrics
        """
        # Separate features and target
        if target_column not in training_data.columns:
            raise ValueError(f"Target column '{target_column}' not in training_data")

        X = training_data.drop(columns=[target_column])
        y = training_data[target_column].values

        # Fit the scaler on training features
        self.feature_engineer.fit_scaler(X)
        X_scaled = self.feature_engineer.transform(X)
        feature_names = X_scaled.columns.tolist()

        # Get or optimize hyperparameters
        if params is None and optimize_hyperparams:
            params = self._optimize_hyperparams(X_scaled.values, y)
        elif params is None:
            params = XGBoostParams().to_dict()

        # Train final model
        model = self._train_model(X_scaled.values, y, params)

        # Evaluate model
        metrics = self._evaluate_model(model, X_scaled.values, y)

        return TrainedModel(
            model=model,
            feature_engineer=self.feature_engineer,
            params=params,
            metrics=metrics,
            feature_names=feature_names,
        )

    def _optimize_hyperparams(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_space: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Optimize hyperparameters using Optuna.

        Args:
            X: Feature matrix
            y: Target labels
            param_space: Optional custom parameter search space

        Returns:
            Dictionary of optimized hyperparameters
        """
        if param_space is None:
            param_space = DEFAULT_PARAM_SPACE

        objective = OptunaObjective(
            X=X,
            y=y,
            n_splits=self.n_cv_splits,
            param_space=param_space,
            scoring="f1_weighted",
        )

        study = create_study(
            study_name=f"xgboost_opt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            direction="maximize",
        )

        study.optimize(objective, n_trials=self.optimization_trials, show_progress_bar=True)

        # Get best params and add fixed params
        best_params: dict[str, Any] = dict(study.best_params)
        best_params.update({
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
            "use_label_encoder": False,
        })

        return best_params

    def _train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> XGBClassifier:
        """Train XGBoost model with given parameters.

        Args:
            X: Feature matrix
            y: Target labels
            params: Model hyperparameters

        Returns:
            Trained XGBClassifier
        """
        model = XGBClassifier(**params)
        model.fit(X, y)
        return model

    def _evaluate_model(
        self,
        model: XGBClassifier,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate model using time-series cross-validation.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target labels

        Returns:
            Dictionary of evaluation metrics
        """
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)

        all_preds: list[Any] = []
        all_true: list[Any] = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone and retrain model for CV
            cv_model = model.__class__(**model.get_params())
            cv_model.fit(X_train, y_train)

            preds = cv_model.predict(X_test)
            all_preds.extend(preds)
            all_true.extend(y_test)

        preds_arr = np.array(all_preds)
        true_arr = np.array(all_true)

        metrics = {
            "accuracy": float(accuracy_score(true_arr, preds_arr)),
            "precision_weighted": float(
                precision_score(true_arr, preds_arr, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(true_arr, preds_arr, average="weighted", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(true_arr, preds_arr, average="weighted", zero_division=0)
            ),
            "precision_macro": float(
                precision_score(true_arr, preds_arr, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(true_arr, preds_arr, average="macro", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(true_arr, preds_arr, average="macro", zero_division=0)
            ),
        }

        # Per-class metrics
        for label in SignalLabel:
            binary_true = (true_arr == label.value).astype(int)
            binary_pred = (preds_arr == label.value).astype(int)

            metrics[f"precision_{label.name.lower()}"] = float(
                precision_score(binary_true, binary_pred, zero_division=0)
            )
            metrics[f"recall_{label.name.lower()}"] = float(
                recall_score(binary_true, binary_pred, zero_division=0)
            )
            metrics[f"f1_{label.name.lower()}"] = float(
                f1_score(binary_true, binary_pred, zero_division=0)
            )

        return metrics

    def train_with_validation(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        params: dict[str, Any] | None = None,
        target_column: str = "label",
        early_stopping_rounds: int = 20,
    ) -> TrainedModel:
        """Train model with separate validation set and early stopping.

        Args:
            train_data: Training DataFrame
            val_data: Validation DataFrame
            params: Optional hyperparameters
            target_column: Name of the target column
            early_stopping_rounds: Rounds without improvement to stop

        Returns:
            TrainedModel with trained model and metrics
        """
        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column].values

        X_val = val_data.drop(columns=[target_column])
        y_val = val_data[target_column].values

        # Fit scaler on training data only
        self.feature_engineer.fit_scaler(X_train)
        X_train_scaled = self.feature_engineer.transform(X_train)
        X_val_scaled = self.feature_engineer.transform(X_val)
        feature_names = X_train_scaled.columns.tolist()

        if params is None:
            params = XGBoostParams().to_dict()

        # Train with early stopping
        model = XGBClassifier(
            early_stopping_rounds=early_stopping_rounds,
            **params,
        )
        model.fit(
            X_train_scaled.values,
            y_train,
            eval_set=[(X_val_scaled.values, y_val)],
            verbose=False,
        )

        # Evaluate on validation set
        val_preds = model.predict(X_val_scaled.values)
        metrics = {
            "accuracy": float(accuracy_score(y_val, val_preds)),
            "precision_weighted": float(
                precision_score(y_val, val_preds, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(y_val, val_preds, average="weighted", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_val, val_preds, average="weighted", zero_division=0)
            ),
        }

        return TrainedModel(
            model=model,
            feature_engineer=self.feature_engineer,
            params=params,
            metrics=metrics,
            feature_names=feature_names,
        )

    def save_model(self, trained_model: TrainedModel, model_name: str) -> Path:
        """Save a trained model to the model directory.

        Args:
            trained_model: TrainedModel to save
            model_name: Name for the model subdirectory

        Returns:
            Path where model was saved
        """
        model_path = self.model_dir / model_name
        trained_model.save(model_path)
        return model_path

    def load_model(self, model_name: str) -> TrainedModel:
        """Load a trained model from the model directory.

        Args:
            model_name: Name of the model subdirectory

        Returns:
            Loaded TrainedModel
        """
        model_path = self.model_dir / model_name
        return TrainedModel.load(model_path)
