"""Hyperparameter configuration for ML model training.

This module provides hyperparameter configurations and Optuna-based
optimization for XGBoost trading models.
"""

from dataclasses import dataclass, field
from typing import Any

import optuna
from optuna import Study, Trial
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBClassifier


@dataclass
class XGBoostParams:
    """XGBoost hyperparameters with sensible defaults for trading.

    Attributes:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Step size shrinkage
        subsample: Subsample ratio of training instances
        colsample_bytree: Subsample ratio of columns
        min_child_weight: Minimum sum of instance weight in child
        gamma: Minimum loss reduction for split
        reg_alpha: L1 regularization term
        reg_lambda: L2 regularization term
        scale_pos_weight: Balance of positive and negative weights
        objective: Learning objective
        eval_metric: Evaluation metric
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel threads
    """

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    scale_pos_weight: float = 1.0
    objective: str = "multi:softprob"
    eval_metric: str = "mlogloss"
    random_state: int = 42
    n_jobs: int = -1

    def to_dict(self) -> dict[str, Any]:
        """Convert parameters to dictionary for XGBoost."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": self.scale_pos_weight,
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "XGBoostParams":
        """Create XGBoostParams from dictionary."""
        valid_keys = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "scale_pos_weight",
            "objective",
            "eval_metric",
            "random_state",
            "n_jobs",
        }
        filtered = {k: v for k, v in params.items() if k in valid_keys}
        return cls(**filtered)


# Default hyperparameter search space for Optuna
DEFAULT_PARAM_SPACE: dict[str, Any] = {
    "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
    "max_depth": {"type": "int", "low": 3, "high": 12},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
    "min_child_weight": {"type": "int", "low": 1, "high": 10},
    "gamma": {"type": "float", "low": 0.0, "high": 5.0},
    "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
}


@dataclass
class OptunaObjective:
    """Optuna objective for XGBoost hyperparameter optimization.

    Uses time-series cross-validation to evaluate hyperparameters,
    which is essential for trading models to avoid look-ahead bias.

    Attributes:
        X: Feature matrix
        y: Target labels
        n_splits: Number of cross-validation splits
        param_space: Hyperparameter search space
        scoring: Scoring metric for cross-validation
    """

    X: Any  # numpy array or DataFrame
    y: Any  # numpy array or Series
    n_splits: int = 5
    param_space: dict[str, Any] = field(default_factory=lambda: DEFAULT_PARAM_SPACE.copy())
    scoring: str = "f1_weighted"

    def __call__(self, trial: Trial) -> float:
        """Evaluate a trial with sampled hyperparameters.

        Args:
            trial: Optuna trial object

        Returns:
            Mean cross-validation score
        """
        params = self._sample_params(trial)

        model = XGBClassifier(
            **params,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
        )

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        scores = cross_val_score(
            model,
            self.X,
            self.y,
            cv=tscv,
            scoring=self.scoring,
            n_jobs=-1,
        )

        return float(scores.mean())

    def _sample_params(self, trial: Trial) -> dict[str, Any]:
        """Sample hyperparameters from search space.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of sampled hyperparameters
        """
        params: dict[str, Any] = {}

        for name, space in self.param_space.items():
            param_type = space.get("type", "float")

            if param_type == "int":
                params[name] = trial.suggest_int(
                    name,
                    space["low"],
                    space["high"],
                    step=space.get("step", 1),
                )
            elif param_type == "float":
                params[name] = trial.suggest_float(
                    name,
                    space["low"],
                    space["high"],
                    log=space.get("log", False),
                )
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, space["choices"])

        return params


def create_study(
    study_name: str = "xgboost_optimization",
    direction: str = "maximize",
    storage: str | None = None,
    load_if_exists: bool = True,
    sampler: optuna.samplers.BaseSampler | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
) -> Study:
    """Create an Optuna study for hyperparameter optimization.

    Args:
        study_name: Name of the study
        direction: Optimization direction ("maximize" or "minimize")
        storage: Database URL for study persistence (optional)
        load_if_exists: Load existing study if it exists
        sampler: Optuna sampler (defaults to TPESampler)
        pruner: Optuna pruner (defaults to MedianPruner)

    Returns:
        Configured Optuna study
    """
    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=42)

    if pruner is None:
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=load_if_exists,
        sampler=sampler,
        pruner=pruner,
    )

    return study
