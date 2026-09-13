"""Machine Learning module for quantitative trading.

This module provides ML components for feature engineering, model training,
prediction, and model serving for news-driven trading signals.
"""

from daytrade_it.application.ml.feature_engineer import (
    DEFAULT_KEYWORDS,
    MARKET_CLOSE,
    MARKET_OPEN,
    FeatureEngineer,
    NewsArticleProtocol,
    PriceBar,
    SentimentAnalysisProtocol,
)
from daytrade_it.application.ml.hyperparams import (
    DEFAULT_PARAM_SPACE,
    OptunaObjective,
    XGBoostParams,
    create_study,
)
from daytrade_it.application.ml.mlflow_tracker import MLflowTracker
from daytrade_it.application.ml.model_info import ModelInfo
from daytrade_it.application.ml.prediction import (
    PredictionDirection,
    PredictionResult,
)
from daytrade_it.application.ml.predictor import ModelPredictor
from daytrade_it.application.ml.registry import ModelRegistry
from daytrade_it.application.ml.storage import ModelStorage
from daytrade_it.application.ml.trainer import (
    LABEL_NAMES,
    ModelTrainer,
    SignalLabel,
    TrainedModel,
)

__all__ = [
    # Feature engineering
    "DEFAULT_KEYWORDS",
    "FeatureEngineer",
    "MARKET_CLOSE",
    "MARKET_OPEN",
    "NewsArticleProtocol",
    "PriceBar",
    "SentimentAnalysisProtocol",
    # Hyperparameters
    "DEFAULT_PARAM_SPACE",
    "OptunaObjective",
    "XGBoostParams",
    "create_study",
    # MLflow tracking
    "MLflowTracker",
    # Model registry
    "ModelInfo",
    "ModelRegistry",
    "ModelStorage",
    # Model training
    "LABEL_NAMES",
    "ModelTrainer",
    "SignalLabel",
    "TrainedModel",
    # Model serving
    "ModelPredictor",
    "PredictionDirection",
    "PredictionResult",
]
