"""Machine Learning module for quantitative trading.

This module provides ML components for feature engineering, model training,
prediction, and model serving for news-driven trading signals.
"""

from krx_signal_engine.application.ml.feature_engineer import (
    DEFAULT_KEYWORDS,
    MARKET_CLOSE,
    MARKET_OPEN,
    FeatureEngineer,
    NewsArticleProtocol,
    PriceBar,
    SentimentAnalysisProtocol,
)
from krx_signal_engine.application.ml.hyperparams import (
    DEFAULT_PARAM_SPACE,
    OptunaObjective,
    XGBoostParams,
    create_study,
)
from krx_signal_engine.application.ml.mlflow_tracker import MLflowTracker
from krx_signal_engine.application.ml.model_info import ModelInfo
from krx_signal_engine.application.ml.prediction import (
    PredictionDirection,
    PredictionResult,
)
from krx_signal_engine.application.ml.predictor import ModelPredictor
from krx_signal_engine.application.ml.registry import ModelRegistry
from krx_signal_engine.application.ml.storage import ModelStorage
from krx_signal_engine.application.ml.trainer import (
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
