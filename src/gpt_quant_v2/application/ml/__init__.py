"""Machine Learning module for quantitative trading.

This module provides ML components for feature engineering, model training,
prediction, and model serving for news-driven trading signals.
"""

from gpt_quant_v2.application.ml.feature_engineer import (
    DEFAULT_KEYWORDS,
    MARKET_CLOSE,
    MARKET_OPEN,
    FeatureEngineer,
    NewsArticleProtocol,
    PriceBar,
    SentimentAnalysisProtocol,
)
from gpt_quant_v2.application.ml.hyperparams import (
    DEFAULT_PARAM_SPACE,
    OptunaObjective,
    XGBoostParams,
    create_study,
)
from gpt_quant_v2.application.ml.mlflow_tracker import MLflowTracker
from gpt_quant_v2.application.ml.model_info import ModelInfo
from gpt_quant_v2.application.ml.prediction import (
    PredictionDirection,
    PredictionResult,
)
from gpt_quant_v2.application.ml.predictor import ModelPredictor
from gpt_quant_v2.application.ml.registry import ModelRegistry
from gpt_quant_v2.application.ml.storage import ModelStorage
from gpt_quant_v2.application.ml.trainer import (
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
