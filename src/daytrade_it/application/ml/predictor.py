"""Model predictor for serving ML predictions.

This module provides the ModelPredictor class for loading trained models
and generating predictions for stock tickers.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from daytrade_it.application.ml.feature_engineer import FeatureEngineer
from daytrade_it.application.ml.model_info import ModelInfo
from daytrade_it.application.ml.prediction import PredictionDirection, PredictionResult
from daytrade_it.application.ml.registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class ModelPredictor:
    """Predictor for serving ML model predictions.

    Loads models from the registry and generates predictions using
    the feature engineer for data preprocessing.

    Attributes:
        registry: Model registry for loading models.
        feature_engineer: Feature engineer for preprocessing.
        _model: Currently loaded model.
        _model_info: Info about the currently loaded model.
    """

    registry: ModelRegistry
    feature_engineer: FeatureEngineer
    _model: Any = field(default=None, init=False, repr=False)
    _model_info: ModelInfo | None = field(default=None, init=False, repr=False)

    def load_model(self, name: str, version: str | None = None) -> ModelInfo:
        """Load a model from the registry.

        Args:
            name: Model name.
            version: Model version. If None, loads the latest version.

        Returns:
            ModelInfo for the loaded model.

        Raises:
            KeyError: If model not found.
        """
        if version is None:
            version = self.registry.get_latest_version(name)

        self._model = self.registry.get_model(name, version)
        model_versions = self.registry.list_models(name)
        self._model_info = next(
            (m for m in model_versions if m.version == version),
            None,
        )

        if self._model_info is None:
            raise KeyError(f"Model {name} version {version} not found")

        logger.info(f"Loaded model {name} version {version}")
        return self._model_info

    def predict(self, ticker: str) -> PredictionResult:
        """Generate a prediction for a single ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            PredictionResult with direction, confidence, and probabilities.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._model is None or self._model_info is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        features = self._prepare_features(ticker)
        probabilities = self._get_probabilities(features)

        direction = self._determine_direction(probabilities)
        confidence = self._calculate_confidence(probabilities, direction)

        return PredictionResult(
            ticker=ticker.upper(),
            direction=direction,
            confidence=confidence,
            probabilities={
                "buy_prob": probabilities[0],
                "sell_prob": probabilities[1],
                "hold_prob": probabilities[2],
            },
            features_used=features.iloc[0].to_dict() if not features.empty else {},
            model_version=self._model_info.version,
            predicted_at=datetime.now(),
        )

    def predict_batch(self, tickers: list[str]) -> list[PredictionResult]:
        """Generate predictions for multiple tickers.

        Args:
            tickers: List of stock ticker symbols.

        Returns:
            List of PredictionResult objects.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._model is None or self._model_info is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        results = []
        for ticker in tickers:
            try:
                result = self.predict(ticker)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict for {ticker}: {e}")
                results.append(self._create_error_result(ticker, str(e)))

        return results

    def get_loaded_model_info(self) -> ModelInfo | None:
        """Get information about the currently loaded model.

        Returns:
            ModelInfo if a model is loaded, None otherwise.
        """
        return self._model_info

    def _prepare_features(self, ticker: str) -> pd.DataFrame:  # noqa: ARG002
        """Prepare features for prediction.

        Creates default/placeholder features for the ticker.
        In a real implementation, this would fetch actual market data.

        Args:
            ticker: Stock ticker symbol (used for fetching market data).

        Returns:
            DataFrame with features ready for prediction.
        """
        default_features = self.feature_engineer._get_default_price_features()

        news_features = {
            "title_length": 50.0,
            "content_length": 500.0,
            "sentiment_score": 0.0,
            "sentiment_confidence": 0.5,
            "hour_of_day": float(datetime.now().hour),
            "day_of_week": float(datetime.now().weekday()),
            "time_since_market_open": 0.0,
        }

        for keyword in self.feature_engineer.keywords:
            news_features[f"keyword_count_{keyword}"] = 0.0

        combined = {**news_features, **default_features}
        df = pd.DataFrame([combined])

        if self.feature_engineer._is_fitted:
            df = df.reindex(columns=self.feature_engineer.feature_names, fill_value=0.0)
            df = self.feature_engineer.transform(df)

        return df

    def _get_probabilities(self, features: pd.DataFrame) -> np.ndarray:
        """Get class probabilities from the model.

        Args:
            features: Feature DataFrame for prediction.

        Returns:
            Array of probabilities [buy_prob, sell_prob, hold_prob].
        """
        if hasattr(self._model, "predict_proba"):
            probs = self._model.predict_proba(features)[0]
            if len(probs) == 3:
                return probs
            elif len(probs) == 2:
                return np.array([probs[0], probs[1], 0.0])
        elif hasattr(self._model, "predict"):
            prediction = self._model.predict(features)[0]
            probs = np.zeros(3)
            if isinstance(prediction, (int, np.integer)):
                probs[prediction] = 1.0
            return probs

        return np.array([0.33, 0.33, 0.34])

    def _determine_direction(self, probabilities: np.ndarray) -> PredictionDirection:
        """Determine trading direction from probabilities.

        Args:
            probabilities: Array of [buy_prob, sell_prob, hold_prob].

        Returns:
            PredictionDirection based on highest probability.
        """
        max_idx = np.argmax(probabilities)
        directions = [
            PredictionDirection.BUY,
            PredictionDirection.SELL,
            PredictionDirection.HOLD,
        ]
        return directions[max_idx]

    def _calculate_confidence(
        self,
        probabilities: np.ndarray,
        direction: PredictionDirection,
    ) -> float:
        """Calculate confidence score for the prediction.

        Args:
            probabilities: Array of class probabilities.
            direction: Predicted direction.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        direction_idx = {
            PredictionDirection.BUY: 0,
            PredictionDirection.SELL: 1,
            PredictionDirection.HOLD: 2,
        }
        return float(probabilities[direction_idx[direction]])

    def _create_error_result(self, ticker: str, error: str) -> PredictionResult:
        """Create a HOLD result for error cases.

        Args:
            ticker: Stock ticker symbol.
            error: Error message.

        Returns:
            PredictionResult with HOLD direction and zero confidence.
        """
        return PredictionResult(
            ticker=ticker.upper(),
            direction=PredictionDirection.HOLD,
            confidence=0.0,
            probabilities={
                "buy_prob": 0.0,
                "sell_prob": 0.0,
                "hold_prob": 1.0,
            },
            features_used={"error": error},
            model_version=self._model_info.version if self._model_info else "unknown",
            predicted_at=datetime.now(),
        )
