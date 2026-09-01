"""Feature engineering module for ML-based trading signals.

This module provides the FeatureEngineer class for creating ML features
from news articles and price data for quantitative trading models.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Market hours (US Eastern Time)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# Default keywords for financial analysis
DEFAULT_KEYWORDS = [
    "earnings",
    "revenue",
    "profit",
    "loss",
    "guidance",
    "upgrade",
    "downgrade",
    "buy",
    "sell",
    "merger",
    "acquisition",
    "dividend",
    "buyback",
    "lawsuit",
    "fda",
    "approval",
    "beat",
    "miss",
    "outlook",
    "growth",
]


class NewsArticleProtocol(Protocol):
    """Protocol for NewsArticle-like objects."""

    title: str
    content: str
    published_at: datetime


class SentimentAnalysisProtocol(Protocol):
    """Protocol for SentimentAnalysis-like objects."""

    sentiment_score: float
    confidence: float


@dataclass
class PriceBar:
    """Represents an OHLCV price bar.

    Attributes:
        timestamp: Bar timestamp
        open: Opening price
        high: High price
        low: Low price
        close: Closing price
        volume: Trading volume
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PriceBar":
        """Create PriceBar from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=int(data["volume"]),
        )


@dataclass
class FeatureEngineer:
    """Feature engineering for ML-based trading signals.

    Creates ML features from news articles and price data for use in
    quantitative trading models. Handles feature extraction, scaling,
    normalization, and missing data.

    Attributes:
        keywords: List of keywords to count in news content
        scaler: StandardScaler for feature normalization
        feature_names: List of feature names after fitting
    """

    keywords: list[str] = field(default_factory=lambda: DEFAULT_KEYWORDS.copy())
    scaler: StandardScaler = field(default_factory=StandardScaler)
    feature_names: list[str] = field(default_factory=list)
    _is_fitted: bool = field(default=False, init=False)

    def extract_news_features(
        self,
        article: NewsArticleProtocol,
        sentiment: SentimentAnalysisProtocol,
    ) -> dict[str, float]:
        """Extract features from a news article and its sentiment analysis.

        Args:
            article: News article with title, content, and published_at
            sentiment: Sentiment analysis with score and confidence

        Returns:
            Dictionary of extracted features:
                - title_length: Character count of title
                - content_length: Character count of content
                - sentiment_score: Sentiment score (-1 to 1)
                - sentiment_confidence: Confidence level (0 to 1)
                - hour_of_day: Hour when article was published (0-23)
                - day_of_week: Day of week (0=Monday, 6=Sunday)
                - time_since_market_open: Minutes since market open
                - keyword_count_*: Count for each tracked keyword
        """
        features: dict[str, float] = {}

        # Text length features
        features["title_length"] = float(len(article.title))
        features["content_length"] = float(len(article.content))

        # Sentiment features
        features["sentiment_score"] = float(sentiment.sentiment_score)
        features["sentiment_confidence"] = float(sentiment.confidence)

        # Time features
        published_at = article.published_at
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=UTC)

        features["hour_of_day"] = float(published_at.hour)
        features["day_of_week"] = float(published_at.weekday())
        features["time_since_market_open"] = self._calculate_time_since_market_open(published_at)

        # Keyword counts
        content_lower = article.content.lower()
        title_lower = article.title.lower()
        combined_text = f"{title_lower} {content_lower}"

        for keyword in self.keywords:
            features[f"keyword_count_{keyword}"] = float(combined_text.count(keyword.lower()))

        return features

    def extract_price_features(
        self,
        bars: list[PriceBar],
        window: int = 20,
    ) -> dict[str, float]:
        """Extract features from price bars.

        Args:
            bars: List of PriceBar objects, ordered by timestamp (oldest first)
            window: Lookback window for calculations (default 20)

        Returns:
            Dictionary of extracted features:
                - return_1m: 1-minute return
                - return_5m: 5-minute return
                - return_10m: 10-minute return
                - return_20m: 20-minute return
                - volatility: Standard deviation of returns
                - volume_ratio: Current volume vs average
                - rsi_14: 14-period RSI
                - ma_distance: Distance from moving average (%)
        """
        features: dict[str, float] = {}

        if not bars:
            return self._get_default_price_features()

        # Convert to arrays for calculation
        closes = np.array([bar.close for bar in bars])
        volumes = np.array([bar.volume for bar in bars])

        n_bars = len(bars)

        # Return calculations
        features["return_1m"] = self._calculate_return(closes, 1)
        features["return_5m"] = self._calculate_return(closes, 5)
        features["return_10m"] = self._calculate_return(closes, 10)
        features["return_20m"] = self._calculate_return(closes, min(20, n_bars - 1))

        # Volatility (standard deviation of returns)
        features["volatility"] = self._calculate_volatility(closes, window)

        # Volume ratio
        features["volume_ratio"] = self._calculate_volume_ratio(volumes, window)

        # RSI (14 period)
        features["rsi_14"] = self._calculate_rsi(closes, 14)

        # Moving average distance
        features["ma_distance"] = self._calculate_ma_distance(closes, window)

        return features

    def combine_features(
        self,
        news_features: dict[str, float],
        price_features: dict[str, float],
    ) -> pd.DataFrame:
        """Combine news and price features into a single DataFrame.

        Args:
            news_features: Dictionary of news-derived features
            price_features: Dictionary of price-derived features

        Returns:
            DataFrame with all combined features as a single row
        """
        combined = {**news_features, **price_features}
        return pd.DataFrame([combined])

    def prepare_training_data(
        self,
        articles: list[tuple[NewsArticleProtocol, SentimentAnalysisProtocol]],
        prices: dict[datetime, list[PriceBar]],
        window: int = 20,
    ) -> pd.DataFrame:
        """Prepare training data from articles and price data.

        Args:
            articles: List of (article, sentiment) tuples
            prices: Dictionary mapping timestamps to price bars
            window: Lookback window for price features

        Returns:
            DataFrame with all features for each article, scaled and normalized
        """
        if not articles:
            return pd.DataFrame()

        all_features = []

        for article, sentiment in articles:
            # Extract news features
            news_features = self.extract_news_features(article, sentiment)

            # Find matching price data
            price_bars = self._find_price_bars(article.published_at, prices, window)
            price_features = self.extract_price_features(price_bars, window)

            # Combine features
            combined = {**news_features, **price_features}
            all_features.append(combined)

        df = pd.DataFrame(all_features)

        # Handle missing data
        df = self._handle_missing_data(df)

        # Store feature names
        self.feature_names = df.columns.tolist()

        return df

    def fit_scaler(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Fit the scaler on training data.

        Args:
            df: Training data DataFrame

        Returns:
            Self for method chaining
        """
        if df.empty:
            return self

        self.scaler.fit(df)
        self._is_fitted = True
        self.feature_names = df.columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using the fitted scaler.

        Args:
            df: DataFrame with features to transform

        Returns:
            Scaled DataFrame with same columns
        """
        if not self._is_fitted:
            raise ValueError("Scaler must be fitted before transform. Call fit_scaler first.")

        if df.empty:
            return df

        # Ensure columns are in the same order as during fitting
        df = df.reindex(columns=self.feature_names, fill_value=0.0)

        scaled = self.scaler.transform(df)
        return pd.DataFrame(scaled, columns=self.feature_names, index=df.index)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and transform features in one step.

        Args:
            df: DataFrame with features to fit and transform

        Returns:
            Scaled DataFrame with same columns
        """
        if df.empty:
            return df

        self.feature_names = df.columns.tolist()
        scaled = self.scaler.fit_transform(df)
        self._is_fitted = True
        return pd.DataFrame(scaled, columns=self.feature_names, index=df.index)

    def _calculate_time_since_market_open(self, timestamp: datetime) -> float:
        """Calculate minutes since market open.

        Args:
            timestamp: Article publication timestamp

        Returns:
            Minutes since market open, or negative if before market open
        """
        article_time = timestamp.time()
        market_open_minutes = MARKET_OPEN.hour * 60 + MARKET_OPEN.minute
        article_minutes = article_time.hour * 60 + article_time.minute

        return float(article_minutes - market_open_minutes)

    def _calculate_return(self, closes: np.ndarray, periods: int) -> float:
        """Calculate return over specified periods.

        Args:
            closes: Array of closing prices
            periods: Number of periods for return calculation

        Returns:
            Percentage return, or 0.0 if insufficient data
        """
        if len(closes) <= periods:
            return 0.0

        current = closes[-1]
        previous = closes[-(periods + 1)]

        if previous == 0:
            return 0.0

        return float((current - previous) / previous * 100)

    def _calculate_volatility(self, closes: np.ndarray, window: int) -> float:
        """Calculate volatility as standard deviation of returns.

        Args:
            closes: Array of closing prices
            window: Lookback window

        Returns:
            Volatility (std of returns), or 0.0 if insufficient data
        """
        if len(closes) < 2:
            return 0.0

        # Use available data up to window size
        effective_window = min(window, len(closes))
        recent_closes = closes[-effective_window:]

        # Calculate log returns
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = np.diff(np.log(recent_closes))
            returns = returns[np.isfinite(returns)]

        if len(returns) == 0:
            return 0.0

        return float(np.std(returns))

    def _calculate_volume_ratio(self, volumes: np.ndarray, window: int) -> float:
        """Calculate current volume vs average volume.

        Args:
            volumes: Array of volumes
            window: Lookback window for average

        Returns:
            Ratio of current volume to average, or 1.0 if insufficient data
        """
        if len(volumes) == 0:
            return 1.0

        effective_window = min(window, len(volumes))
        avg_volume = np.mean(volumes[-effective_window:])

        if avg_volume == 0:
            return 1.0

        current_volume = volumes[-1]
        return float(current_volume / avg_volume)

    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index.

        Args:
            closes: Array of closing prices
            period: RSI period (default 14)

        Returns:
            RSI value (0-100), or 50.0 if insufficient data
        """
        if len(closes) < period + 1:
            return 50.0  # Neutral RSI for insufficient data

        # Calculate price changes
        deltas = np.diff(closes)

        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Calculate average gains and losses
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _calculate_ma_distance(self, closes: np.ndarray, window: int) -> float:
        """Calculate distance from moving average as percentage.

        Args:
            closes: Array of closing prices
            window: Moving average window

        Returns:
            Percentage distance from MA, or 0.0 if insufficient data
        """
        if len(closes) == 0:
            return 0.0

        effective_window = min(window, len(closes))
        ma = np.mean(closes[-effective_window:])

        if ma == 0:
            return 0.0

        current = closes[-1]
        return float((current - ma) / ma * 100)

    def _get_default_price_features(self) -> dict[str, float]:
        """Return default price features for missing data.

        Returns:
            Dictionary with neutral/zero values for all price features
        """
        return {
            "return_1m": 0.0,
            "return_5m": 0.0,
            "return_10m": 0.0,
            "return_20m": 0.0,
            "volatility": 0.0,
            "volume_ratio": 1.0,
            "rsi_14": 50.0,
            "ma_distance": 0.0,
        }

    def _find_price_bars(
        self,
        timestamp: datetime,
        prices: dict[datetime, list[PriceBar]],
        window: int,
    ) -> list[PriceBar]:
        """Find price bars matching or closest to timestamp.

        Args:
            timestamp: Target timestamp
            prices: Dictionary of timestamp to price bars
            window: Number of bars to return

        Returns:
            List of price bars, or empty list if none found
        """
        if not prices:
            return []

        # Normalize timestamp to naive UTC for comparison
        target_ts = self._normalize_timestamp(timestamp)

        # Try exact match first
        if timestamp in prices:
            return prices[timestamp][-window:]

        # Find closest timestamp before the article
        available_times = sorted(prices.keys())
        closest_time = None

        for t in available_times:
            t_normalized = self._normalize_timestamp(t)
            if t_normalized <= target_ts:
                closest_time = t
            else:
                break

        if closest_time is None:
            # No data before this timestamp, use earliest available
            closest_time = available_times[0] if available_times else None

        if closest_time is None:
            return []

        return prices[closest_time][-window:]

    def _normalize_timestamp(self, ts: datetime) -> datetime:
        """Normalize timestamp to naive UTC for comparison.

        Args:
            ts: Timestamp to normalize

        Returns:
            Naive datetime in UTC
        """
        if ts.tzinfo is not None:
            # Convert to UTC and remove tzinfo
            utc_ts = ts.astimezone(UTC)
            return utc_ts.replace(tzinfo=None)
        return ts

    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data in the feature DataFrame.

        Strategy:
            - Numeric columns: Fill with median
            - Still missing after median: Fill with 0

        Args:
            df: DataFrame with potential missing values

        Returns:
            DataFrame with missing values handled
        """
        if df.empty:
            return df

        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df[col] = df[col].fillna(median_val)

        # Fill any remaining NaN with 0
        df = df.fillna(0.0)

        # Replace infinite values
        df = df.replace([np.inf, -np.inf], 0.0)

        return df
