from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from trading_bot.data.models import MarketRegime
from trading_bot.data.storage import Storage
from trading_bot.features.indicators import compute_indicators
from trading_bot.features.regime import build_regime_features, heuristic_regime
from trading_bot.ml.regime_classifier import RegimeClassifier


class Trainer:
    """
    Handles periodic retraining of the regime classifier using stored candle data.

    Labelling strategy:
      - For each candle window, extract regime features
      - Use heuristic_regime() to generate soft labels for training
      - Over time, labels can be replaced with outcome-based labels
        (e.g. which strategy actually performed best in that window)
    """

    def __init__(
        self,
        storage: Storage,
        classifier: RegimeClassifier,
        min_samples: int = 500,
        lookback_candles: int = 100,
    ):
        self.storage = storage
        self.classifier = classifier
        self.min_samples = min_samples
        self.lookback_candles = lookback_candles
        self._last_trained: Optional[datetime] = None

    def build_training_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 5000,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build a feature matrix X and label series y from stored candle history.
        """
        candles = self.storage.load_candles(symbol, timeframe, limit=limit)
        if len(candles) < self.min_samples:
            logger.warning(
                "Not enough candles for training: {} < {}", len(candles), self.min_samples
            )
            return pd.DataFrame(), pd.Series(dtype=int)

        df = pd.DataFrame(
            [
                {
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in candles
            ]
        ).set_index("timestamp")

        df = compute_indicators(df)

        rows: List[dict] = []
        labels: List[int] = []

        for i in range(self.lookback_candles, len(df)):
            window = df.iloc[i - self.lookback_candles: i + 1]
            features = build_regime_features(window, lookback=self.lookback_candles)
            if not features:
                continue
            regime = heuristic_regime(features)
            rows.append(features)
            labels.append(RegimeClassifier.REGIME_LABELS[regime])

        if not rows:
            return pd.DataFrame(), pd.Series(dtype=int)

        X = pd.DataFrame(rows).fillna(0.0)
        y = pd.Series(labels, name="regime")
        return X, y

    def train(
        self,
        symbol: str,
        timeframe: str,
        force: bool = False,
        retrain_interval_hours: int = 24,
    ) -> bool:
        """
        Train (or retrain) the regime classifier.
        Returns True if training was performed.
        """
        if not force and self._last_trained is not None:
            age = datetime.utcnow() - self._last_trained
            if age < timedelta(hours=retrain_interval_hours):
                logger.debug("Skipping training, last trained {} ago", age)
                return False

        logger.info("Starting regime classifier training for {} {}", symbol, timeframe)
        X, y = self.build_training_data(symbol, timeframe)

        if X.empty:
            logger.warning("No training data available, skipping")
            return False

        self.classifier.fit(X, y, save=True)
        self._last_trained = datetime.utcnow()
        logger.info("Training complete. {} samples used.", len(X))
        return True

    def should_retrain(self, interval_hours: int = 24) -> bool:
        if self._last_trained is None:
            return True
        return datetime.utcnow() - self._last_trained >= timedelta(hours=interval_hours)
