from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from trading_bot.data.models import MarketRegime
from trading_bot.features.regime import heuristic_regime


class RegimeClassifier:
    """
    XGBoost-based market regime classifier.

    Classifies the current market into one of:
      - trending
      - ranging
      - volatile
      - breakout

    Falls back to heuristic_regime() when the model is not yet trained.
    """

    REGIME_LABELS = {
        MarketRegime.TRENDING: 0,
        MarketRegime.RANGING: 1,
        MarketRegime.VOLATILE: 2,
        MarketRegime.BREAKOUT: 3,
    }
    LABEL_TO_REGIME = {v: k for k, v in REGIME_LABELS.items()}

    def __init__(self, model_path: str = "models/regime_classifier.pkl"):
        self.model_path = Path(model_path)
        self._model = None
        self._feature_names: List[str] = []
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if self.model_path.exists():
            try:
                import joblib
                data = joblib.load(self.model_path)
                self._model = data["model"]
                self._feature_names = data["feature_names"]
                logger.info("Loaded regime classifier from {}", self.model_path)
            except Exception as exc:
                logger.warning("Could not load regime classifier: {}", exc)

    def predict(self, features: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """
        Predict the market regime and return (regime, confidence).
        Falls back to heuristic if model not trained.
        """
        if self._model is None:
            regime = heuristic_regime(features)
            return regime, 0.6

        try:
            feature_vector = self._build_vector(features)
            proba = self._model.predict_proba([feature_vector])[0]
            label = int(np.argmax(proba))
            confidence = float(proba[label])
            regime = self.LABEL_TO_REGIME.get(label, MarketRegime.RANGING)
            return regime, confidence
        except Exception as exc:
            logger.warning("Regime prediction failed: {}. Using heuristic.", exc)
            return heuristic_regime(features), 0.5

    def _build_vector(self, features: Dict[str, float]) -> List[float]:
        if self._feature_names:
            return [features.get(f, 0.0) for f in self._feature_names]
        return list(features.values())

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save: bool = True,
    ) -> "RegimeClassifier":
        """Train the XGBoost classifier."""
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.error("xgboost not installed. Run: pip install xgboost")
            return self

        self._feature_names = list(X.columns)
        self._model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
        )
        self._model.fit(X, y)
        logger.info("Regime classifier trained on {} samples", len(X))

        if save:
            self._save()
        return self

    def _save(self) -> None:
        import joblib
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self._model, "feature_names": self._feature_names},
            self.model_path,
        )
        logger.info("Regime classifier saved to {}", self.model_path)

    @property
    def is_trained(self) -> bool:
        return self._model is not None
